"""Rebuild chemistry from dated UKK observations and monthly digitized UKK reports."""
from pathlib import Path
from collections import defaultdict, Counter, deque
import json, re, math
import pandas as pd
import numpy as np
import openpyxl
import xlrd
from fill_rules import fill_daily

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'input/chemistry_sources'
PARAMS = {'CO2_mg_l': 'CO2, мг/л', 'pH': 'рН', 'Min_mg_l': 'Минерализация, мг/л',
          'H2S_water_mg_l': 'H2S, мг/л', 'kvch_mg_l': 'КВЧ, мг/л'}
POSITIVE_CHEMISTRY = {'CO2_mg_l', 'H2S_water_mg_l', 'Min_mg_l'}
LINEAR_NEAREST_PARAMS = {'CO2_mg_l', 'H2S_water_mg_l'}

def number(v):
    if v is None: return np.nan
    s = str(v).strip().replace('\u00a0', '').replace(' ', '').replace(',', '.')
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?', s): return np.nan
    f = float(s)
    return f if math.isfinite(f) else np.nan

def field(v):
    s = str(v).strip().lower().replace('ё', 'е')
    if 'ву онгкм' in s or s == 'оренбургское': return 'оренбургское'
    if s in ('царичанское', 'царичанское+филатовское'): return 'царичанское+филатовское'
    return {'новозаринсое': 'новозаринское', 'ц.уранское': 'центрально-уранское'}.get(s,s)

def point(v):
    s = str(v).strip()
    return str(int(float(s))) if re.fullmatch(r'\d+(?:\.0+)?',s) else s

def distances(adj, start):
    result = {start: 0}; q = deque([start])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in result: result[v] = result[u]+1; q.append(v)
    return result

def fill_linear_nearest(series):
    s = series.sort_index().astype(float)
    if not s.index.is_unique:
        raise ValueError('Duplicate dates')
    labels = pd.Series('missing', index=s.index, dtype=object)
    observed = s.dropna()
    if observed.empty:
        return s, labels
    labels.loc[observed.index] = 'observed'
    filled = s.interpolate(method='time', limit_area='inside')
    interpolated = s.isna() & filled.notna()
    labels.loc[interpolated] = 'time_interpolation'
    before_first = s.index < observed.index.min()
    after_last = s.index > observed.index.max()
    filled = filled.bfill().ffill()
    labels.loc[before_first & s.isna()] = 'nearest_first_value_at_left_edge'
    labels.loc[after_last & s.isna()] = 'nearest_last_value_at_right_edge'
    assert np.allclose(filled.reindex(observed.index), observed, rtol=0, atol=0)
    return filled, labels

def main():
    cfg=json.loads((ROOT/'run_config.json').read_text()); selected=cfg['expected_pipe_ids']
    graph=json.loads((SRC/'graph_332.json').read_text()); nodes={str(n['id']):n['graph'] for n in graph['nodes']}
    dirs=json.loads((SRC/'directions.json').read_text()); adj=defaultdict(set)
    for e in graph['edges']:
        if e['verification_status'] in {'confirmed_by_physical_node_id','name_match_same_field_context'}:
            u,v=str(e['source']),str(e['target']);adj[u].add(v);adj[v].add(u)
    book=xlrd.open_workbook(str(SRC/'ukk.xls')); mapping=defaultdict(set)
    for sheet in book.sheets():
        if sheet.nrows == 0: continue
        h=sheet.row_values(0)
        if not all(k in h for k in ['Месторождение','Номер точки','ID простого участка']): continue
        for i in range(1,sheet.nrows):
            pid=point(sheet.cell_value(i,h.index('ID простого участка')))
            mapping[field(sheet.cell_value(i,h.index('Месторождение'))),point(sheet.cell_value(i,h.index('Номер точки')))].add(pid)
    wb=openpyxl.load_workbook(SRC/'ngs_digitized.xlsx',read_only=True,data_only=True)
    rows=[]; audit=[]; sheet_audit=[]
    for ws in wb:
        values=list(ws.values)
        sheet_audit.append({'sheet':ws.title,'rows':len(values),'nonempty_cells':sum(v is not None for r in values for v in r)})
        if ws.title!='Данные': continue
        headers=values[0]
        for i,vs in enumerate(values[1:],2):
            r=dict(zip(headers,vs)); key=(field(r['месторождение']),point(r['УКК №']))
            candidates=mapping.get(key,set()); match=next(iter(candidates)) if len(candidates)==1 else None
            status='сопоставлено' if match in nodes else 'вне 332 труб' if match else 'неоднозначно' if candidates else 'нет привязки УКК'
            month=str(r['дата отчета']); assert re.fullmatch(r'20\d\d-\d\d',month)
            vals={p:number(r[col]) for p,col in PARAMS.items()}
            for p,v in list(vals.items()):
                if (
                    not np.isfinite(v)
                    or (p == 'pH' and not 0 < v <= 10)
                    or (p in POSITIVE_CHEMISTRY and v <= 0)
                    or (p not in {'pH', *POSITIVE_CHEMISTRY} and v < 0)
                ):
                    vals[p]=np.nan
            audit.append({'Строка':i,'Месяц':month,'Месторождение':r['месторождение'],'УКК':r['УКК №'],'Направление отчета':r['направление'],'ID трубы':match,'Название трубы ПОТ':nodes.get(match,{}).get('simple_name'),'Статус':status,'Есть числовая химия':any(np.isfinite(v) for v in vals.values())})
            if status=='сопоставлено': rows.append({'id':match,'month':month,**vals})
    wb.close()
    monthly=pd.DataFrame(rows).groupby(['id','month'])[list(PARAMS)].mean().reset_index()
    monthly.to_csv(SRC/'ngs_monthly_mapped.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(audit).to_csv(SRC/'ngs_mapping_audit.csv',index=False,encoding='utf-8-sig')
    raw=pd.read_csv(SRC/'ukk_normalized.csv',dtype={'id':str})
    # Gas/oil H2S is not converted into water H2S in the current approved logic.
    raw['H2S_water_mg_l']=np.nan
    raw['date']=pd.to_datetime(raw['date'],format='%Y-%m-%d')
    # The phase and mg/l interpretation are explicitly confirmed by the user.
    raw['kvch_mg_l']=np.nan
    raw.loc[~raw.pH.between(0,10,inclusive='right'),'pH']=np.nan
    for col in POSITIVE_CHEMISTRY:
        if col in raw.columns:
            raw.loc[pd.to_numeric(raw[col], errors='coerce') <= 0, col] = np.nan
    obs=raw.groupby(['id','date'])[list(PARAMS)].mean()
    tech_kv=pd.read_csv(SRC/'kvch_technical_original.csv',dtype={'id':str})
    tech_kv['date']=pd.to_datetime(tech_kv['date'],format='%Y-%m-%d')
    tech_kv=tech_kv[tech_kv.kvch_mg_l.gt(0)].groupby(['id','date'])[['kvch_mg_l']].mean()
    obs=obs.combine_first(tech_kv)
    # Optional dated monthly tables: no gas-phase H2S is admitted as aqueous H2S.
    path=SRC/'additional_dated_chemistry.csv'
    if path.exists():
        more=pd.read_csv(path,dtype={'id':str});more['date']=pd.to_datetime(more['date'],format='%Y-%m-%d')
        for col in ['CO2_mg_l','Min_mg_l']:
            if col in more.columns:
                more.loc[pd.to_numeric(more[col], errors='coerce') <= 0, col] = np.nan
        if 'pH' in more.columns:
            more.loc[~pd.to_numeric(more['pH'], errors='coerce').between(0,10,inclusive='right'),'pH'] = np.nan
        more=more.groupby(['id','date'])[['CO2_mg_l','pH','Min_mg_l']].mean()
        obs=obs.combine_first(more)
    req=json.loads((ROOT/'input/graph_orenburg_requested_daily_params.json').read_text())
    output=[]; provenance=[]; cache={}; counts=Counter(); fill_provenance=[]
    for pid in selected:
        print('Preparing chemistry:',pid,flush=True)
        dates=pd.DatetimeIndex([r['Дата'] for r in req['by_id'][pid]['daily']])
        out=pd.DataFrame({'id простого участка':pid,'Дата контроля':dates.strftime('%Y-%m-%d')})
        dist=distances(adj,pid)
        for par in PARAMS:
            daily=obs[par].dropna(); mm=monthly.dropna(subset=[par])
            donors=set(daily.index.get_level_values(0))|set(mm.id)
            scores=Counter(daily.index.get_level_values(0));scores.update(mm.id)
            same=[d for d in donors if set(dirs.get(pid,[]))&set(dirs.get(d,[]))]
            pool=[d for d in donors if d in dist]
            if pid in donors: donor=pid;method='собственные данные'
            elif same: donor=min(same,key=lambda d:(-scores[d],d));method='направление ПОТ'
            elif pool: donor=min(pool,key=lambda d:(dist[d],-scores[d],d));method='граф в обе стороны'
            else: donor=None;method='нет источника'
            counts[par+' / '+method]+=1
            provenance.append({'ID трубы':pid,'Параметр':par,'ID донора':donor,'Способ':method,'Датированных значений':int((daily.index.get_level_values(0)==donor).sum()),'Месяцев НГС':int((mm.id==donor).sum())})
            if donor is None:out[par]=np.nan;continue
            key=(donor,par)
            if key not in cache:
                ds=daily.xs(donor,level=0) if donor in daily.index.get_level_values(0) else pd.Series(dtype=float)
                ms=mm[mm.id.eq(donor)]
                span=pd.date_range('2015-01-01','2026-08-27'); span=pd.DatetimeIndex(span.union(ds.index)).sort_values()
                series=ds.reindex(span); monthly_mask=pd.Series(False,index=span); span_months=span.strftime('%Y-%m')
                for r in ms.itertuples():
                    mask=(span_months==r.month)&series.isna().to_numpy()
                    series.loc[mask]=getattr(r,par);monthly_mask.loc[mask]=True
                if par in LINEAR_NEAREST_PARAMS:
                    filled,tags=fill_linear_nearest(series)
                else:
                    filled,tags=fill_daily(series)
                tags.loc[monthly_mask]='monthly_NGS_not_exact_sample_date'
                assert np.allclose(filled.reindex(ds.index),ds,rtol=0,atol=0)
                cache[key]=(filled,tags)
            filled,tags=cache[key];out[par]=filled.reindex(dates).to_numpy()
            counts[par+' / monthly_runtime_days']+=int((tags.reindex(dates)=='monthly_NGS_not_exact_sample_date').sum())
            fill_provenance.append({'id':pid,'parameter':par,'donor':donor,'counts':dict(Counter(tags.reindex(dates)))})
        output.append(out)
    result=pd.concat(output,ignore_index=True)
    result['CO2']=result.CO2_mg_l;result['Общая минерализация']=result.Min_mg_l
    result['Общая минерализация, г/л']=result.Min_mg_l/1000;result['H2S_UKK_source']=result.H2S_water_mg_l
    assert len(result)==235344 and not result.duplicated(['id простого участка','Дата контроля']).any()
    assert result[list(PARAMS)[:-1]].notna().all().all()
    assert (pd.to_numeric(result['CO2'], errors='coerce') > 0).all()
    assert (pd.to_numeric(result['H2S_UKK_source'], errors='coerce') > 0).all()
    assert (pd.to_numeric(result['Общая минерализация'], errors='coerce') > 0).all()
    result.to_csv(ROOT/'input/chem_daily_orenburg.csv',index=False,encoding='utf-8-sig')
    # Preserve direct technical-regime solids, then use the UKK-derived chemistry series.
    kv=pd.read_csv(SRC/'kvch_technical_original.csv',dtype={'id':str}).set_index(['id','date'])
    extra=result.rename(columns={'id простого участка':'id','Дата контроля':'date'}).set_index(['id','date'])[['kvch_mg_l']]
    kv=kv.combine_first(extra);kv.reset_index().to_csv(ROOT/'input/kvch_daily_orenburg.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(provenance).to_csv(ROOT/'input/chemistry_donors.csv',index=False,encoding='utf-8-sig')
    (ROOT/'input/chemistry_fill_provenance.json').write_text(json.dumps(fill_provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'ngs_sheets':sheet_audit,'ngs_rows':len(audit),'ngs_mapping':dict(Counter(r['Статус'] for r in audit)),'ngs_with_numeric_chemistry':sum(r['Есть числовая химия'] for r in audit),'coverage':dict(counts),'h2s_water_max_mg_l':float(result.H2S_water_mg_l.max()),'ngs_h2s_max_mg_l':float(monthly.H2S_water_mg_l.max()),'notes':['Current approved H2S logic keeps gas phase separate; gas/oil H2S is not converted into water H2S.','CO2 and H2S use time interpolation inside the row and nearest observed edge values outside it.','Month of report is not an exact sampling date.','Mapping uses unique field + UKK number from UKK register; ambiguous mappings excluded.','NGS data are UKK water-phase reports, not well chemistry.','Chemistry propagation is copying, not flow-weighted mixing.']}
    (ROOT/'CHEMISTRY_ALL_SOURCES_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
