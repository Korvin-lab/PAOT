"""Preserve prepared chemistry and unknown auxiliary values without global placeholders."""
from pathlib import Path
import argparse,json
from collections import Counter
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def build_positive_series_lookup(source, dates_by_id, column):
 out=[]
 for pid, dates in sorted(dates_by_id.items()):
  dates=pd.DatetimeIndex(pd.to_datetime(sorted(dates),format='%Y-%m-%d',errors='raise'))
  part=source[source.id.eq(pid)].copy()
  part['date']=pd.to_datetime(part['date'],format='%Y-%m-%d',errors='raise')
  s=part.drop_duplicates('date').set_index('date')[column].astype(float)
  s=s.where(s>0).dropna()
  if s.empty:
   continue
  idx=dates.union(s.index).sort_values()
  filled=s.reindex(idx).interpolate(method='time',limit_area='inside').bfill().ffill().reindex(dates)
  out.append(pd.DataFrame({'id':pid,'date':dates.strftime('%Y-%m-%d'),column:filled.to_numpy(float)}))
 return pd.concat(out,ignore_index=True) if out else pd.DataFrame(columns=['id','date',column])

def finalize(source,destination,chem_daily_csv):
 config=json.loads((ROOT/'run_config.json').read_text())
 chem=pd.read_csv(chem_daily_csv,dtype={'id простого участка':str})
 chem=chem.rename(columns={'id простого участка':'id','Дата контроля':'date'})
 chem['date']=pd.to_datetime(chem['date'],format='%Y-%m-%d',errors='raise').dt.strftime('%Y-%m-%d')
 assert not chem.duplicated(['id','date']).any()
 lookup=chem.set_index(['id','date'])
 dates_by_id=chem.groupby('id')['date'].apply(list).to_dict()
 h2s_water_lookup=None
 two_phase_path=ROOT/'input/h2s_two_phase_daily_base.csv'
 if two_phase_path.exists():
  h2s_base=pd.read_csv(two_phase_path,dtype={'id':str,'date':str})
  if {'id','date','H2S in Water Phase'} <= set(h2s_base.columns):
   h2s_water_lookup=build_positive_series_lookup(h2s_base,dates_by_id,'H2S in Water Phase').set_index(['id','date'])
 aux_path=ROOT/'input/auxiliary_filled.csv'
 aux=pd.read_csv(aux_path,dtype={'id':str,'date':str}).set_index(['id','date']) if aux_path.exists() else None
 if aux is not None:assert not aux.index.duplicated().any()
 destination=Path(destination);destination.parent.mkdir(exist_ok=True,parents=True)
 temp=destination.with_suffix('.partial.csv');counts=Counter();ids=set();days=set();missing=Counter();first=True
 for chunk in pd.read_csv(source,chunksize=250000,dtype={'id':str,'date':str}):
  chunk['date']=pd.to_datetime(chunk['date'],format='%Y-%m-%d',errors='raise').dt.strftime('%Y-%m-%d')
  index=pd.MultiIndex.from_frame(chunk[['id','date']]);aligned=lookup.reindex(index)
  for target,original in [('CO2 in Water Phase','CO2'),('Min','Общая минерализация, г/л'),('pH','pH')]:
   expected=aligned[original].to_numpy(float);actual=pd.to_numeric(chunk[target],errors='coerce').to_numpy(float)
   if not np.allclose(actual,expected,rtol=1e-8,atol=1e-10,equal_nan=False):raise AssertionError('Prepared chemistry changed: '+target)
  chunk['temperature']=chunk['seg_temperature']
  if h2s_water_lookup is not None:
   chunk['H2S in Water Phase']=h2s_water_lookup.reindex(index)['H2S in Water Phase'].to_numpy(float)
  else:
   if not config.get('h2s_water_phase_confirmed') or config.get('h2s_to_mg_l_factor') is None:
    raise ValueError('Missing input/h2s_two_phase_daily_base.csv and no confirmed fallback H2S conversion in run_config.json')
   factor=float(config['h2s_to_mg_l_factor'])
   if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid H2S fallback conversion factor')
   chunk['H2S in Water Phase']=aligned['H2S_UKK_source'].to_numpy(float)*factor
  if not (pd.to_numeric(chunk['CO2 in Water Phase'],errors='coerce')>0).all():raise AssertionError('CO2 in Water Phase contains zero/negative values')
  water_h2s=pd.to_numeric(chunk['H2S in Water Phase'],errors='coerce')
  if (water_h2s.notna() & (water_h2s<=0)).any():raise AssertionError('H2S in Water Phase contains zero/negative values')
  if not (pd.to_numeric(chunk['Min'],errors='coerce')>0).all():raise AssertionError('Min contains zero/negative values')
  if aux is not None and 'ing_factor' in chunk:
   old=pd.to_numeric(chunk['ing_factor'],errors='coerce')
   chunk['ing_factor']=old.where(old.ge(0),pd.Series(aux.reindex(index)['ing_factor'].to_numpy(),index=chunk.index))
  if 'ing_sum' in chunk:chunk=chunk.drop(columns='ing_sum')
  for col in ['temperature','seg_temperature']:
   if not chunk[col].between(5,90).all():raise AssertionError('Invalid '+col)
  if not (chunk.seg_p_start>.1).all():raise AssertionError('Pressure <=0.01 MPa')
  numeric=chunk.select_dtypes(include=[np.number]);assert not np.isinf(numeric.to_numpy()).any()
  core=['id','date','segment_id','seg_p_start','seg_temperature','CO2 in Water Phase','Min','pH']
  if chunk[core].isna().any().any():raise AssertionError('Missing required prepared source or hydraulic result')
  missing.update({c:int(v) for c,v in chunk.isna().sum().items() if v})
  ids.update(chunk.id);days.update(zip(chunk.id,chunk.date));counts['rows']+=len(chunk)
  chunk.to_csv(temp,index=False,mode='w' if first else 'a',header=first,encoding='utf-8-sig' if first else 'utf-8');first=False
 if first:raise AssertionError('Empty strict result')
 expected_ids=set(config['expected_pipe_ids'])
 if ids!=expected_ids:raise AssertionError('Not all requested pipes calculated: '+str(sorted(expected_ids-ids)))
 temp.replace(destination)
 result={'status':'COMPLETE_WITH_MISSING_VALUES' if missing else 'PASS','rows':counts['rows'],'pipes':len(ids),'id_dates':len(days),'missing_by_column':dict(missing),'chemistry_identity_checked':True,'h2s_water_source':'No approved target-pipe water-phase source: NaN preserved; gas phase is not converted.','h2s_water_unit':'mg/l','unknown_values_not_replaced_with_global_mean_or_zero':True}
 (destination.parent/'FINAL_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--chem-daily-csv',required=True);a=p.parse_args();finalize(a.source,a.destination,a.chem_daily_csv)
