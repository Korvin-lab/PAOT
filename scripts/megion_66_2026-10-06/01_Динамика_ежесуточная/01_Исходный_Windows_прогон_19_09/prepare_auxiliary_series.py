"""Fill viscosity and inhibition only from the same pipe's own observations."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from fill_rules import fill_daily
from build_final_dataset_stage1_co2 import load_dosage_lookup

ROOT=Path(__file__).resolve().parent

def main():
    req=json.loads((ROOT/'input/graph_orenburg_requested_daily_params.json').read_text())
    vis=pd.read_csv(ROOT/'input/chemistry_sources/viscosity_original.csv',dtype={'id':str})
    vis['date']=pd.to_datetime(vis.date,format='%Y-%m-%d')
    vis=vis[vis.mu_liq.gt(0)].groupby(['id','date']).mu_liq.mean()
    dose=load_dosage_lookup([ROOT/'input/dosage_orenburg_normalized.xlsx'])
    dose['date']=pd.to_datetime(dose.date,format='%Y-%m-%d')
    dose=dose[dose.ing_factor.ge(0)].set_index(['id','date']).ing_factor
    rows=[];counts=[]
    for pid,payload in req['by_id'].items():
        dates=pd.DatetimeIndex([r['Дата'] for r in payload['daily']]);r=pd.DataFrame({'id':pid,'date':dates.strftime('%Y-%m-%d')})
        count={'id':pid,'runtime_dates':len(dates)}
        for par,source in [('mu_liq',vis),('ing_factor',dose)]:
            if pid not in source.index.get_level_values(0):r[par]=np.nan;count[par+'_filled_dates']=0;count[par+'_original_dates']=0;continue
            s=source.xs(pid,level=0).dropna();span=pd.date_range(min(dates.min(),s.index.min()),max(dates.max(),s.index.max()))
            filled,tags=fill_daily(s.reindex(span));r[par]=filled.reindex(dates).to_numpy()
            count[par+'_filled_dates']=int(r[par].notna().sum());count[par+'_original_dates']=int(s.index.isin(dates).sum())
            assert np.allclose(filled.reindex(s.index),s,rtol=0,atol=0)
        rows.append(r);counts.append(count)
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(ROOT/'input/auxiliary_filled.csv',index=False,encoding='utf-8-sig')
    result[['id','date','mu_liq']].to_csv(ROOT/'input/visc_daily_orenburg.csv',index=False,encoding='utf-8-sig')
    report={'rule':'Same-pipe observations only; <6 month interpolation, seasonal analogue, same-pipe mean; no cross-pipe/global/zero fallback',
            'pipes_with_viscosity':sum(r['mu_liq_filled_dates']>0 for r in counts),
            'pipes_with_ing_factor':sum(r['ing_factor_filled_dates']>0 for r in counts),'by_pipe':counts}
    (ROOT/'AUXILIARY_FILL_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print({k:v for k,v in report.items() if k!='by_pipe'})

if __name__=='__main__':main()
