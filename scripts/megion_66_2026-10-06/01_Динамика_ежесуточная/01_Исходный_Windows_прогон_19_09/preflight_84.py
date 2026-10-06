from pathlib import Path
import argparse,json,math,importlib.util,sys,struct
from datetime import date
from collections import Counter
import pandas as pd
ROOT=Path(__file__).resolve().parent

def inspect(require_ready=False):
 cfg=json.loads((ROOT/'run_config.json').read_text());req=json.loads((ROOT/'input/graph_orenburg_requested_daily_params.json').read_text());g=json.loads((ROOT/'input/graph_orenburg_master.json').read_text())
 ids=set(req['by_id']);assert ids==set(cfg['expected_pipe_ids'])=={str(n['id']) for n in g['nodes']}
 chem=pd.read_csv(ROOT/'input/chem_daily_orenburg.csv',dtype={'id простого участка':str});assert not chem.duplicated(['id простого участка','Дата контроля']).any()
 assert set(chem['id простого участка'])==ids
 assert chem[['CO2','pH','Общая минерализация']].notna().all().all()
 assert (pd.to_numeric(chem['CO2'],errors='coerce')>0).all()
 if 'H2S_UKK_source' in chem.columns:
  assert (pd.to_numeric(chem['H2S_UKK_source'],errors='coerce')>0).all()
 assert chem.pH.between(0,10,inclusive='right').all()
 assert (chem['Общая минерализация']>0).all()
 assert ((chem['Общая минерализация, г/л']*1000-chem['Общая минерализация']).abs()<1e-7).all()
 h2s_base_path=ROOT/'input/h2s_two_phase_daily_base.csv'
 if not h2s_base_path.exists():
  raise AssertionError('Missing input/h2s_two_phase_daily_base.csv with separate H2S water/gas columns')
 h2s=pd.read_csv(h2s_base_path,dtype={'id':str,'date':str})
 assert {'id','date','CO2 in Water Phase','H2S in Water Phase','H2S in Gas Phase'} <= set(h2s.columns)
 for col in ['CO2 in Water Phase','H2S in Water Phase','H2S in Gas Phase']:
  values=pd.to_numeric(h2s[col],errors='coerce')
  if values.isna().any() or not (values>0).all():
   raise AssertionError(f'{col} contains missing/zero/non-positive values after positive fill')
  pos=h2s[values>0].groupby('id').size()
  missing=sorted(ids-set(pos.index))
  if missing:raise AssertionError(f'{col} has no positive source rows for pipes: {missing}')
 spec=importlib.util.spec_from_file_location('pipe84',ROOT/'main_pipeline_final_csv.py');pipeline=importlib.util.module_from_spec(spec);sys.modules['pipe84']=pipeline;spec.loader.exec_module(pipeline)
 active=ready=0;reasons=Counter();pipe_reports=[];bad_rows=[];reverse_bad=Counter()
 for pid,payload in req['by_id'].items():
  ds=payload['daily'];seen=set();ok=ac=0
  boundaries=pipeline.assign_temperature_boundaries(pd.DataFrame({'date':[r['Дата'] for r in ds],'source_t':[r['source_t'] for r in ds]}))
  sides=dict(zip(boundaries.date,boundaries.temperature_boundary_side))
  for r in ds:
   day=r['Дата'];assert date.fromisoformat(day).isoformat()==day;assert day not in seen;seen.add(day)
   assert 5<=r['t']<=90 and 0<r['p']<=4
   def num(k):
    try:v=float(r.get(k));return v if math.isfinite(v) else None
    except (TypeError,ValueError):return None
   ql,qo,qg=[num(k) for k in ['Жидкости, м3/сут (дебит)','Нефти, т/сут (дебит)','Общего газа, тыс.м3/сут (дебит)']]
   if not any(v is not None and v>0 for v in [ql,qo,qg]):continue
   active+=1;ac+=1;fail=[]
   for key in ['Жидкости, м3/сут (дебит)','Нефти, т/сут (дебит)','Нефти, кг/м3','Жидкости, кг/м3','Газа, кг/м3','D','S','L']:
    if num(key) is None or num(key)<=0:fail.append(key)
   if r.get('dns_paot_pipe_kind')!='naporny' and (qg is None or qg<=0):fail.append('Qgas')
   wc=num('Обводненность, %')
   if wc is None or not 0<=wc<=100:fail.append('watercut')
   if r['p']<=.01:fail.append('P<=0.01 MPa')
   if not fail and sides.get(day)=='end':
    D=num('D');D=D/1000 if D>1.5 else D
    row=pd.Series({'t':r['t'],'p':r['p'],'D':D,'q_liq':ql,'q_oil':qo,'q_gas':qg,'rho_gas':num('Газа, кг/м3'),'watercut':wc})
    est=pipeline.estimate_reverse_temperature_start(row,num('L'))
    if est is None or not math.isfinite(est) or not 5<=est<=90:fail.append('reverse_temperature_infeasible');reverse_bad[pid]+=1
   if fail:reasons.update(fail);bad_rows.append({'id':pid,'date':day,'reasons':'; '.join(fail)})
   else:ready+=1;ok+=1
  pipe_reports.append({'ID трубы':pid,'Исходных дат':len(ds),'Активных дат':ac,'Предварительно пригодных дат':ok,'Предварительное покрытие активных дат':ok/ac if ac else 0})
 blocker=[]
 if require_ready and (sys.platform!='win32' or struct.calcsize('P')!=8):blocker.append('Требуется Windows и Python x64 для PE2 DLL')
 if any(p['Предварительно пригодных дат']==0 for p in pipe_reports):blocker.append('Есть трубы без пригодных активных дат')
 report={'status':'BLOCKED' if blocker else 'PASS_WITH_EXCLUSIONS' if active!=ready else 'PASS','pipes':len(ids),'active_dates':active,'forecast_ready_dates':ready,'excluded_dates':active-ready,'reasons_overlap':dict(reasons),'reverse_temperature_rejected':dict(reverse_bad),'blockers':blocker,'note':'Forecast only. Full Windows PE2 not run. No auxiliary values manufactured to pass a mask.'}
 (ROOT/'PREFLIGHT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame(pipe_reports).to_excel(ROOT/'Предварительная_готовность_84_труб.xlsx',index=False)
 pd.DataFrame(bad_rows,columns=['id','date','reasons']).to_csv(ROOT/'input/excluded_active_dates_forecast.csv',index=False,encoding='utf-8-sig')
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
 if require_ready and blocker:raise SystemExit(2)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--require-ready',action='store_true');a=p.parse_args();inspect(a.require_ready)
