#!/usr/bin/env python3
"""Platform-independent integrity checks for prepared Megion Windows inputs."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
inp=ROOT/'input'
cfg=json.loads((ROOT/'run_config.json').read_text(encoding='utf-8'))
master=json.loads((inp/'graph_megion_master.json').read_text(encoding='utf-8'))
requested=json.loads((inp/'graph_megion_requested_daily_params.json').read_text(encoding='utf-8'))
ids=set(cfg['expected_pipe_ids'])
assert len(ids)==66, f'Prepared scope must contain 66 pipes; got {len(ids)}'
assert ids==set(requested['by_id'])=={str(n['id']) for n in master['nodes'] if str(n['id']) in ids}
assert float(cfg['temperature_model_surrounding_c'])==5.0
daily=[]; active_dates=0
for pid,payload in requested['by_id'].items():
 rows=payload['daily']; assert rows and len({r['Дата'] for r in rows})==len(rows)
 for r in rows:
  assert 5<float(r['t'])<=90 and .01<float(r['p'])<=9
  for c in ('L','D','S','Нефти, кг/м3','Жидкости, кг/м3','Газа, кг/м3'):
   assert r.get(c) is not None and float(r[c])>0
  assert r.get('Обводненность, %') is not None and 0 <= float(r['Обводненность, %']) <= 100
  flows=[float(r[c]) if r.get(c) is not None else 0.0 for c in ('Жидкости, м3/сут (дебит)','Нефти, т/сут (дебит)','Общего газа, тыс.м3/сут (дебит)')]
  if any(v>0 for v in flows): active_dates+=1
  daily.append((pid,r['Дата']))
assert len(daily)==len(set(daily))
chem=pd.read_csv(inp/'chem_daily_megion.csv',dtype={'id простого участка':str,'Дата контроля':str})
h2s=pd.read_csv(inp/'h2s_two_phase_daily_base.csv',dtype={'id':str,'date':str})
for d, columns in [(chem,['CO2','pH','Общая минерализация']), (h2s,['CO2 in Water Phase','H2S in Gas Phase'])]:
 assert not d.duplicated(d.columns[:2].tolist()).any()
 for c in columns: assert pd.to_numeric(d[c],errors='coerce').gt(0).all(),c
assert set(chem['id простого участка'])==ids==set(h2s['id'])
report={'status':'PASS_WITH_KNOWN_H2S_WATER_GAPS','prepared_pipes':66,'excluded_target_pipe':'1750004671','daily_pipe_dates':len(daily),'active_dates_before_strict_hydraulic_mask':active_dates,'graph_nodes':len(master['nodes']),'graph_edges':len(master['edges']),'chem_rows':len(chem),'two_phase_rows':len(h2s),'h2s_water_phase_missing_rows':int(h2s['H2S in Water Phase'].isna().sum()),'ambient_temperature_c':5.0}
(ROOT/'PREFLIGHT_MEGION.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
