from collections import defaultdict, Counter
from datetime import datetime, date
from pathlib import Path
import importlib.util, json, re, math
import xlrd, openpyxl
ROOT=Path('/Volumes/KINGSTON/Газпром работа/Прогнозирование утонения стенки трубы')
DATA=ROOT/'Данные для сбора датасета Мегион'; PKG=ROOT/'работа с json/MEGION_WINDOWS_FULL_PE2__PREPARING_2026-09-18'
spec=importlib.util.spec_from_file_location('bmi',PKG/'build_megion_windows_inputs.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);pot=mod.load_pot()
targets=set(json.loads((PKG/'run_config.json').read_text())['expected_pipe_ids']);dirs={pot[p]['direction_id'] for p in targets}
def id(v):
 try:return str(int(float(v)))
 except:return ''
def day(v,mode='xls'):
 if isinstance(v,datetime):return v.date().isoformat()
 if isinstance(v,date):return v.isoformat()
 if isinstance(v,(int,float)):return xlrd.xldate.xldate_as_datetime(v,0).date().isoformat()
 s=str(v).strip()
 for fmt in ('%d.%m.%Y','%Y-%m-%d','%d.%m.%Y %H:%M:%S'):
  try:return datetime.strptime(s,fmt).date().isoformat()
  except ValueError:pass
 return ''
def num(v):
 try:
  x=float(str(v).replace(',','.'))
  return x if math.isfinite(x) else None
 except:return None
def norm(v):return re.sub(r'[^a-zа-я0-9]','',str(v).lower().replace('ё','е'))
well_dirs=defaultdict(set);well_fields=defaultdict(set);well_source_days=defaultdict(lambda:defaultdict(set));c=Counter();src_examples=[]
for f in sorted((DATA/'Список источников').glob('*.xls')):
 if f.name.startswith('._'):continue
 w=xlrd.open_workbook(str(f))
 for sh in w.sheets():
  for rn in range(1,sh.nrows):
   r=sh.row_values(rn);wid=id(r[10]);pid=id(r[0]);dt=day(r[3]);
   if not wid or not pid or not dt or pid not in pot:continue
   dr=pot[pid]['direction_id'];well_dirs[wid].add(dr);well_fields[wid].add(norm(r[7]));well_source_days[wid][dr].add(dt)
   c['source_well_rows']+=1
   if dr in dirs:c['target_direction_source_rows']+=1
   if len(src_examples)<3 and dr in dirs:src_examples.append((f.name,sh.name,rn+1,pid,dr,wid,dt))
 print('SOURCE',f.name,'sheets',w.nsheets,flush=True)
fhs=defaultdict(list);fhsfields=defaultdict(set);w=openpyxl.load_workbook(DATA/'ФХС 2015-2026 MEGION.xlsx',read_only=True,data_only=True)
for rn,r in enumerate(w.active.iter_rows(min_row=2,values_only=True),2):
 wid=id(r[0]);v=num(r[33]);dt=day(r[6]);
 if not wid or v is None or v<=0 or not dt:continue
 fhs[wid].append((dt,v,rn));fhsfields[wid].add(norm(r[1]));c['positive_fhs']+=1
w.close()
matched=set(fhs)&set(well_dirs);tmatched={wid for wid in matched if well_dirs[wid]&dirs}
c['all_positive_wells']=len(fhs);c['all_matched_wells']=len(matched);c['target_direction_wells']=len(tmatched)
c['target_direction_samples']=sum(len(fhs[w]) for w in tmatched)
c['multi_direction_wells_target']=sum(len(well_dirs[w]&dirs)>1 for w in tmatched)
c['field_mismatch_target']=sum(not (well_fields[w]&fhsfields[w]) for w in tmatched)
for wid in tmatched:
 for d,v,_ in fhs[wid]:
  if any(d in well_source_days[wid][dr] for dr in well_dirs[wid]&dirs):c['same_day_source_sample']+=1
  if d<'2024-01-01':c['pre_2024_samples']+=1
bydir={dr:{'wells':sum(dr in well_dirs[w] for w in fhs),'samples':sum(len(fhs[w]) for w in fhs if dr in well_dirs[w]),'target_pipes':sorted(p for p in targets if pot[p]['direction_id']==dr)} for dr in dirs}
print('COUNTS',json.dumps(c,ensure_ascii=False),flush=True);print('DIRECTIONS',json.dumps(bydir,ensure_ascii=False),flush=True);print('EXAMPLES',src_examples,flush=True)
out={'counts':dict(c),'directions':bydir,'source_examples':src_examples,'well_direction_links':{w:sorted(well_dirs[w]&dirs) for w in tmatched}}
(Path(__file__).parent/'link_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
