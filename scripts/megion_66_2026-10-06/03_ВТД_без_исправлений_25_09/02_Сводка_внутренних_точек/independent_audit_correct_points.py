from pathlib import Path
import xlrd,re,json,gc
def clean(v):return re.sub(r'\s+',' ',str(v if v is not None else '').replace('\n',' ').strip())
def key(v):return re.sub(r'[^a-zа-яё0-9%]+','',clean(v).casefold())
def num(v):
 m=re.search(r'[-+]?\d+(?:[\.,]\d+)?',clean(v).replace('%',' '));return float(m.group().replace(',','.')) if m else None
def ident(v):
 s=clean(v)
 try:n=float(s.replace(',','.'));return str(int(n)) if n.is_integer() else s
 except:return s
raw=correct=0;pairs=set();details=[];bad=[]
for p in sorted(Path('.').glob('*.xls')):
 if p.name.startswith('._'):continue
 b=xlrd.open_workbook(str(p),on_demand=True)
 for sn in b.sheet_names():
  sh=b.sheet_by_name(sn);h={key(x):i for i,x in enumerate(sh.row_values(0))};n=ok=0
  for r in range(1,sh.nrows):
   row=sh.row_values(r);mid=ident(row[h[key('ID замера')]]);pipe=ident(row[h[key('ID простого участка')]])
   if not(mid or pipe):continue
   n+=1;pairs.add((mid,pipe));code=clean(row[h[key('Положение дефекта на стенке')]]);name=clean(row[h[key('Положение на стенке наим')]]).casefold();s=num(row[h[key('Толщина стенки элемента, мм')]]);d=num(row[h[key('Максимальная измер.глубина,%')]])
   if code=='XD0002' and name=='внутреннее' and s is not None and s>0 and d is not None and 0<d<=100 and 0<=s*(1-d/100)<s:ok+=1
   if (code=='XD0002') != (name=='внутреннее'):bad.append((p.name,sn,r+1,code,name))
  raw+=n;correct+=ok;details.append({'Файл':p.name,'Лист':sn,'Строк':n,'Корректных внутренних точек':ok});sh=None;gc.collect()
 b.release_resources();del b;gc.collect()
out={'status':'PASS' if raw==631692 and len(pairs)==631692 and correct==489779 and not bad else 'FAIL','raw_rows':raw,'unique_pairs':len(pairs),'correct_internal_points':correct,'conflicting_wall_labels':len(bad),'rule':'XD0002 и внутреннее; S>0; 0<глубина, %<=100; 0<=S*(1-глубина%/100)<S','per_sheet':details}
Path('analysis_correct_points/independent_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(out,ensure_ascii=False,indent=2))
