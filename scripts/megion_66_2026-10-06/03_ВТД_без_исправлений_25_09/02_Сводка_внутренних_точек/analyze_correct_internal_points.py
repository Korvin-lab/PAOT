from pathlib import Path
import xlrd, sqlite3, re, json, gc
from datetime import datetime

ROOT=Path('.'); OUT=ROOT/'analysis_correct_points'; OUT.mkdir(exist_ok=True); DB=OUT/'correct_points.sqlite'
def clean(v):
 s=str(v if v is not None else '').replace('\n',' ').strip()
 while len(s)>1 and s[0]=='"' and s[-1]=='"':s=s[1:-1].strip()
 return re.sub(r'\s+',' ',s)
def key(v):return re.sub(r'[^a-zа-яё0-9%]+','',clean(v).casefold())
def ident(v):
 s=clean(v)
 try:
  n=float(s.replace(',','.'));return str(int(n)) if n.is_integer() else s
 except:return s
def num(v):
 s=clean(v).replace('\xa0',' ').replace(',','.').replace('%',' ');m=re.search(r'[-+]?\d+(?:\.\d+)?',s)
 return float(m.group()) if m else None
def dateval(b,c):
 try:return datetime(*xlrd.xldate_as_tuple(float(c.value),b.datemode)).date().isoformat()
 except:return clean(c.value)

con=sqlite3.connect(DB);con.executescript('''PRAGMA journal_mode=WAL;PRAGMA synchronous=NORMAL;
CREATE TABLE m(file TEXT,sheet TEXT,row_no INTEGER,dt TEXT,insp TEXT,pipe TEXT,measure TEXT,wall_code TEXT,wall_name TEXT,s REAL,depth_pct REAL,depth_mm REAL,PRIMARY KEY(measure,pipe));''')
raw=0;sources=[];header0=None
for p in sorted(ROOT.glob('*.xls')):
 if p.name.startswith('._'):continue
 b=xlrd.open_workbook(str(p),on_demand=True)
 for sn in b.sheet_names():
  sh=b.sheet_by_name(sn);h=sh.row_values(0);hm={key(x):i for i,x in enumerate(h)}
  names=['Дата освидетельствования','ID освидетельствования','ID простого участка','ID замера','Положение дефекта на стенке','Положение на стенке наим','Толщина стенки элемента, мм','Максимальная измер.глубина,%','Максимальная измер.глубина,мм']
  assert all(key(x) in hm for x in names),f'{p.name}/{sn}: нет обязательной колонки'
  batch=[];nr=0
  for r in range(1,sh.nrows):
   row=sh.row(r); pipe=ident(row[hm[key('ID простого участка')]].value);measure=ident(row[hm[key('ID замера')]].value);insp=ident(row[hm[key('ID освидетельствования')]].value)
   if not(pipe or measure or insp):continue
   batch.append((p.name,sn,r+1,dateval(b,row[hm[key('Дата освидетельствования')]]),insp,pipe,measure,clean(row[hm[key('Положение дефекта на стенке')]].value),clean(row[hm[key('Положение на стенке наим')]].value).casefold(),num(row[hm[key('Толщина стенки элемента, мм')]].value),num(row[hm[key('Максимальная измер.глубина,%')]].value),num(row[hm[key('Максимальная измер.глубина,мм')]].value)));nr+=1
   if len(batch)>=5000:con.executemany('INSERT OR IGNORE INTO m VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',batch);con.commit();batch=[]
  if batch:con.executemany('INSERT OR IGNORE INTO m VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',batch);con.commit()
  raw+=nr;sources.append({'Файл':p.name,'Лист':sn,'Строк':nr,'Колонок':sh.ncols,'Заголовки_совпадают':header0 is None or h==header0});header0=h if header0 is None else header0
  print(p.name,sn,nr,flush=True);b.unload_sheet(sn);gc.collect()
 b.release_resources();del b;gc.collect()

internal="wall_code='XD0002' AND wall_name='внутреннее'"
depth="depth_pct>0 AND depth_pct<=100"
nom="s>0"
residual="s*(1-depth_pct/100.0)>=0 AND s*(1-depth_pct/100.0)<s"
mm="depth_mm>0 AND depth_mm<s"
strict=f'{internal} AND {nom} AND {depth} AND {residual} AND {mm}'
def scalar(q):return con.execute(q).fetchone()[0]
pair_count=scalar('SELECT COUNT(*) FROM m')
stats={
 'raw_rows':raw,
 'unique_pairs':pair_count,
 'duplicates_removed':raw-pair_count,
 'all_headers_identical':all(x['Заголовки_совпадают'] for x in sources),
 'internal':scalar(f'SELECT COUNT(*) FROM m WHERE {internal}'),
 'positive_nominal':scalar(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom}'),
 'valid_depth_percent':scalar(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth}'),
 'residual_less_than_nominal':scalar(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth} AND {residual}'),
 'depth_mm_present_positive_less_nominal':scalar(f'SELECT COUNT(*) FROM m WHERE {strict}'),
 'invalid_depth_mm_with_other_conditions':scalar(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth} AND {residual} AND NOT ({mm})')
}
valid=con.execute(f'''WITH g AS(SELECT pipe,insp,MIN(dt) dt,COUNT(*) n FROM m WHERE {strict} GROUP BY pipe,insp),p AS(SELECT pipe,COUNT(*) ni,SUM(n) nt FROM g GROUP BY pipe) SELECT g.pipe,g.insp,g.dt,g.n,p.ni,p.nt FROM g JOIN p USING(pipe) ORDER BY CAST(g.pipe AS INTEGER),g.dt,CAST(g.insp AS INTEGER)''').fetchall()
summary=con.execute(f'''WITH a AS(SELECT pipe,COUNT(*) all_n,COUNT(DISTINCT insp) all_i FROM m GROUP BY pipe),v AS(SELECT pipe,COUNT(*) n,COUNT(DISTINCT insp) i,MIN(dt) first_dt,MAX(dt) last_dt FROM m WHERE {strict} GROUP BY pipe) SELECT a.pipe,a.all_n,a.all_i,COALESCE(v.n,0),COALESCE(v.i,0),v.first_dt,v.last_dt FROM a LEFT JOIN v USING(pipe) ORDER BY CAST(a.pipe AS INTEGER)''').fetchall()
src=[]
for f,sh in [(x['Файл'],x['Лист']) for x in sources]:src.append({'Файл':f,'Лист':sh,'Строк':scalar('SELECT COUNT(*) FROM m WHERE file=? AND sheet=?',(f,sh)),'Корректных_внутренних_точек':scalar(f'SELECT COUNT(*) FROM m WHERE file=? AND sheet=? AND {strict}',(f,sh))})
stats.update({'valid_pairs':len(valid),'valid_pipes':len(set(x[0] for x in valid)),'valid_inspections':len(set(x[1] for x in valid)),'valid_total':sum(x[3] for x in valid)})
assert stats['unique_pairs']==raw and stats['duplicates_removed']==0
assert stats['valid_total']==scalar(f'SELECT COUNT(*) FROM m WHERE {strict}')
(OUT/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'valid.json').write_text(json.dumps(valid,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8');(OUT/'sources.json').write_text(json.dumps(src,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
con.close()
