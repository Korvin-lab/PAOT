from pathlib import Path
import sqlite3,json
ROOT=Path('.');OUT=ROOT/'analysis_correct_points';con=sqlite3.connect(OUT/'correct_points.sqlite')
internal="wall_code='XD0002' AND wall_name='внутреннее'"; depth="depth_pct>0 AND depth_pct<=100";nom="s>0";residual="s*(1-depth_pct/100.0)>=0 AND s*(1-depth_pct/100.0)<s";mm="depth_mm>0 AND depth_mm<s";strict=f'{internal} AND {nom} AND {depth} AND {residual} AND {mm}'
def one(q,args=()):return con.execute(q,args).fetchone()[0]
raw=one('SELECT COUNT(*) FROM m')
stats={'raw_rows':raw,'unique_pairs':raw,'duplicates_removed':0,'internal':one(f'SELECT COUNT(*) FROM m WHERE {internal}'),'positive_nominal':one(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom}'),'valid_depth_percent':one(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth}'),'residual_less_than_nominal':one(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth} AND {residual}'),'depth_mm_present_positive_less_nominal':one(f'SELECT COUNT(*) FROM m WHERE {strict}'),'invalid_depth_mm_with_other_conditions':one(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth} AND {residual} AND NOT ({mm})')}
valid=con.execute(f'''WITH g AS(SELECT pipe,insp,MIN(dt) dt,COUNT(*) n FROM m WHERE {strict} GROUP BY pipe,insp),p AS(SELECT pipe,COUNT(*) ni,SUM(n) nt FROM g GROUP BY pipe) SELECT g.pipe,g.insp,g.dt,g.n,p.ni,p.nt FROM g JOIN p USING(pipe) ORDER BY CAST(g.pipe AS INTEGER),g.dt,CAST(g.insp AS INTEGER)''').fetchall()
summary=con.execute(f'''WITH a AS(SELECT pipe,COUNT(*) all_n,COUNT(DISTINCT insp) all_i FROM m GROUP BY pipe),v AS(SELECT pipe,COUNT(*) n,COUNT(DISTINCT insp) i,MIN(dt) first_dt,MAX(dt) last_dt FROM m WHERE {strict} GROUP BY pipe) SELECT a.pipe,a.all_n,a.all_i,COALESCE(v.n,0),COALESCE(v.i,0),v.first_dt,v.last_dt FROM a LEFT JOIN v USING(pipe) ORDER BY CAST(a.pipe AS INTEGER)''').fetchall()
src=[]
for f,sh in con.execute('SELECT DISTINCT file,sheet FROM m ORDER BY file,sheet'):
 src.append({'Файл':f,'Лист':sh,'Строк':one('SELECT COUNT(*) FROM m WHERE file=? AND sheet=?',(f,sh)),'Корректных_внутренних_точек':one(f'SELECT COUNT(*) FROM m WHERE file=? AND sheet=? AND {strict}',(f,sh))})
stats.update({'valid_pairs':len(valid),'valid_pipes':len({x[0] for x in valid}),'valid_inspections':len({x[1] for x in valid}),'valid_total':sum(x[3] for x in valid)})
assert stats['valid_total']==one(f'SELECT COUNT(*) FROM m WHERE {strict}')
for n,x in [('stats.json',stats),('valid.json',valid),('summary.json',summary),('sources.json',src)]: (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(stats,ensure_ascii=False,indent=2));con.close()
