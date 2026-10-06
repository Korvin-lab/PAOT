from pathlib import Path
import sqlite3,json
from openpyxl import Workbook,load_workbook
from openpyxl.styles import Font,PatternFill,Alignment,Border,Side
from openpyxl.utils import get_column_letter
ROOT=Path('.');DATA=ROOT/'analysis_correct_points';DB=DATA/'correct_points.sqlite';OUT=ROOT/'Мегион_сводка_ВТД_внутренние_дефекты__2026_09_17.xlsx'
audit=json.loads((DATA/'independent_audit.json').read_text(encoding='utf8'));assert audit['status']=='PASS'
con=sqlite3.connect(DB);internal="wall_code='XD0002' AND wall_name='внутреннее'";depth="depth_pct>0 AND depth_pct<=100";nom="s>0";res="s*(1-depth_pct/100.0)>=0 AND s*(1-depth_pct/100.0)<s";valid=f'{internal} AND {nom} AND {depth} AND {res}'
rows=con.execute(f'''WITH g AS(SELECT pipe,insp,MIN(dt) dt,COUNT(*) n FROM m WHERE {valid} GROUP BY pipe,insp),p AS(SELECT pipe,COUNT(*) ni,SUM(n) nt FROM g GROUP BY pipe) SELECT g.pipe,g.insp,g.dt,g.n,p.ni,p.nt FROM g JOIN p USING(pipe) ORDER BY CAST(g.pipe AS INTEGER),g.dt,CAST(g.insp AS INTEGER)''').fetchall()
summary=con.execute(f'''WITH a AS(SELECT pipe,COUNT(*) all_n,COUNT(DISTINCT insp) all_i FROM m GROUP BY pipe),v AS(SELECT pipe,COUNT(*) n,COUNT(DISTINCT insp) i FROM m WHERE {valid} GROUP BY pipe) SELECT a.pipe,a.all_n,a.all_i,COALESCE(v.n,0),COALESCE(v.i,0) FROM a LEFT JOIN v USING(pipe) ORDER BY CAST(a.pipe AS INTEGER)''').fetchall()
stages=[('Исходных строк ВТД',con.execute('SELECT COUNT(*) FROM m').fetchone()[0],'Все строки всех листов'),('Уникальных пар ID замера + ID простого участка',con.execute('SELECT COUNT(*) FROM m').fetchone()[0],'Точных дублей нет'),('Подтверждённо внутренних дефектов',con.execute(f'SELECT COUNT(*) FROM m WHERE {internal}').fetchone()[0],'XD0002 и «внутреннее»'),('С положительной номинальной толщиной S',con.execute(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom}').fetchone()[0],'S > 0'),('С корректной глубиной',con.execute(f'SELECT COUNT(*) FROM m WHERE {internal} AND {nom} AND {depth}').fetchone()[0],'0 < глубина, % ≤ 100'),('С остаточной толщиной меньше номинальной',con.execute(f'SELECT COUNT(*) FROM m WHERE {valid}').fetchone()[0],'0 ≤ S × (1 − глубина% / 100) < S'),('Потеря металла / коррозия', 'Не применялось','Тип особенности не является условием отбора'),('Независимая проверка',audit['status'],'Повторный прямой разбор всех исходных XLS')];
src=con.execute(f'''SELECT file,sheet,COUNT(*),SUM(CASE WHEN {valid} THEN 1 ELSE 0 END) FROM m GROUP BY file,sheet ORDER BY file,sheet''').fetchall();con.close()
assert len(rows)>0 and sum(x[3] for x in rows)==489779 and len(summary)==80
wb=Workbook();a=wb.active;a.title='Внутренние дефекты';a.append(['ID простого участка','ID освидетельствования','Дата освидетельствования','Количество корректных внутренних дефектов в этом ВТД','Количество ВТД с корректными внутренними дефектами по трубе','Всего корректных внутренних дефектов по трубе'])
for r in rows:a.append(r)
b=wb.create_sheet('Сводка по трубам');b.append(['ID простого участка','Замеры ВТД без фильтра','Количество замеров ВТД без фильтра','Количество ВТД в файлах замеров','Корректные внутренние дефекты','Количество корректных внутренних дефектов','Количество ВТД с корректными внутренними дефектами','Перечень освидетельствований','Журнал ВТД по перечню','Журнал УЗТ по перечню'])
for pipe,alln,alli,n,i in summary:b.append([pipe,'Есть',alln,alli,'Есть' if n else 'Нет',n,i,'Нет сведений: файл не предоставлен','Нет сведений: файл не предоставлен','Нет сведений: файл не предоставлен'])
c=wb.create_sheet('ВТД без замеров');c.merge_cells('A1:F1');c['A1']='Контроль полноты источников и ВТД без замеров';c.append(['Статус','Невозможно сформировать перечень «ВТД без замеров»']);c.append(['Причина','В новой папке отсутствуют отдельные файлы перечня освидетельствований и журналов ВТД. Неизвестные данные не подменялись значением «Нет».']);c.append(['Правило корректной точки','Внутренний дефект: XD0002 и «внутреннее»; S > 0; 0 < глубина, % ≤ 100; остаточная толщина S × (1 − глубина% / 100) меньше S и неотрицательна.']);c.append([]);c.append(['Контрольный показатель','Значение','Правило / результат'])
for r in stages:c.append(r)
c.append([]);c.append(['Файл','Лист','Строк после дедупликации','Корректных внутренних точек'])
for r in src:c.append(r)
navy='1F4E78';white='FFFFFF';green='E2F0D9';red='FCE4D6';yellow='FFF2CC';thin=Side(style='thin',color='D9E1F2')
for ws in wb.worksheets:
 ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions;ws.sheet_view.showGridLines=False;ws.row_dimensions[1].height=42
 for cell in ws[1]:cell.fill=PatternFill('solid',fgColor=navy);cell.font=Font(color=white,bold=True);cell.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
 for row in ws.iter_rows():
  for cell in row:cell.alignment=Alignment(vertical='top',wrap_text=True);cell.border=Border(bottom=thin)
 for col in range(1,ws.max_column+1):ws.column_dimensions[get_column_letter(col)].width=min(max(14,max(len(str(ws.cell(r,col).value or '')) for r in range(1,min(ws.max_row,300)+1))+2),58)
 for col in range(1,ws.max_column+1):
  if str(ws.cell(1,col).value).startswith('ID '):
   for row in range(2,ws.max_row+1):ws.cell(row,col).number_format='@'
for r in range(2,b.max_row+1):
 b.cell(r,2).fill=PatternFill('solid',fgColor=green);b.cell(r,5).fill=PatternFill('solid',fgColor=green if b.cell(r,5).value=='Есть' else red)
 for x in (8,9,10):b.cell(r,x).fill=PatternFill('solid',fgColor=yellow)
for row in range(2,a.max_row+1):
 for col in (4,5,6):a.cell(row,col).number_format='#,##0'
for r in (6,16):
 for col in range(1,c.max_column+1):
  x=c.cell(r,col)
  if x.value is not None:x.fill=PatternFill('solid',fgColor=navy);x.font=Font(color=white,bold=True);x.alignment=Alignment(horizontal='center',vertical='center',wrap_text=True)
c.column_dimensions['A'].width=50;c.column_dimensions['B'].width=58;c.column_dimensions['C'].width=31;c.column_dimensions['D'].width=31;c.row_dimensions[3].height=48;c.row_dimensions[4].height=62
wb.save(OUT)
check=load_workbook(OUT,read_only=True,data_only=False);assert check.sheetnames==['Внутренние дефекты','Сводка по трубам','ВТД без замеров'];assert check['Внутренние дефекты'].max_row-1==len(rows);assert sum(check['Внутренние дефекты'].cell(r,4).value for r in range(2,check['Внутренние дефекты'].max_row+1))==489779;assert sum(check['Сводка по трубам'].cell(r,6).value for r in range(2,check['Сводка по трубам'].max_row+1))==489779;check.close();print('PASS',OUT)
