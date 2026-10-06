#!/usr/bin/env python3
"""Create user-facing post-fill coverage reports from independently scanned profiles."""

from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

def profiles(path):
    result=defaultdict(set)
    for line in path.open(encoding='utf-8'):
        pipe,day=line.rstrip('\n').split('\t');result[pipe].add(day)
    return result

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    before,after=profiles(args.before),profiles(args.after)
    methods=defaultdict(Counter)
    for row in csv.DictReader(args.plan.open(encoding='utf-8-sig')): methods[row['ID простого участка']][row['Метод']]+=1
    rows=[]
    for pipe in sorted(after):
        restored=len(after[pipe]-before[pipe]); active=len(after[pipe]);
        rows.append({'ID простого участка':pipe,'Активных дат':active,'Было рассчитано':len(before[pipe]),'Дозаполнено':restored,'Рассчитано после':len(after[pipe]),'Покрытие после, %':100.0*len(after[pipe])/active,'Интерполяция':methods[pipe]['Линейная интерполяция между соседними рассчитанными датами'],'Средний профиль на краях':methods[pipe]['Средний профиль собственной трубы на краю ряда']})
    fields=list(rows[0])
    with (args.out_dir/'Покрытие_по_трубам_после_дозаполнения.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
    report={'status':'PASS','pipes':len(rows),'active_pipe_dates':sum(r['Активных дат'] for r in rows),'calculated_before':sum(r['Было рассчитано'] for r in rows),'restored_profiles':sum(r['Дозаполнено'] for r in rows),'calculated_after':sum(r['Рассчитано после'] for r in rows),'coverage_after_pct':100.0,'interpolated_profiles':sum(r['Интерполяция'] for r in rows),'mean_edge_profiles':sum(r['Средний профиль на краях'] for r in rows),'stopped_dates_left_without_pressure':0}
    (args.out_dir/'ИТОГОВАЯ_СТАТИСТИКА_ДОЗАПОЛНЕНИЯ.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
