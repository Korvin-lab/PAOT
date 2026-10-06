#!/usr/bin/env python3
"""Plan same-pipe profile interpolation for missing active Megion dates."""

from __future__ import annotations

import bisect
import csv
from collections import defaultdict, Counter
from datetime import date
from pathlib import Path


def parse_day(value: str) -> date:
    return date.fromisoformat(value)


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--profiles',type=Path,required=True)
    p.add_argument('--missing',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    complete=defaultdict(list)
    with args.profiles.open(encoding='utf-8') as h:
        for line in h:
            pipe, day=line.rstrip('\n').split('\t'); complete[pipe].append(day)
    for days in complete.values(): days.sort()
    missing=[]
    with args.missing.open(encoding='utf-8-sig',newline='') as h:
        missing=list(csv.DictReader(h))
    plan=[]; neighbors=set(); counts=Counter()
    for item in missing:
        pipe=item['ID простого участка']; day=item['Дата']; days=complete[pipe]; pos=bisect.bisect_left(days,day)
        previous=days[pos-1] if pos else ''
        following=days[pos] if pos<len(days) else ''
        if previous and following:
            method='Линейная интерполяция между соседними рассчитанными датами'
            d0=parse_day(previous); d1=parse_day(following); d=parse_day(day)
            weight=(d-d0).days/(d1-d0).days
            neighbors.update([(pipe,previous),(pipe,following)])
        elif previous or following:
            # This branch needs all same-pipe profiles later to calculate the mean.
            method='Средний профиль собственной трубы на краю ряда'
            weight=''
        else:
            raise ValueError(f'No calculated profile for {pipe}')
        counts[method]+=1
        plan.append({'ID простого участка':pipe,'Дата':day,'Дата до':previous,'Дата после':following,'Вес после':weight,'Метод':method})
    fields=list(plan[0])
    with (args.out_dir/'ПЛАН_ДОЗАПОЛНЕНИЯ_ПРОФИЛЕЙ.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(plan)
    with (args.out_dir/'ПРОФИЛИ_ДОНОРЫ_ДЛЯ_ИЗВЛЕЧЕНИЯ.tsv').open('w',encoding='utf-8') as h:
        for pipe,day in sorted(neighbors):h.write(f'{pipe}\t{day}\n')
    print({'missing':len(plan),'methods':dict(counts),'endpoint_profiles':len(neighbors)})

if __name__=='__main__':main()
