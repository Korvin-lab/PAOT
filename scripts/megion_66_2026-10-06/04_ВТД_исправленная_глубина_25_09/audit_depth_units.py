from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import xlrd


ROOT = Path('/Volumes/KINGSTON/Газпром работа/Прогнозирование утонения стенки трубы/Перевыгрузка по ВТД_Мегион_сентябрь2026')
OUT = Path(__file__).resolve().parent


def number(value):
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(value) else None
    try:
        return float(str(value).strip().replace(',', '.'))
    except (TypeError, ValueError):
        return None


def bucket(value):
    if value is None:
        return 'missing'
    if value <= 0:
        return 'nonpositive'
    if value < 1:
        return 'fraction_candidate'
    if value == 1:
        return 'one_ambiguous'
    if value <= 100:
        return 'percent_candidate'
    return 'over_100'


def main():
    totals = Counter()
    by_sheet = []
    samples = defaultdict(list)
    ratios = defaultdict(Counter)
    with (OUT / 'examples.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['Файл', 'Лист', 'Строка Excel', 'ID замера', 'ID трубы', 'S мм', 'Глубина % (как в источнике)', 'Глубина мм (как в источнике)', 'Категория', 'mm/(S*p/100)', 'mm/(S*p)'])
        for path in sorted(ROOT.glob('Выгрузка_ВТД_СН-МНГ_*.xls')):
            book = xlrd.open_workbook(str(path), on_demand=True, ignore_workbook_corruption=True)
            for name in book.sheet_names():
                sheet = book.sheet_by_name(name)
                headers = sheet.row_values(0)
                col = {str(v): i for i, v in enumerate(headers)}
                pcol = next(i for i, v in enumerate(headers) if 'глубина,%' in str(v).lower())
                mcol = next(i for i, v in enumerate(headers) if 'глубина,мм' in str(v).lower())
                scol = col['Толщина стенки элемента, мм']
                c = Counter()
                for i in range(1, sheet.nrows):
                    row = sheet.row_values(i)
                    if not any(str(v).strip() for v in row):
                        continue
                    p, mm, s = number(row[pcol]), number(row[mcol]), number(row[scol])
                    b = bucket(p)
                    c[b] += 1
                    c['rows'] += 1
                    if mm is not None:
                        c['mm_present'] += 1
                        c[b + '_mm_present'] += 1
                    if p is not None and p > 0 and mm is not None and mm > 0 and s is not None and s > 0:
                        c['both_positive'] += 1
                        percent_ratio = mm / (s * p / 100)
                        fraction_ratio = mm / (s * p)
                        if abs(mm - s*p/100) <= 0.02:
                            c[b + '_percent_match_002'] += 1
                        if abs(mm - s*p) <= 0.02:
                            c[b + '_fraction_match_002'] += 1
                        ratios[b][round(percent_ratio, 2)] += 1
                    else:
                        percent_ratio = fraction_ratio = ''
                    if len(samples[(path.name, name, b, mm is not None)]) < 5:
                        samples[(path.name, name, b, mm is not None)].append(i)
                        writer.writerow([path.name, name, i+1, row[7], row[3], s, p, mm, b, percent_ratio, fraction_ratio])
                by_sheet.append({'file': path.name, 'sheet': name, **dict(c)})
                totals.update(c)
                print(path.name, name, c['rows'], c['fraction_candidate'], c['fraction_candidate_mm_present'], c['both_positive'], flush=True)
                book.unload_sheet(name)
            book.release_resources()
    report = {'totals': dict(totals), 'by_sheet': by_sheet, 'ratio_modes': {k: v.most_common(12) for k, v in ratios.items()}}
    (OUT/'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['totals'], ensure_ascii=False, indent=2))
    print('ratio modes', report['ratio_modes'])


if __name__ == '__main__':
    main()
