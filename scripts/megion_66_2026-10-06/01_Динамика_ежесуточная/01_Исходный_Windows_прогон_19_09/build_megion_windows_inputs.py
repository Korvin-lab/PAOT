#!/usr/bin/env python3
"""Build reproducible Megion inputs for the strict Windows PE2 package.

This program is intentionally run on macOS before delivery.  The resulting
JSON/CSV files are the only inputs required by the Windows calculation; raw
Excel workbooks are not silently re-read on Windows.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path('/Volumes/KINGSTON/Газпром работа/Прогнозирование утонения стенки трубы')
PACKAGE = Path(__file__).resolve().parent
DATA = ROOT / 'Данные для сбора датасета Мегион'
TARGET_XLSX = ROOT / 'Мегион_нужные трубы.xlsx'
POT_XLSX = DATA / 'Перечень ответственных трубопроводов СН МНГ.xlsx'
CHEM_POT_XLSX = ROOT / 'Перечень_ответственных_трубопроводов_СН_МНГ 2.xlsx'
TECH_XLSX = DATA / 'Технологический режим Мегион.xlsx'
SHTR_XLSX = DATA / 'Тех.режим 2015-2026 MEGION.xlsx'
FHS_XLSX = DATA / 'ФХС 2015-2026 MEGION.xlsx'
OLD_GRAPH = ROOT / 'работа с json/megion_shtr_fhs_graph_2026-08-17/graph_megion_master__with_shtr_fhs.json'
UKK_PARSER = ROOT / 'работа с json/recheck_megion_ukk_by_ukk_all_directions_2026_09_16.py'
OUT = PACKAGE / 'input'
AUDIT = PACKAGE / 'audit_megion'

SHORT_EDGE_DAYS = 31
LARGE_GAP_DAYS = 180
NUMERIC_COLUMNS = [
    'Жидкости, кг/м3', 'Жидкости, м3/сут (дебит)', 'Обводненность, %',
    'Нефти, т/сут (дебит)', 'Газа, кг/м3', 'Газа, кг/(м*с)*1000',
    'Жидкости, кг/(м*с)', 'Общего газа, тыс.м3/сут (дебит)', 'Нефти, кг/м3',
    'L', 'D', 'S', 't', 'p', 'Общая минерализация',
]


def text(v: Any) -> str:
    return '' if v is None else re.sub(r'\s+', ' ', str(v).replace('\xa0', ' ')).strip()


def norm_id(v: Any) -> str:
    s = text(v).replace(' ', '')
    return s[:-2] if re.fullmatch(r'\d+\.0', s) else s


def number(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(str(v).replace('\xa0', '').replace(' ', '').replace(',', '.'))
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def day(v: Any) -> str | None:
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    s = text(v)
    for fmt in ('%d.%m.%Y %H:%M', '%d.%m.%Y', '%Y-%m-%d', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_targets() -> dict[str, dict[str, str]]:
    wb = load_workbook(TARGET_XLSX, read_only=True, data_only=True)
    ws = wb.active
    result = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        pid = norm_id(row[0] if row else None)
        if pid:
            result[pid] = {'name': text(row[1] if len(row) > 1 else ''), 'purpose': text(row[2] if len(row) > 2 else '')}
    wb.close()
    if len(result) != 67:
        raise RuntimeError(f'Expected 67 target pipes, got {len(result)}')
    return result


def load_pot() -> dict[str, dict[str, Any]]:
    wb = load_workbook(POT_XLSX, read_only=True, data_only=True)
    ws = wb.active
    result: dict[str, dict[str, Any]] = {}
    for rn, row in enumerate(ws.iter_rows(min_row=3, values_only=True), 3):
        pid = norm_id(row[48] if len(row) > 48 else None)
        if not pid:
            continue
        if pid in result:
            raise RuntimeError(f'Duplicate simple section ID in current POT: {pid}')
        name = text(row[49] if len(row) > 49 else '')
        parts = re.split(r'\s*(?:->|–|—|\-|—)\s*', name, maxsplit=1)
        length_km = number(row[50] if len(row) > 50 else None)
        result[pid] = {
            'pot_row': rn, 'id': pid, 'name': name, 'field': text(row[5] if len(row) > 5 else ''),
            'purpose': text(row[12] if len(row) > 12 else ''), 'direction_id': norm_id(row[42] if len(row) > 42 else None),
            'direction_name': text(row[43] if len(row) > 43 else ''), 'main_id': norm_id(row[45] if len(row) > 45 else None),
            'L': length_km * 1000 if length_km and length_km > 0 else None,
            'D': number(row[20] if len(row) > 20 else None), 'S': number(row[21] if len(row) > 21 else None),
            'start': text(parts[0]) if parts else '', 'end': text(parts[1]) if len(parts) > 1 else '',
        }
    wb.close()
    return result


def load_old_graph(targets: dict[str, Any], pot: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    old = json.loads(OLD_GRAPH.read_text(encoding='utf-8'))
    old_ids = {norm_id(n['id']) for n in old['nodes']}
    missing = old_ids - set(pot)
    if missing:
        raise RuntimeError(f'Graph nodes absent from current POT: {sorted(missing)}')
    if set(targets) - old_ids:
        raise RuntimeError('Current target list is not covered by confirmed graph')
    mismatches = []
    nodes = []
    for old_node in old['nodes']:
        pid = norm_id(old_node['id'])
        cur = pot[pid]
        if old_node.get('name') and text(old_node['name']) != cur['name']:
            mismatches.append({'id': pid, 'old_name': old_node.get('name'), 'current_name': cur['name']})
        nodes.append({
            'id': pid,
            'graph': {'found_in_graph': True, 'simple_id': pid, 'main_id': cur['main_id'],
                      'simple_name': cur['name'], 'object_name': cur['direction_name'],
                      'start_node_norm': cur['start'], 'end_node_norm': cur['end'],
                      'L_m': cur['L'], 'D_mm': cur['D'], 'S_mm': cur['S'], 'is_complete': bool(cur['L'] and cur['D'] and cur['S'])},
            'kust_binding': {'eligible_one_kust': False, 'kust': '', 'kust_count_in_sources': 0, 'location_by_sources': ''},
            'source_coverage': {},
        })
    edges = [{'source': norm_id(e['source']), 'target': norm_id(e['target']), 'main_id': '',
              'via_node': text(e.get('via_node')), 'edge_type': text(e.get('status', 'confirmed_graph')),
              'match_method': 'confirmed_2026_08_graph'} for e in old['edges']]
    return old, nodes, mismatches


def read_tech(valid_ids: set[str]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    wb = load_workbook(TECH_XLSX, read_only=True, data_only=True)
    ws = wb.active
    headers = [text(x) for x in next(ws.iter_rows(min_row=1, max_row=1, values_only=True))]
    required = {'ID простого участка': 7, 'Дата расчёта': 15, 'L,м': 12, 'D,мм': 13, 'S,мм': 14}
    for label, index in required.items():
        if index >= len(headers) or headers[index] != label:
            raise RuntimeError(f'Tech regime schema changed at column {index + 1}: expected {label!r}, got {headers[index] if index < len(headers) else None!r}')
    rows = []
    duplicate_rows = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for rn, row in enumerate(ws.iter_rows(min_row=2, values_only=True), 2):
        pid = norm_id(row[7] if len(row) > 7 else None)
        dt = day(row[15] if len(row) > 15 else None)
        if pid not in valid_ids or not dt:
            continue
        r = {
            'id': pid, 'date': dt, 'L': number(row[12]), 'D': number(row[13]), 'S': number(row[14]),
            'Жидкости, м3/сут (дебит)': number(row[16]), 'Общего газа, тыс.м3/сут (дебит)': number(row[18]),
            'Обводненность, %': number(row[19]), 'Нефти, т/сут (дебит)': number(row[20]),
            'p': (number(row[21]) / 10 if number(row[21]) is not None else None),
            't': number(row[59]), 'Жидкости, кг/м3': number(row[39]), 'Газа, кг/м3': number(row[40]),
            'Нефти, кг/м3': number(row[42]), 'Жидкости, кг/(м*с)': number(row[44]),
            'Газа, кг/(м*с)*1000': number(row[45]), 'Общая минерализация': number(row[62]),
            '_source_row': rn,
        }
        key = (pid, dt)
        if key in seen:
            duplicate_rows.append({'id': pid, 'date': dt, 'kept_source_row': seen[key]['_source_row'], 'discarded_source_row': rn})
            continue
        seen[key] = r
        rows.append(r)
    wb.close()
    if not rows:
        raise RuntimeError('No current technical-regime rows found for graph nodes')
    return pd.DataFrame(rows), duplicate_rows


def source_links_to_shtr(old_graph: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for link in old_graph.get('source_links', []):
        if text(link.get('Статус')) != 'Подтверждено':
            continue
        pid = norm_id(link.get('ID первой трубы после источника'))
        typ = text(link.get('Тип источника')).lower()
        out[pid].append({'type': typ, 'number': text(link.get('Номер источника')), 'field': text(link.get('Месторождение по 7.1'))})
    return out


def read_shtr_sources(old_graph: dict[str, Any]) -> pd.DataFrame:
    links = source_links_to_shtr(old_graph)
    if not links:
        return pd.DataFrame(columns=['id', 'date', 't_shtr', 'p_shtr'])
    wanted = {(x['type'], x['number'].lower(), x['field'].lower(), pid) for pid, xs in links.items() for x in xs}
    rows = []
    wb = load_workbook(SHTR_XLSX, read_only=True, data_only=True)
    for ws in wb.worksheets:
        for row in ws.iter_rows(min_row=2, values_only=True):
            if len(row) < 14:
                continue
            field, bush, well, dt = text(row[2]).lower(), text(row[3]).lower(), text(row[1]).lower(), day(row[6])
            if not dt:
                continue
            ql, p, t = number(row[7]), number(row[11]), number(row[12])
            if ql is None or ql <= 0:
                continue
            for typ, source_no, source_field, pid in wanted:
                token = bush if typ == 'куст' else well
                if field == source_field and token == source_no:
                    # Exact T=P rows are a known mass-copy anomaly; retain P but reject T.
                    rows.append({'id': pid, 'date': dt, 't_shtr': t if t and 5 < t <= 90 and t != p else None,
                                 'p_shtr': p / 10 if p and 1 <= p <= 90 else None})
    wb.close()
    if not rows:
        return pd.DataFrame(columns=['id', 'date', 't_shtr', 'p_shtr'])
    df = pd.DataFrame(rows)
    return df.groupby(['id', 'date'], as_index=False).median(numeric_only=True)


def fhs_mineralization_median() -> float:
    """Use a measured FHS median only when the technical regime has no salt value."""
    wb = load_workbook(FHS_XLSX, read_only=True, data_only=True)
    ws = wb.active
    values = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        v = number(row[27] if len(row) > 27 else None)
        if v is not None and v > 0:
            values.append(v)
    wb.close()
    if not values:
        raise RuntimeError('FHS contains no positive mineralization for the last-resort chemistry fallback')
    return float(np.median(values))


def fill_one_series(s: pd.Series, dates: pd.Series, field_value: float, valid: callable) -> tuple[pd.Series, pd.Series]:
    """Temporal fill with provenance: internal linear, seasonal, pipe mean, field mean."""
    x = pd.to_numeric(s, errors='coerce').where(lambda z: valid(z))
    out = x.copy()
    src = pd.Series(np.where(x.notna(), 'исходное', ''), index=x.index, dtype=object)
    if x.notna().sum() == 0:
        out[:] = field_value
        src[:] = 'среднее по месторождению'
        return out, src
    # Small internal gaps are interpolated; long gaps first use same calendar day from prior years.
    known = np.flatnonzero(x.notna().to_numpy())
    for left, right in zip(known[:-1], known[1:]):
        gap = right - left - 1
        if gap <= 0:
            continue
        if gap <= SHORT_EDGE_DAYS:
            out.iloc[left + 1:right] = np.linspace(out.iloc[left], out.iloc[right], gap + 2)[1:-1]
            src.iloc[left + 1:right] = 'интерполяция внутри ряда'
    by_md = {(pd.Timestamp(d).month, pd.Timestamp(d).day): [] for d in dates}
    for i, value in enumerate(x):
        if pd.notna(value):
            by_md.setdefault((pd.Timestamp(dates.iloc[i]).month, pd.Timestamp(dates.iloc[i]).day), []).append(float(value))
    for i in np.flatnonzero(out.isna().to_numpy()):
        values = by_md.get((pd.Timestamp(dates.iloc[i]).month, pd.Timestamp(dates.iloc[i]).day), [])
        if values:
            out.iloc[i] = float(np.mean(values)); src.iloc[i] = 'тот же календарный день других лет'
    # Edge gaps up to one month use nearest direct observation, then pipe mean.
    direct_idx = np.flatnonzero(x.notna().to_numpy())
    for i in np.flatnonzero(out.isna().to_numpy()):
        distance = min(abs(i - j) for j in direct_idx)
        if distance <= SHORT_EDGE_DAYS:
            j = min(direct_idx, key=lambda k: (abs(i - k), k > i))
            out.iloc[i] = x.iloc[j]; src.iloc[i] = 'ближайшее значение на краю'
    mean_pipe = float(x.mean())
    out = out.fillna(mean_pipe)
    src = src.mask(src.eq(''), 'среднее по трубе')
    return out, src


def apply_tp_and_own_fill(df: pd.DataFrame, edges: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    df = df.copy().sort_values(['id', 'date']).reset_index(drop=True)
    flow_columns = ['Жидкости, м3/сут (дебит)', 'Нефти, т/сут (дебит)', 'Общего газа, тыс.м3/сут (дебит)']
    raw_flow = df[flow_columns].apply(pd.to_numeric, errors='coerce')
    # A true stop is not a missing measurement and must never be overwritten.
    df['is_true_stop'] = raw_flow.fillna(0).le(0).all(axis=1)
    for col, low, high in [('t', 5, 90), ('p', 0.01, 9)]:
        df[col] = pd.to_numeric(df[col], errors='coerce').where(lambda x: (x > low) & (x <= high))
        df[f'source_{col}'] = np.where(df[col].notna(), 'трубный техрежим', '')
    # Fill from predecessor graph only for T/P and only on exactly same date.
    pred: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        pred[norm_id(edge['target'])].append(norm_id(edge['source']))
    reports = []
    for pid in df['id'].drop_duplicates():
        idx = df.index[df['id'].eq(pid)]
        for col, low, high in [('t', 5, 90), ('p', 0.01, 9)]:
            missing = df.loc[idx, col].isna()
            transferred = 0
            for source in pred.get(pid, []):
                donor = df[df['id'].eq(source)][['date', col]].dropna().set_index('date')[col]
                if donor.empty:
                    continue
                dates = df.loc[idx, 'date']
                values = dates.map(donor)
                take = missing & values.notna() & (values > low) & (values <= high)
                if take.any():
                    ridx = idx[take.to_numpy()]
                    df.loc[ridx, col] = values.loc[take].to_numpy()
                    df.loc[ridx, f'source_{col}'] = f'граф от предшественника {source}'
                    missing.loc[take] = False; transferred += int(take.sum())
            reports.append({'id': pid, 'parameter': col, 'rows_from_graph': transferred})
        # Temporal T/P and all non-T/P fields only within the same pipe.
        part = df.loc[idx].copy().sort_values('date')
        field = text(part['field'].iloc[0])
        for col in NUMERIC_COLUMNS:
            if col not in part:
                continue
            if col == 't':
                lo, hi = 5, 90
            elif col == 'p':
                lo, hi = 0.01, 9
            elif col == 'Обводненность, %':
                lo, hi = 0, 100
            else:
                lo, hi = 0, math.inf
            all_valid = pd.to_numeric(df[col], errors='coerce')
            inclusive_zero = col == 'Обводненность, %'
            lower_ok = all_valid.ge(lo) if inclusive_zero else all_valid.gt(lo)
            pool = all_valid[(df['field'].eq(field)) & lower_ok & (all_valid <= hi)]
            field_mean = float(pool.mean()) if not pool.empty else float(all_valid[(all_valid > lo) & (all_valid <= hi)].mean())
            if not math.isfinite(field_mean):
                raise RuntimeError(f'No usable field/global mean for {col} on {pid}')
            working = ~part['is_true_stop'] if col in flow_columns else pd.Series(True, index=part.index)
            source = part.loc[working, col]
            dates = part.loc[working, 'date']
            filled, provenance = fill_one_series(source, dates, field_mean, lambda s: ((s >= lo) if inclusive_zero else (s > lo)) & (s <= hi))
            part.loc[working, col] = filled
            if col in ('t', 'p'):
                source_col = f'source_{col}'
                missing_source = part[source_col].eq('')
                part.loc[missing_source, source_col] = provenance.loc[missing_source]
            else:
                part[f'source_{col}'] = ''
                part.loc[working, f'source_{col}'] = provenance
        df.loc[part.index, part.columns] = part
    return df, reports


def load_ukk_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream the validated 61-column FHS templates without expanding merges.

    The 2026-09-16 audit already established the full workbook/sheet inventory.
    Here we re-read every workbook for values, but use read_only mode: date and
    UKK blocks are carried forward from merged-cell anchors, exactly as in the
    audit, while avoiding materialising a million formatted blank cells.
    """
    cache = AUDIT / 'ukk_value_extract_cache.json'
    if cache.exists():
        saved=json.loads(cache.read_text(encoding='utf-8'))
        return saved['records'], saved['structures']
    spec = importlib.util.spec_from_file_location('ukk', UKK_PARSER)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    files=[]
    for source_dir in (ROOT/'УКК от ДО', ROOT/'УКК от ДО 2'):
        files.extend(p for p in source_dir.rglob('*.xlsx') if not p.name.startswith(('._','~$')))
    raw=[]; structures=[]; seen={}; duplicates=[]
    for file_no,path in enumerate(sorted(files),1):
        is_fhs='физ-хим' in path.name.lower()
        wb=load_workbook(path,read_only=True,data_only=True)
        for ws in wb.worksheets:
            header_rows=list(ws.iter_rows(min_row=1,max_row=min(3,ws.max_row),max_col=min(61,ws.max_column),values_only=True))
            def h(c): return ' '.join(text(r[c-1] if len(r)>=c else '') for r in header_rows).upper().replace('Ё','Е')
            valid=is_fhs and ws.max_column>=61 and all([
                'УКК' in h(14), 'ДАТА' in h(20), 'PH' in h(45), 'CO2' in h(58),
                ('O2' in h(59) or 'О2' in h(59)) and 'CO2' not in h(59),
                'H2S' in h(60) and 'ГАЗ' in h(60), 'H2S' in h(61) and 'НЕФТ' in h(61),
            ])
            structures.append({'Файл':str(path.relative_to(ROOT)),'Лист':ws.title,'Строк':ws.max_row,'Столбцов':ws.max_column,'Валидный физ-хим шаблон':valid})
            if not valid: continue
            carry={'ukk':'','date':None}; empty_run=max_empty=0
            for rn,row in enumerate(ws.iter_rows(min_row=4,max_col=61,values_only=True),4):
                # 1-based fixed columns from independently audited template schema.
                ukk_raw=row[13] if len(row)>13 else None; date_raw=row[19] if len(row)>19 else None
                chem=[number(row[i] if len(row)>i else None) for i in (44,57,59,60)]
                if all(v in (None,'') for v in (ukk_raw,date_raw,*chem)):
                    empty_run+=1;max_empty=max(max_empty,empty_run);continue
                empty_run=0
                if text(ukk_raw): carry['ukk']=module.norm_ukk(ukk_raw)
                parsed=module.parse_date(date_raw)
                if parsed: carry['date']=parsed
                ph,co2,hgas,hoil=chem
                if carry['date'] is None or not any([co2 is not None and co2>=0, ph is not None and 0<ph<=14, hgas is not None and hgas>=0, hoil is not None and hoil>=0]):
                    continue
                rec={'Файл':str(path.relative_to(ROOT)),'Лист':ws.title,'Строка':rn,'Дата':carry['date'].isoformat(),'Номер УКК':carry['ukk'],'pH':ph,'CO2 в водной фазе':co2,'H2S в газовой фазе':hgas,'H2S в нефтяной фазе':hoil}
                key=(rec['Дата'],rec['Номер УКК'],rec['pH'],rec['CO2 в водной фазе'],rec['H2S в газовой фазе'],rec['H2S в нефтяной фазе'])
                if key in seen:
                    duplicates.append({'Файл':rec['Файл'],'Лист':rec['Лист'],'Строка':rn,'Дублирует':seen[key]})
                else:
                    seen[key]=f"{rec['Файл']}::{rec['Лист']}::{rn}";raw.append(rec)
            structures[-1]['Максимальный пустой интервал строк']=max_empty
        wb.close()
        print(f'UKK {file_no}/{len(files)}: {path.name}',flush=True)
    cache.write_text(json.dumps({'records':raw,'structures':structures},ensure_ascii=False),encoding='utf-8')
    return raw, structures


def build_chemistry(df: pd.DataFrame, pot: dict[str, dict[str, Any]], targets: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records, structures = load_ukk_records()
    # UKK mapping remains strictly by UKK number, per approved 2026-09-16 rule.
    by_ukk: dict[str, set[str]] = defaultdict(set)
    for p in pot.values():
        # Current POT's UKK column is the same schema position as the audited "2" edition.
        # Direct row access is deliberate and schema is asserted by the builder manifest.
        pass
    # The user-approved UKK reconciliation was performed on the numbered POT
    # edition.  It is the chemistry link authority; the current POT remains
    # the authority for geometry and graph construction.
    wb = load_workbook(CHEM_POT_XLSX, read_only=True, data_only=True); ws = wb.active
    for row in ws.iter_rows(min_row=3, values_only=True):
        pid, did = norm_id(row[48] if len(row)>48 else None), norm_id(row[42] if len(row)>42 else None)
        ukk = re.sub(r'[^0-9A-ZА-Я]+', '', text(row[244] if len(row)>244 else '').upper().replace('Ё','Е')).replace('УКК','').replace('N','').replace('№','')
        if ukk and did:
            by_ukk[ukk].add(did)
    wb.close()
    target_by_dir: dict[str, list[str]] = defaultdict(list)
    for pid in targets:
        target_by_dir[pot[pid]['direction_id']].append(pid)
    chem_rows = []
    excluded = 0
    for rec in records:
        dirs = by_ukk.get(text(rec.get('Номер УКК')))
        if not dirs:
            excluded += 1; continue
        for did in dirs:
            for pid in target_by_dir.get(did, []):
                chem_rows.append({'id': pid, 'date': rec['Дата'], 'CO2': rec.get('CO2 в водной фазе'), 'pH': rec.get('pH'),
                                  'H2S in Gas Phase': rec.get('H2S в газовой фазе'), 'H2S in Water Phase': (number(rec.get('H2S в нефтяной фазе')) or 0) / 1000 if number(rec.get('H2S в нефтяной фазе')) is not None else None,
                                  'source': f"УКК {rec.get('Номер УКК')}", 'direction_id': did})
    raw = pd.DataFrame(chem_rows)
    if raw.empty:
        raise RuntimeError('No UKK chemistry mapped to target directions')
    raw = raw.groupby(['id','date'],as_index=False).agg({'CO2':'median','pH':'median','H2S in Gas Phase':'median','H2S in Water Phase':'median','source':'first'})
    dates = df[['id','date','field']].drop_duplicates().copy()
    out = dates.merge(raw, on=['id','date'], how='left')
    # Do not turn zero measurements into chemical concentrations: zero is source-missing for final filling.
    for col, lo, hi in [('CO2',0,math.inf),('pH',0,14),('H2S in Gas Phase',0,math.inf)]:
        out[col] = pd.to_numeric(out[col],errors='coerce').where(lambda x: (x>lo)&(x<=hi))
        values = out[col]
        field_mean = float(values[values.notna()].median())
        if not math.isfinite(field_mean):
            # Water-phase H2S is absent for some directions; use the field-wide positive median only after the audited donor pool.
            raise RuntimeError(f'No positive chemistry value available for {col}')
        filled_parts=[]
        for _, part in out.groupby('id',sort=False):
            filled, provenance = fill_one_series(part[col],part['date'],field_mean,lambda s:(s>lo)&(s<=hi))
            q=part.copy();q[col]=filled;q[f'{col} source']=np.where(part[col].notna(),'прямая УКК',provenance);filled_parts.append(q)
        out=pd.concat(filled_parts,ignore_index=True)
    # No approved gas-to-water conversion exists.  Preserve water-phase H2S
    # as missing rather than fabricating a physically different phase.
    out['H2S in Water Phase source'] = np.where(out['H2S in Water Phase'].notna(), 'прямая УКК: нефтяная фаза / 1000', 'нет подтвержденного источника водной фазы')
    out['Общая минерализация'] = pd.to_numeric(df.set_index(['id','date']).reindex(pd.MultiIndex.from_frame(out[['id','date']]))['Общая минерализация'].to_numpy(),errors='coerce')
    min_positive = out['Общая минерализация'].where(out['Общая минерализация']>0)
    out['Общая минерализация']=min_positive.fillna(float(min_positive.median()))
    report={'source_records_total':len(records),'source_sheets_valid':sum(bool(x.get('Валидный физ-хим шаблон')) for x in structures),'records_excluded_without_ukk_direction':excluded,'mapped_pipe_date_rows_raw':len(raw),'target_pipes_with_direct_chemistry':int(raw['id'].nunique()),'chemistry_pot':str(CHEM_POT_XLSX),'chemistry_pot_sha256':sha256(CHEM_POT_XLSX),'h2s_water_phase_policy':'No approved source/conversion for target pipes: retain NaN, do not fill with gas, zero, pipe mean, or field mean.','rule':'UKK number -> all directions in approved numbered POT -> all target pipes of those directions; H2S gas is filled independently; oil H2S remains an audited separate source.'}
    return out, report


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def main() -> None:
    OUT.mkdir(exist_ok=True); AUDIT.mkdir(exist_ok=True)
    targets=load_targets(); pot=load_pot()
    missing=set(targets)-set(pot)
    if missing: raise RuntimeError(f'Target pipes absent from current POT: {sorted(missing)}')
    old,nodes,name_mismatches=load_old_graph(targets,pot)
    graph_ids={n['id'] for n in nodes}
    tech,dups=read_tech(graph_ids)
    tech['field']=tech['id'].map(lambda x: pot[x]['field'])
    # Some legacy technical-regime rows have no salt concentration at all.
    # The documented last resort is a measured field data median, not zero.
    fhs_min_median = fhs_mineralization_median()
    tech['Общая минерализация'] = pd.to_numeric(tech['Общая минерализация'], errors='coerce')
    tech.loc[tech['Общая минерализация'] <= 0, 'Общая минерализация'] = np.nan
    tech['Общая минерализация'] = tech['Общая минерализация'].fillna(fhs_min_median)
    # The complete SHTR workbook has two million rows.  It is read only when
    # the newer pipe technical regime lacks a valid T or P value.  This is a
    # source-priority decision, not a shortcut: direct pipe measurements win.
    tp_missing_before_shtr = int((~((tech['t'] > 5) & (tech['t'] <= 90) & (tech['p'] > 0.01) & (tech['p'] <= 9))).sum())
    shtr = pd.DataFrame(columns=['id', 'date', 't_shtr', 'p_shtr'])
    if tp_missing_before_shtr:
        shtr=read_shtr_sources(old)
        if not shtr.empty:
            tech=tech.merge(shtr,on=['id','date'],how='left')
            missing_t = ~((tech['t'] > 5) & (tech['t'] <= 90))
            missing_p = ~((tech['p'] > 0.01) & (tech['p'] <= 9))
            tech.loc[missing_t, 't'] = tech.loc[missing_t, 't_shtr']
            tech.loc[missing_p, 'p'] = tech.loc[missing_p, 'p_shtr']
    edges=[{'source':norm_id(e['source']),'target':norm_id(e['target']),'via_node':text(e.get('via_node')),'status':text(e.get('status','confirmed'))} for e in old['edges']]
    prepared,graph_rep=apply_tp_and_own_fill(tech,edges)
    prepared['flow_direction_coef']=1
    prepared['dns_paot_pipe_kind']='naporny'
    prepared['id простого участка']=prepared['id']
    prepared['id основного участка']=prepared['id'].map(lambda x:pot[x]['main_id'])
    prepared['Дата']=prepared['date']
    prepared['source_t']=prepared['source_t'].replace({'трубный техрежим':'трубный техрежим: температура в начале'})
    prepared['source_p']=prepared['source_p'].replace({'трубный техрежим':'трубный техрежим: P фактическое начало'})
    selected=prepared[prepared['id'].isin(targets)].copy()
    chem,chem_rep=build_chemistry(selected,pot,set(targets))
    requested={'meta':{'dataset':'megion_67_windows_prepared_2026_09_18','source_tech':str(TECH_XLSX),'pot_sha256':sha256(POT_XLSX),'rules':'T/P: direct technical regime, then confirmed SHTR, then confirmed graph predecessor; other parameters only same pipe; temporal interpolation/seasonal/pipe mean/field mean.'},'by_id':{}}
    keep=['Жидкости, кг/м3','Жидкости, м3/сут (дебит)','Обводненность, %','Нефти, т/сут (дебит)','Газа, кг/м3','Газа, кг/(м*с)*1000','Жидкости, кг/(м*с)','Общего газа, тыс.м3/сут (дебит)','Нефти, кг/м3','Дата','id простого участка','id основного участка','L','D','S','flow_direction_coef','t','source_t','p','source_p','dns_paot_pipe_kind']
    for pid, part in selected.groupby('id',sort=True):
        q=part.sort_values('date')[keep].replace({np.nan:None})
        daily=q.to_dict('records')
        requested['by_id'][pid]={'daily':daily,'records_count':len(daily),'date_min':daily[0]['Дата'],'date_max':daily[-1]['Дата']}
    master={'meta':{'dataset':'megion_67_confirmed_graph_2026_09_18','target_pipes':67,'conditional_bridge_pipes':7,'pot_sha256':sha256(POT_XLSX),'surrounding_temperature_c':5.0},'nodes':nodes,'edges':edges}
    (OUT/'graph_megion_master.json').write_text(json.dumps(master,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'graph_megion_requested_daily_params.json').write_text(json.dumps(requested,ensure_ascii=False),encoding='utf-8')
    chem_out=chem[['id','date','CO2','Общая минерализация','pH']].rename(columns={'id':'id простого участка','date':'Дата контроля'})
    chem_out['Общая минерализация, г/л']=chem_out['Общая минерализация']/1000
    write_csv(OUT/'chem_daily_megion.csv',chem_out)
    write_csv(OUT/'h2s_two_phase_daily_base.csv',chem[['id','date','CO2','H2S in Water Phase','H2S in Gas Phase']].rename(columns={'CO2':'CO2 in Water Phase'}))
    write_csv(OUT/'kvch_daily_megion.csv',selected[['id','date']].assign(kvch_mg_l=np.nan))
    write_csv(AUDIT/'duplicate_tech_rows.csv',pd.DataFrame(dups))
    write_csv(AUDIT/'graph_fill.csv',pd.DataFrame(graph_rep))
    coverage=[]
    for pid,part in selected.groupby('id'):
        active=(pd.to_numeric(part['Жидкости, м3/сут (дебит)'],errors='coerce')>0)|(pd.to_numeric(part['Нефти, т/сут (дебит)'],errors='coerce')>0)|(pd.to_numeric(part['Общего газа, тыс.м3/сут (дебит)'],errors='coerce')>0)
        coverage.append({'ID простого участка':pid,'Наименование':pot[pid]['name'],'Дат всего':len(part),'Активных дат':int(active.sum()),'T из техрежима':int(part['source_t'].str.startswith('трубный').sum()),'T по графу':int(part['source_t'].str.startswith('граф').sum()),'T временно дозаполнено':int((~part['source_t'].str.startswith(('трубный','граф'))).sum()),'P из техрежима':int(part['source_p'].str.startswith('трубный').sum()),'P по графу':int(part['source_p'].str.startswith('граф').sum()),'P временно дозаполнено':int((~part['source_p'].str.startswith(('трубный','граф'))).sum())})
    coverage_df=pd.DataFrame(coverage).sort_values('ID простого участка')
    write_csv(AUDIT/'предварительное_покрытие_по_трубам.csv',coverage_df)
    report={'status':'PASS','target_pipes':67,'graph_nodes':len(nodes),'graph_edges':len(edges),'technical_rows_graph_scope':len(tech),'technical_rows_target_scope':len(selected),'source_files':{str(p.relative_to(ROOT)):sha256(p) for p in [TARGET_XLSX,POT_XLSX,TECH_XLSX,SHTR_XLSX,FHS_XLSX,OLD_GRAPH]},'fhs_mineralization_median_mg_l_for_missing_tech_values':fhs_min_median,'pot_name_mismatches_vs_confirmed_graph':name_mismatches,'tech_duplicate_id_date_rows_discarded':len(dups),'tp_rows_missing_before_shtr':tp_missing_before_shtr,'shtr_rows_used':int(shtr.shape[0]),'chemistry':chem_rep,'coverage_totals':{'target_active_dates':int(coverage_df['Активных дат'].sum()),'temperature_direct':int(coverage_df['T из техрежима'].sum()),'temperature_graph':int(coverage_df['T по графу'].sum()),'temperature_temporal':int(coverage_df['T временно дозаполнено'].sum()),'pressure_direct':int(coverage_df['P из техрежима'].sum()),'pressure_graph':int(coverage_df['P по графу'].sum()),'pressure_temporal':int(coverage_df['P временно дозаполнено'].sum())},'limitations':['Windows PE2 is not executable on macOS; this preflight verifies prepared inputs only.','Only confirmed 2026-08 graph edges and confirmed SHTR links are used when direct pipe T/P is unavailable.','Chemistry transfer between different directions is not yet inferred from a common DNS: the current package first preserves only approved UKK-to-direction coverage. Any DNS donor expansion must be explicitly audited before it is enabled.']}
    (AUDIT/'PRE_WINDOWS_MEGION_INPUT_REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
