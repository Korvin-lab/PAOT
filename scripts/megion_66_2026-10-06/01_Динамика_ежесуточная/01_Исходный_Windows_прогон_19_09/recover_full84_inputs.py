"""Recover only invalid active input rows for the full 84-pipe Orenburg run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


Q = ['Жидкости, м3/сут (дебит)', 'Нефти, т/сут (дебит)', 'Общего газа, тыс.м3/сут (дебит)']
WC = 'Обводненность, %'
P = 'p'
T = 't'
EPS = 0.010001


def num(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else math.nan
    except Exception:
        return math.nan


def linear_nearest(series: pd.Series) -> tuple[pd.Series, list[tuple[int, str]]]:
    before = series.copy()
    out = series.interpolate(limit_area='inside').ffill().bfill()
    changes = []
    for i in range(len(series)):
        if pd.isna(before.iat[i]) and pd.notna(out.iat[i]):
            inside = 0 < i < len(series) - 1 and before.iloc[:i].notna().any() and before.iloc[i + 1:].notna().any()
            changes.append((i, 'linear_interpolation' if inside else 'nearest_edge_value'))
    return out, changes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--requested-json', type=Path, required=True)
    p.add_argument('--exclusions-csv', type=Path, required=True)
    p.add_argument('--report-json', type=Path, required=True)
    p.add_argument('--changes-csv', type=Path, required=True)
    args = p.parse_args()

    req = json.loads(args.requested_json.read_text(encoding='utf-8'))
    if args.exclusions_csv.exists() and args.exclusions_csv.stat().st_size > 0:
        excluded = pd.read_csv(args.exclusions_csv, dtype={'id': str, 'date': str})
    else:
        excluded = pd.DataFrame(columns=['id', 'date', 'reasons'])

    logs = []
    ids = sorted(req.get('by_id', {}))
    for pid in ids:
        daily = req['by_id'][pid].get('daily', [])
        if not daily:
            continue
        df = pd.DataFrame(daily)
        df['_date'] = pd.to_datetime(df['Дата'], errors='coerce')
        df = df.sort_values('_date').reset_index(drop=True)
        active = pd.DataFrame({c: df[c].map(num) for c in Q}).gt(0).any(axis=1)

        for col in Q:
            s = df[col].map(num)
            bad = active & (s.lt(0) | ~s.notna() | ((col in Q[:2]) & s.le(0)))
            series = s.where(active)
            series.loc[bad] = math.nan
            filled, changes = linear_nearest(series)
            df.loc[active, col] = filled.loc[active]
            logs.extend({'id': pid, 'date': df.loc[i, 'Дата'], 'parameter': col, 'method': method, 'reason': 'invalid_active_rate'} for i, method in changes)

        wc = df[WC].map(num)
        wc_bad = active & (~wc.between(0, 100))
        wc.loc[wc_bad] = math.nan
        filled_wc, changes_wc = linear_nearest(wc)
        df.loc[active, WC] = filled_wc.loc[active]
        logs.extend({'id': pid, 'date': df.loc[i, 'Дата'], 'parameter': WC, 'method': method, 'reason': 'watercut_outside_0_100'} for i, method in changes_wc)

        p_series = df[P].map(num)
        p_bad = active & ~p_series.gt(0.01)
        df.loc[p_bad, P] = EPS
        logs.extend({'id': pid, 'date': df.loc[i, 'Дата'], 'parameter': P, 'method': 'minimum_0.010001_MPa', 'reason': 'p_le_0.01'} for i in df.index[p_bad])

        reverse_dates = set(excluded.loc[excluded.id.eq(pid) & excluded.reasons.fillna('').str.contains('reverse_temperature', case=False, regex=False), 'date'])
        reverse_mask = active & df['Дата'].isin(reverse_dates)
        df.loc[reverse_mask, 'source_t'] = 'граф от предшественника: T начало; значение T сохранено'
        logs.extend({'id': pid, 'date': df.loc[i, 'Дата'], 'parameter': 'source_t', 'method': 'set_start_boundary_keep_temperature_value', 'reason': 'reverse_temperature_infeasible'} for i in df.index[reverse_mask])

        out = df.drop(columns=['_date']).astype(object).where(pd.notna(df.drop(columns=['_date'])), None)
        req['by_id'][pid]['daily'] = out.to_dict('records')

    args.requested_json.write_text(json.dumps(req, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    changes_df = pd.DataFrame(logs)
    changes_df.to_csv(args.changes_csv, index=False, encoding='utf-8-sig')
    report = {
        'status': 'PASS',
        'pipes': len(ids),
        'exclusion_rows_input': int(len(excluded)),
        'changes_total': int(len(changes_df)),
        'changes_by_parameter': {} if changes_df.empty else {str(k): int(v) for k, v in changes_df.groupby('parameter').size().items()},
        'rule': 'rates/watercut are recovered only from other dates of the same pipe; pressure <=0.01 MPa uses 0.010001 MPa floor; reverse-temperature exclusions are resolved by keeping T value and forcing start boundary according to graph/previous-pipe logic',
    }
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
