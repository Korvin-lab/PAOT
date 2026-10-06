from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--final-csv', type=Path, required=True)
    p.add_argument('--report-json', type=Path, required=True)
    p.add_argument('--chunksize', type=int, default=750000)
    p.add_argument('--temperature-min', type=float, default=5.0)
    p.add_argument('--floor-epsilon', type=float, default=0.05)
    p.add_argument('--max-floor-share-total', type=float, default=0.005)
    p.add_argument('--fail-on-floor-like-share', action='store_true')
    args = p.parse_args()

    required = ['date', 'id', 'seg_temperature', 'CO2 in Water Phase', 'H2S in Water Phase', 'H2S in Gas Phase', 'Min', 'pH']
    rows = 0
    ids = set()
    counts = Counter()
    extrema = {}

    for chunk in pd.read_csv(args.final_csv, chunksize=args.chunksize, low_memory=False):
        missing = set(required) - set(chunk.columns)
        if missing:
            raise ValueError(f'Missing columns: {sorted(missing)}')
        rows += len(chunk)
        ids.update(chunk['id'].astype(str).unique())
        for col in required[2:]:
            s = pd.to_numeric(chunk[col], errors='coerce')
            counts[f'{col} nan'] += int(s.isna().sum())
            if col in {'CO2 in Water Phase', 'H2S in Water Phase', 'H2S in Gas Phase', 'Min'}:
                counts[f'{col} zero_or_negative'] += int(s.le(0).sum())
            cur = extrema.setdefault(col, {'min': None, 'max': None})
            if s.notna().any():
                vmin, vmax = float(s.min()), float(s.max())
                cur['min'] = vmin if cur['min'] is None else min(cur['min'], vmin)
                cur['max'] = vmax if cur['max'] is None else max(cur['max'], vmax)
        t = pd.to_numeric(chunk['seg_temperature'], errors='coerce')
        counts['seg_temperature floor_like'] += int(t.le(args.temperature_min + args.floor_epsilon).sum())
        counts['seg_temperature below_min'] += int(t.lt(args.temperature_min).sum())

    floor_share = counts['seg_temperature floor_like'] / rows if rows else 0.0
    checks = {
        'has_51_required_columns': True,
        'no_required_nan': all(v == 0 for k, v in counts.items() if k.endswith(' nan')),
        'no_nonpositive_chemistry': all(v == 0 for k, v in counts.items() if k.endswith(' zero_or_negative')),
        'no_temperature_below_min': counts['seg_temperature below_min'] == 0,
        'temperature_floor_like_share_ok': (floor_share <= args.max_floor_share_total) if args.fail_on_floor_like_share else True,
    }
    report = {
        'status': 'PASS' if all(checks.values()) else 'FAIL',
        'rows': rows,
        'pipes': len(ids),
        'checks': checks,
        'counts': dict(counts),
        'temperature_floor_like_share': floor_share,
        'limits': {
            'floor_like_temperature_c': args.temperature_min + args.floor_epsilon,
            'max_floor_share_total': args.max_floor_share_total,
            'fail_on_floor_like_share': bool(args.fail_on_floor_like_share),
            'note': 'Near-ambient values are reported, but not a failure unless --fail-on-floor-like-share is used.',
        },
        'extrema': extrema,
    }
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
