from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Summarize pipe-days at or below configured minimum pressure.')
    p.add_argument('--out-dir', type=Path, required=True, help='Pipeline output folder (contains zero csv).')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    zero_csv = out_dir / 'расчет_сегментов__уход_в_ноль.csv'
    zero_days_csv = out_dir / 'трубы_и_даты__исключены_из_за_p_le_0_01MPa.csv'

    if zero_days_csv.exists():
        dropped = pd.read_csv(zero_days_csv)
        dropped['id'] = dropped['id'].astype(str)
        dropped['date'] = pd.to_datetime(dropped['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        dropped = dropped.dropna(subset=['date']).drop_duplicates(['id', 'date'])
    elif zero_csv.exists():
        z = pd.read_csv(zero_csv)
        z['id'] = z['id'].astype(str)
        z['date'] = pd.to_datetime(z['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        dropped = z[z['metric'].astype(str) == 'p_day_dropped_below_min'][['id', 'date']].dropna().drop_duplicates(['id', 'date'])
    else:
        dropped = pd.DataFrame(columns=['id', 'date'])

    summary = (
        dropped.groupby('id', as_index=False)
        .agg(
            bad_dates_count=('date', 'nunique'),
            date_min=('date', 'min'),
            date_max=('date', 'max'),
        )
        .sort_values(['bad_dates_count', 'id'], ascending=[False, True])
    ) if not dropped.empty else pd.DataFrame(columns=['id', 'bad_dates_count', 'date_min', 'date_max'])

    out_rows = out_dir / 'итог__трубы_и_даты__p_le_0_01MPa.csv'
    out_summary = out_dir / 'итог__сводка__трубы_с_p_le_0_01MPa.csv'
    dropped.to_csv(out_rows, index=False)
    summary.to_csv(out_summary, index=False)

    print('pipes_with_bad_days', int(summary['id'].nunique()) if not summary.empty else 0)
    print('bad_id_date_rows', int(len(dropped)))
    print('rows_file', out_rows)
    print('summary_file', out_summary)


if __name__ == '__main__':
    main()
