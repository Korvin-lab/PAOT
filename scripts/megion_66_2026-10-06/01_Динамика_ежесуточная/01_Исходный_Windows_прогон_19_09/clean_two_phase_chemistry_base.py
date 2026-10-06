"""Make the prepared two-phase chemistry base positive and gap-free per pipe."""
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "input" / "h2s_two_phase_daily_base.csv"
RAW_BACKUP = ROOT / "input" / "h2s_two_phase_daily_base__RAW_BEFORE_POSITIVE_FILL.csv"
REPORT = ROOT / "input" / "h2s_two_phase_daily_base_positive_fill_report.json"
VALUE_COLUMNS = ["CO2 in Water Phase", "H2S in Water Phase", "H2S in Gas Phase"]


def fill_one(group: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    dates = pd.to_datetime(group["date"], format="%Y-%m-%d", errors="raise")
    values = pd.to_numeric(group[column], errors="coerce").astype(float)
    series = pd.Series(values.to_numpy(), index=dates)
    positive = series.where(series > 0).dropna()
    if positive.empty:
        raise ValueError(f"No positive values for id={group['id'].iloc[0]} column={column}")
    filled = series.where(series > 0).interpolate(method="time", limit_area="inside")
    origin = pd.Series("", index=series.index, dtype=object)
    origin.loc[series > 0] = "source_positive"
    origin.loc[(series <= 0) & filled.notna()] = "time_interpolation"
    before_first = series.index < positive.index.min()
    after_last = series.index > positive.index.max()
    filled = filled.bfill().ffill()
    origin.loc[(series <= 0) & before_first] = "nearest_first_value_at_left_edge"
    origin.loc[(series <= 0) & after_last] = "nearest_last_value_at_right_edge"
    if filled.isna().any() or not (filled > 0).all():
        raise AssertionError(f"{column} still contains missing/non-positive values for id={group['id'].iloc[0]}")
    return filled.reset_index(drop=True), origin.reset_index(drop=True)


def main() -> None:
    if not BASE.is_file():
        raise FileNotFoundError(BASE)
    if not RAW_BACKUP.exists():
        shutil.copy2(BASE, RAW_BACKUP)
    df = pd.read_csv(BASE, dtype={"id": str, "date": str})
    missing = sorted(set(VALUE_COLUMNS + ["id", "date"]) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    report = {"status": "PASS", "rows": int(len(df)), "pipes": int(df["id"].nunique()), "columns": {}}
    for column in VALUE_COLUMNS:
        before = pd.to_numeric(df[column], errors="coerce")
        origins = []
        filled_parts = []
        for _, group in df.groupby("id", sort=False):
            filled, origin = fill_one(group, column)
            filled_parts.append(filled)
            origins.append(origin)
        df[column] = pd.concat(filled_parts, ignore_index=True).to_numpy(float)
        origin_col = f"{column} positive fill origin"
        df[origin_col] = pd.concat(origins, ignore_index=True).to_numpy(object)
        after = pd.to_numeric(df[column], errors="coerce")
        report["columns"][column] = {
            "zeros_before": int((before == 0).sum()),
            "nonpositive_before": int((before <= 0).sum()),
            "nan_before": int(before.isna().sum()),
            "zeros_after": int((after == 0).sum()),
            "nonpositive_after": int((after <= 0).sum()),
            "nan_after": int(after.isna().sum()),
            "min_after": float(after.min()),
            "max_after": float(after.max()),
            "origins": dict(Counter(df[origin_col])),
        }
    if not np.isfinite(df[VALUE_COLUMNS].to_numpy(float)).all():
        raise AssertionError("Non-finite values remain in two-phase chemistry base")
    temp = BASE.with_suffix(".positive_fill.tmp.csv")
    df.to_csv(temp, index=False, encoding="utf-8-sig")
    temp.replace(BASE)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
