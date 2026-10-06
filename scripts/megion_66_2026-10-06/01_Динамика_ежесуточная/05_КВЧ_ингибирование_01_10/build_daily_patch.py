"""Build auditable daily KVCH and inhibition values for the already-active Megion keys."""
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import csv
import json
import math
import statistics

import numpy as np
import openpyxl
import pandas as pd
import xlrd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "Данные для сбора датасета Мегион"
ACTIVE = HERE.parent / "megion_mineralization_rebuild_2026-09-30/daily_min_active_patch.csv"
LINKS = HERE / "link_audit.json"
OUT = HERE / "daily_patch.csv"


def norm_id(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return ""


def number(value):
    try:
        result = float(str(value).replace(",", "."))
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def date(value, datemode=None):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, (float, int)) and datemode is not None:
        return xlrd.xldate.xldate_as_datetime(value, datemode).date().isoformat()
    return pd.Timestamp(value).date().isoformat()


def filled_direction_series(observations, dates, positive_only):
    if not observations:
        return pd.Series(np.nan, index=dates), pd.Series("no_source", index=dates)
    values = pd.Series({pd.Timestamp(day): statistics.median(items)
                        for day, items in observations.items()}, dtype=float).sort_index()
    if positive_only:
        assert (values > 0).all()
    else:
        assert (values >= 0).all()
    index = values.index.union(dates).sort_values()
    series = values.reindex(index).interpolate(method="time", limit_area="inside")
    series = series.bfill().ffill().reindex(dates)
    origin = pd.Series("interpolated", index=dates)
    origin.loc[dates < values.index.min()] = "nearest_first"
    origin.loc[dates > values.index.max()] = "nearest_last"
    origin.loc[dates.intersection(values.index)] = "sample_date"
    assert series.notna().all()
    return series, origin


def main():
    links = json.loads(LINKS.read_text())
    assert links["counts"]["field_mismatch_target"] == 0
    assert links["counts"]["multi_direction_wells_target"] == 0
    with ACTIVE.open(newline="") as fh:
        keys = pd.DataFrame((r["id"], r["date"]) for r in csv.DictReader(fh))
    keys.columns = ["id", "date"]
    assert len(keys) == 71460 and not keys.duplicated().any()
    keys["date"] = pd.to_datetime(keys["date"], format="%Y-%m-%d")
    max_date = keys.date.max()

    # The package's audited POT is authoritative for simple-section directions.
    import importlib.util
    package = HERE.parent / "MEGION_WINDOWS_FULL_PE2__PREPARING_2026-09-18"
    spec = importlib.util.spec_from_file_location("megion_builder", package / "build_megion_windows_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pot = module.load_pot()
    keys["direction"] = keys.id.map(lambda pid: pot[pid]["direction_id"])
    assert keys.direction.notna().all()

    by_well = links["well_direction_links"]
    kvch = defaultdict(lambda: defaultdict(list))
    counts = Counter()
    workbook = openpyxl.load_workbook(DATA / "ФХС 2015-2026 MEGION.xlsx", read_only=True, data_only=True)
    for row in workbook.active.iter_rows(min_row=2, values_only=True):
        well = norm_id(row[0])
        value = number(row[33]) if len(row) > 33 else None
        if well not in by_well or value is None or value <= 0 or row[6] is None:
            continue
        sample_date = date(row[6])
        if pd.Timestamp(sample_date) > max_date:
            counts["kvch_future_samples_excluded"] += 1
            continue
        for direction in by_well[well]:
            kvch[direction][sample_date].append(value)
            counts["kvch_source_samples_used"] += 1
    workbook.close()
    assert counts["kvch_source_samples_used"] > 10000

    # Quotes are part of the XLS header text, so normalize exact names first.
    book = xlrd.open_workbook(str(DATA / "Ингибирование Мегион.xls"))
    ing = defaultdict(lambda: defaultdict(list))
    for sheet in book.sheets():
        headers = [str(v).strip().strip('"').strip() for v in sheet.row_values(0)]
        required = ("ID простого участка", "Дата расчета", "Дозировка уч факт г/м3", "Дозировка сегмент план")
        assert all(name in headers for name in required), (sheet.name, headers)
        ix = {name: headers.index(name) for name in required}
        for rn in range(1, sheet.nrows):
            row = sheet.row_values(rn)
            pipe = norm_id(row[ix["ID простого участка"]])
            if pipe not in pot:
                continue
            actual = number(row[ix["Дозировка уч факт г/м3"]])
            planned = number(row[ix["Дозировка сегмент план"]])
            if actual is None or actual < 0 or planned is None or planned <= 0:
                continue
            sample_date = date(row[ix["Дата расчета"]], book.datemode)
            if pd.Timestamp(sample_date) > max_date:
                continue
            direction = pot[pipe]["direction_id"]
            if direction in set(keys.direction):
                ing[direction][sample_date].append(actual / planned)
                counts["ing_source_rows_used"] += 1
    assert counts["ing_source_rows_used"] == 770

    result = keys.copy()
    result["kvch"] = np.nan
    result["ing_factor"] = np.nan
    method_counts = Counter()
    direction_report = []
    for direction, loc in result.groupby("direction"):
        dates = pd.DatetimeIndex(sorted(loc.date.unique()))
        kv, kv_method = filled_direction_series(kvch.get(direction), dates, True)
        factor, factor_method = filled_direction_series(ing.get(direction), dates, False)
        idx = loc.index
        result.loc[idx, "kvch"] = kv.reindex(result.loc[idx, "date"]).to_numpy()
        result.loc[idx, "ing_factor"] = factor.reindex(result.loc[idx, "date"]).to_numpy()
        method_counts.update({"kvch_" + name: int(value) for name, value in kv_method.value_counts().items()})
        method_counts.update({"ing_" + name: int(value) for name, value in factor_method.value_counts().items()})
        direction_report.append({"direction": direction, "pipes": int(loc.id.nunique()), "active_pipe_dates": len(loc),
                                 "kvch_sample_dates": len(kvch.get(direction, {})), "kvch_wells": links["directions"][direction]["wells"],
                                 "kvch_filled_pipe_dates": int(result.loc[idx, "kvch"].notna().sum()),
                                 "ing_sample_dates": len(ing.get(direction, {})),
                                 "ing_filled_pipe_dates": int(result.loc[idx, "ing_factor"].notna().sum())})
    result["date"] = result.date.dt.strftime("%Y-%m-%d")
    result = result.sort_values(["id", "date"])
    assert len(result) == 71460 and not result.duplicated(["id", "date"]).any()
    assert result.kvch.dropna().gt(0).all()
    assert result.ing_factor.dropna().ge(0).all()
    result[["id", "date", "kvch", "ing_factor"]].to_csv(OUT, index=False)
    report = {"active_keys": len(result), "kvch_filled_keys": int(result.kvch.notna().sum()),
              "ing_filled_keys": int(result.ing_factor.notna().sum()),
              "kvch_pipes": int(result.groupby("id").kvch.apply(lambda s: s.notna().any()).sum()),
              "ing_pipes": int(result.groupby("id").ing_factor.apply(lambda s: s.notna().any()).sum()),
              "counts": dict(counts), "methods": dict(method_counts), "directions": direction_report,
              "kvch_rule": "Exact OIS well ID -> verified source-list direction; positive well samples only; median by direction and sample date; interpolate within, nearest at edges. Historic pre-2024 mapping inferred from stable 2024-26 membership.",
              "ing_rule": "Quoted XLS headers normalized; section actual g/m3 / segment planned g/m3; median by direction and date; interpolate within, nearest at edges. Plan-as-normative is an operational assumption, not a certified regulatory dose.",
              "limits": "Only directions with own source data are filled; no inter-direction transfer or gas/oil phase conversion."}
    (HERE / "daily_patch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report[k] for k in ("active_keys", "kvch_filled_keys", "ing_filled_keys", "kvch_pipes", "ing_pipes", "counts", "methods")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
