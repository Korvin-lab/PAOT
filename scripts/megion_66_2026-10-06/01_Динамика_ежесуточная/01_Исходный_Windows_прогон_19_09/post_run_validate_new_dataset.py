from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--no-gaps-report", type=Path, required=True)
    parser.add_argument("--temperature-segment-report", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    no_gaps = json.loads(args.no_gaps_report.read_text(encoding="utf-8"))
    temperature_segments = json.loads(args.temperature_segment_report.read_text(encoding="utf-8"))
    expected_ids = int(summary.get("requested_pipes_total", 0))
    minimum_pressure_mpa = float(summary.get("minimum_pressure_mpa", 0.01))
    minimum_seg_pressure = minimum_pressure_mpa * 10.0

    rows = 0
    ids = set()
    nulls = 0
    infinities = 0
    low_pressure_rows = 0
    nonpositive_viscosity_rows = 0
    nonpositive_seg_muw_rows = 0
    adjacent_duplicate_keys = 0
    last_key = None
    extrema = {}
    required = {
        "date", "id", "segment_id", "seg_p_start", "viscosity_liquid_work", "seg_muw",
        "Qgas_work_m3s_polytech", "seg_v_mix_polytech", "seg_reynolds_polytech", "seg_knns_polytech",
    }

    for chunk in pd.read_csv(args.final_csv, chunksize=750_000, low_memory=False):
        rows += len(chunk)
        missing = required - set(chunk.columns)
        if missing:
            raise RuntimeError(f"Final CSV misses required columns: {sorted(missing)}")
        ids.update(chunk["id"].astype(str).unique())
        nulls += int(chunk.isna().sum().sum())
        numeric = chunk.select_dtypes(include=[np.number])
        infinities += int(np.isinf(numeric.to_numpy(dtype=float, copy=False)).sum())

        pressure = pd.to_numeric(chunk["seg_p_start"], errors="coerce")
        low_pressure_rows += int((pressure <= minimum_seg_pressure).sum())
        visc = pd.to_numeric(chunk["viscosity_liquid_work"], errors="coerce")
        seg_muw = pd.to_numeric(chunk["seg_muw"], errors="coerce")
        nonpositive_viscosity_rows += int((visc <= 0).sum())
        nonpositive_seg_muw_rows += int((seg_muw <= 0).sum())

        keys = list(zip(chunk["id"].astype(str), chunk["date"].astype(str), chunk["segment_id"].astype(str)))
        if keys:
            if last_key == keys[0]:
                adjacent_duplicate_keys += 1
            adjacent_duplicate_keys += sum(a == b for a, b in zip(keys, keys[1:]))
            last_key = keys[-1]

        for column in ["seg_p_start", "Qgas_work_m3s_polytech", "seg_v_mix_polytech", "seg_reynolds_polytech", "seg_knns_polytech"]:
            values = pd.to_numeric(chunk[column], errors="coerce")
            current = extrema.setdefault(column, {"min": None, "max": None})
            if values.notna().any():
                vmin = float(values.min())
                vmax = float(values.max())
                current["min"] = vmin if current["min"] is None else min(current["min"], vmin)
                current["max"] = vmax if current["max"] is None else max(current["max"], vmax)

    checks = {
        "final_csv_exists_and_nonempty": args.final_csv.is_file() and rows > 0,
        "all_requested_pipes_present": len(ids) == expected_ids,
        "no_nulls": nulls == 0,
        "no_infinities": infinities == 0,
        "no_pressure_at_or_below_configured_minimum": low_pressure_rows == 0,
        "no_nonpositive_working_viscosity": nonpositive_viscosity_rows == 0,
        "no_nonpositive_seg_muw": nonpositive_seg_muw_rows == 0,
        "no_adjacent_duplicate_id_date_segment": adjacent_duplicate_keys == 0,
        "strict_no_missing_pipes": int(summary.get("missing_or_not_calculated_pipes_total", -1)) == 0,
        "no_gaps_report_zero_nulls": int(no_gaps.get("counts", {}).get("total_nulls_after", -1)) == 0,
        "temperature_and_segments_validation_passed": temperature_segments.get("status") == "PASS",
        "temperature_and_segments_have_no_errors": int(
            temperature_segments.get("counts", {}).get("error_events", -1)
        ) == 0,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "rows": rows,
        "ids": len(ids),
        "minimum_pressure_mpa": minimum_pressure_mpa,
        "minimum_seg_p_start_expected": minimum_seg_pressure,
        "nulls": nulls,
        "infinities": infinities,
        "low_pressure_rows": low_pressure_rows,
        "nonpositive_viscosity_rows": nonpositive_viscosity_rows,
        "nonpositive_seg_muw_rows": nonpositive_seg_muw_rows,
        "adjacent_duplicate_keys": adjacent_duplicate_keys,
        "temperature_and_segments_validation": {
            "status": temperature_segments.get("status"),
            "rows": temperature_segments.get("counts", {}).get("rows"),
            "profiles_id_date": temperature_segments.get("counts", {}).get("profiles_id_date"),
            "pipes": temperature_segments.get("counts", {}).get("pipes"),
            "error_events": temperature_segments.get("counts", {}).get("error_events"),
        },
        "extrema": extrema,
    }
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
