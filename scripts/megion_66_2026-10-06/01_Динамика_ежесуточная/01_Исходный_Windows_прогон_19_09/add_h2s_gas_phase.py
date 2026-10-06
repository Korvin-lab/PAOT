"""Append H2S gas phase to a final Orenburg CSV without changing its 50 fields."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_SERIES = ROOT / "input" / "h2s_two_phase_daily_base.csv"
SOURCE_COLUMNS = 50
NEW_COLUMN = "H2S in Gas Phase"


def check_space(source: Path, destination: Path) -> None:
    required = int(source.stat().st_size * 1.05)
    available = shutil.disk_usage(destination.parent).free
    if available < required:
        raise OSError(f"Need at least {required} free bytes for a new CSV; only {available} are available")


def collect_keys(source: Path) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = defaultdict(set)
    with source.open("rb", buffering=16 * 1024 * 1024) as handle:
        header = handle.readline().decode("utf-8-sig").strip("\r\n").split(",")
        if len(header) != SOURCE_COLUMNS or NEW_COLUMN in header:
            raise ValueError("Input must be the original 50-column final CSV without H2S gas phase")
        if header[0:2] != ["date", "id"]:
            raise ValueError("Unexpected final CSV key columns")
        for line in handle:
            parts = line.rstrip(b"\r\n").split(b",")
            if len(parts) != SOURCE_COLUMNS or b'"' in line:
                raise ValueError("Unexpected CSV quoting or schema; do not rewrite this dataset with this byte-preserving tool")
            keys[parts[1].decode()].add(parts[0].decode())
    return keys


def build_lookup(keys_by_id: dict[str, set[str]]) -> tuple[dict[tuple[bytes, bytes], bytes], pd.DataFrame]:
    base = pd.read_csv(BASE_SERIES, dtype={"id": str, "date": str})
    needed = {"id", "date", NEW_COLUMN}
    if not needed <= set(base.columns):
        raise ValueError(f"Base H2S series misses columns: {sorted(needed - set(base.columns))}")
    lookup: dict[tuple[bytes, bytes], bytes] = {}
    audit_rows = []
    for pipe_id, raw_dates in sorted(keys_by_id.items()):
        dates = pd.DatetimeIndex(pd.to_datetime(sorted(raw_dates), format="%Y-%m-%d", errors="raise"))
        source = base.loc[base.id.eq(pipe_id), ["date", NEW_COLUMN]].copy()
        source["date"] = pd.to_datetime(source["date"], format="%Y-%m-%d", errors="raise")
        source = source.drop_duplicates("date").set_index("date")[NEW_COLUMN].astype(float)
        source = source.where(source > 0).dropna()
        if source.empty:
            raise ValueError(f"No prepared H2S gas observations/donor series for pipe {pipe_id}")
        index = dates.union(source.index).sort_values()
        values = source.reindex(index)
        existing = values.notna()
        filled = values.interpolate(method="time", limit_area="inside")
        origin = pd.Series("", index=index, dtype=object)
        origin.loc[existing] = "prepared_two_phase_series"
        origin.loc[~existing & filled.notna()] = "same_pipe_time_interpolation"
        before_first, after_last = index < source.index.min(), index > source.index.max()
        filled = filled.bfill().ffill()
        origin.loc[before_first & origin.eq("")] = "same_pipe_nearest_first"
        origin.loc[after_last & origin.eq("")] = "same_pipe_nearest_last"
        selected = filled.reindex(dates)
        selected_origin = origin.reindex(dates)
        if selected.isna().any() or not np.isfinite(selected.to_numpy(float)).all():
            raise ValueError(f"H2S gas remains missing for pipe {pipe_id}")
        if not (selected > 0).all():
            raise ValueError(f"H2S gas contains zero/negative values for pipe {pipe_id}")
        for day, value, how in zip(dates.strftime("%Y-%m-%d"), selected, selected_origin):
            lookup[(pipe_id.encode(), day.encode())] = repr(float(value)).encode()
            audit_rows.append({"id": pipe_id, "date": day, NEW_COLUMN: float(value), "origin": how})
    return lookup, pd.DataFrame(audit_rows)


def append(source: Path, destination: Path) -> dict:
    if not BASE_SERIES.is_file():
        raise FileNotFoundError(f"Missing prepared two-phase chemistry series: {BASE_SERIES}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_space(source, destination)
    keys = collect_keys(source)
    lookup, audit = build_lookup(keys)
    audit.to_csv(destination.parent / "H2S_GAS_PHASE_DAILY_AUDIT.csv", index=False, encoding="utf-8-sig")
    temp = destination.with_suffix(destination.suffix + ".partial")
    if temp.exists():
        temp.unlink()
    source_hash, output_hash, unchanged_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    rows = 0
    with source.open("rb", buffering=16 * 1024 * 1024) as src, temp.open("wb", buffering=16 * 1024 * 1024) as out:
        header = src.readline()
        source_hash.update(header)
        output_header = header.rstrip(b"\r\n") + b"," + NEW_COLUMN.encode() + b"\n"
        output_hash.update(output_header)
        unchanged_hash.update(header)
        out.write(output_header)
        for line in src:
            source_hash.update(line)
            fields = line.rstrip(b"\r\n").split(b",")
            if len(fields) != SOURCE_COLUMNS or b'"' in line:
                raise ValueError("Unexpected CSV quoting or schema while appending H2S gas")
            value = lookup.get((fields[1], fields[0]))
            if value is None:
                raise ValueError(f"No H2S gas value for {fields[1].decode()} / {fields[0].decode()}")
            output_line = line.rstrip(b"\r\n") + b"," + value + b"\n"
            unchanged_hash.update(line)
            output_hash.update(output_line)
            out.write(output_line)
            rows += 1
    temp.replace(destination)
    report = {
        "status": "PASS",
        "input": str(source),
        "output": str(destination),
        "rows": rows,
        "pipes": len(keys),
        "columns_before": SOURCE_COLUMNS,
        "columns_after": SOURCE_COLUMNS + 1,
        "h2s_gas_unit": "mg/m3",
        "input_sha256": source_hash.hexdigest(),
        "output_sha256": output_hash.hexdigest(),
        "unchanged_50_columns_sha256": unchanged_hash.hexdigest(),
        "daily_origins": audit.origin.value_counts().to_dict(),
    }
    (destination.parent / "H2S_GAS_PHASE_APPEND_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(append(args.source, args.destination), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
