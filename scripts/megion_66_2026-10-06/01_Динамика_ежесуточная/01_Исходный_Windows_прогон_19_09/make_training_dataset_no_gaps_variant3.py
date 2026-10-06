from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

PSI_PER_PA = 0.00014503773773020923


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build training-ready CSV with no missing values from required_columns__filled.csv"
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=250_000)
    p.add_argument("--ph-max", type=float, default=10.0)
    return p.parse_args()


def _calc_comprw_kriel(
    t_c: pd.Series,
    p_atm: pd.Series,
    salinity_ppm: pd.Series,
) -> pd.Series:
    """
    Water compressibility (1/Pa), same correlation family as stage1.
    Inputs:
      - t_c: Celsius
      - p_atm: atmospheres
      - salinity_ppm: ppm
    """
    tv = pd.to_numeric(t_c, errors="coerce").to_numpy(dtype=float)
    pv_mpa = pd.to_numeric(p_atm, errors="coerce").to_numpy(dtype=float) / 10.0
    sv = pd.to_numeric(salinity_ppm, errors="coerce").to_numpy(dtype=float)

    out = np.full_like(tv, np.nan, dtype=float)
    valid = np.isfinite(tv) & np.isfinite(pv_mpa) & np.isfinite(sv) & (pv_mpa > 0)
    if np.any(valid):
        t_f = tv[valid] * 1.8 + 32.0
        p_psi = pv_mpa[valid] * 1e6 * PSI_PER_PA
        denom = 7.033 * p_psi + 0.5415 * sv[valid] - 537.0 * t_f + 403300.0
        with np.errstate(all="ignore"):
            cw = 0.1 * 145.04 / denom
        bad = (~np.isfinite(cw)) | (denom <= 0) | (cw <= 0)
        cw[bad] = np.nan
        out[valid] = cw
    return pd.Series(out, index=t_c.index, dtype=float)


def apply_ph_filter_and_repair(
    csv_path: Path,
    chunksize: int,
    ph_max: float,
) -> dict:
    """
    Enforce pH ceiling:
    - pH > ph_max -> NaN
    - fill internal gaps by linear interpolation per (id, date)
    - edges / unfillable -> per-id mean
    - if per-id mean absent -> global mean
    """
    # Build unique id+date table for pH repair.
    id_date_parts = []
    for ch in pd.read_csv(csv_path, usecols=["id", "date", "pH"], chunksize=chunksize, low_memory=False):
        id_date_parts.append(ch.drop_duplicates(["id", "date"], keep="last"))
    id_date = pd.concat(id_date_parts, ignore_index=True).drop_duplicates(["id", "date"], keep="last")

    id_date["id"] = id_date["id"].astype(str)
    id_date["date"] = pd.to_datetime(id_date["date"], errors="coerce")
    id_date["pH"] = pd.to_numeric(id_date["pH"], errors="coerce")
    id_date = id_date[id_date["date"].notna()].copy()

    bad_mask = id_date["pH"] > ph_max
    ph_over_limit_rows = int(bad_mask.sum())
    id_date.loc[bad_mask, "pH"] = pd.NA

    id_date = id_date.sort_values(["id", "date"]).reset_index(drop=True)
    id_date["pH_fixed"] = id_date["pH"]

    global_mean = float(id_date["pH"].mean(skipna=True)) if id_date["pH"].notna().any() else 0.0

    # Repair pH curve per id.
    for pid, idx in id_date.groupby("id", sort=False).groups.items():
        g = id_date.loc[idx, ["date", "pH_fixed"]].copy()
        g = g.sort_values("date")
        s = g["pH_fixed"].astype(float)

        # Internal linear interpolation only; edges stay NaN for mean-fill step.
        s_interp = s.interpolate(method="linear", limit_area="inside")
        id_mean = float(s_interp.mean(skipna=True)) if s_interp.notna().any() else global_mean
        s_final = s_interp.fillna(id_mean)
        id_date.loc[g.index, "pH_fixed"] = s_final.values

    # Last safety fallback.
    id_date["pH_fixed"] = pd.to_numeric(id_date["pH_fixed"], errors="coerce").fillna(global_mean)

    id_date["key"] = id_date["id"] + "|" + id_date["date"].dt.strftime("%Y-%m-%d")
    ph_map = pd.Series(id_date["pH_fixed"].values, index=id_date["key"].values)

    # Rewrite whole CSV with repaired pH.
    tmp_path = csv_path.with_name(csv_path.stem + "__tmp_phfix.csv")
    if tmp_path.exists():
        tmp_path.unlink()

    first = True
    rows_repaired = 0
    for ch in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        key = ch["id"].astype(str) + "|" + pd.to_datetime(ch["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        ch["pH"] = pd.to_numeric(key.map(ph_map), errors="coerce").fillna(global_mean)
        rows_repaired += len(ch)
        ch.to_csv(tmp_path, mode="w" if first else "a", header=first, index=False)
        first = False

    tmp_path.replace(csv_path)

    # Validation on repaired pH.
    ph_nulls_after = 0
    ph_over_limit_after = 0
    for ch in pd.read_csv(csv_path, usecols=["pH"], chunksize=chunksize, low_memory=False):
        s = pd.to_numeric(ch["pH"], errors="coerce")
        ph_nulls_after += int(s.isna().sum())
        ph_over_limit_after += int((s > ph_max).sum())

    return {
        "ph_max": float(ph_max),
        "ph_over_limit_rows_before_fix_on_id_date": ph_over_limit_rows,
        "rows_repaired": int(rows_repaired),
        "global_mean_ph_used_fallback": float(global_mean),
        "ph_nulls_after": int(ph_nulls_after),
        "ph_over_limit_rows_after": int(ph_over_limit_after),
    }


def main() -> None:
    args = parse_args()

    # Probe columns
    probe = pd.read_csv(args.input_csv, nrows=1000, low_memory=False)
    if "ing_sum" in probe.columns:
        probe = probe.drop(columns=["ing_sum"])
    cols = list(probe.columns)

    # Keep id/date as string-like; everything else try as numeric for gap filling.
    skip_numeric = {"id", "date"}
    numeric_cols = [c for c in cols if c not in skip_numeric]

    # 1st pass: collect per-id and global means for numeric columns.
    global_sum = {c: 0.0 for c in numeric_cols}
    global_cnt = {c: 0 for c in numeric_cols}
    per_id_sum = {c: defaultdict(float) for c in numeric_cols}
    per_id_cnt = {c: defaultdict(int) for c in numeric_cols}

    rows_total = 0
    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize, low_memory=False):
        if "ing_sum" in chunk.columns:
            chunk = chunk.drop(columns=["ing_sum"])
        rows_total += len(chunk)
        ids = chunk["id"].astype(str)

        # Normalize salinity/comprw semantics before mean collection:
        # - seg_salinity: if missing or <=0, fallback from Min
        # - seg_comprw: if missing or <=0, recompute from T/P/salinity
        if "seg_salinity" in chunk.columns and "Min" in chunk.columns:
            sal = pd.to_numeric(chunk["seg_salinity"], errors="coerce")
            mn = pd.to_numeric(chunk["Min"], errors="coerce")
            sal = sal.where(sal.notna() & (sal > 0), mn)
            chunk["seg_salinity"] = sal
        if {"seg_comprw", "seg_temperature", "seg_p_start", "seg_salinity"}.issubset(chunk.columns):
            cw = pd.to_numeric(chunk["seg_comprw"], errors="coerce")
            need = cw.isna() | (cw <= 0)
            if need.any():
                cw_calc = _calc_comprw_kriel(
                    t_c=pd.to_numeric(chunk["seg_temperature"], errors="coerce"),
                    p_atm=pd.to_numeric(chunk["seg_p_start"], errors="coerce"),
                    salinity_ppm=pd.to_numeric(chunk["seg_salinity"], errors="coerce"),
                )
                cw = cw.where(~need, cw_calc)
                chunk["seg_comprw"] = cw

        for c in numeric_cols:
            s = pd.to_numeric(chunk[c], errors="coerce")
            if c == "temperature" and "seg_temperature" in chunk.columns:
                s = s.fillna(pd.to_numeric(chunk["seg_temperature"], errors="coerce"))

            mask = s.notna()
            if not mask.any():
                continue

            v = s[mask]
            idv = ids[mask]
            global_sum[c] += float(v.sum())
            global_cnt[c] += int(mask.sum())

            g = pd.DataFrame({"id": idv.values, "v": v.values}).groupby("id", sort=False)["v"].agg(["sum", "count"])
            for pid, row in g.iterrows():
                per_id_sum[c][pid] += float(row["sum"])
                per_id_cnt[c][pid] += int(row["count"])

    global_mean = {
        c: (global_sum[c] / global_cnt[c]) if global_cnt[c] > 0 else 0.0
        for c in numeric_cols
    }
    per_id_mean = {
        c: {
            pid: (per_id_sum[c][pid] / per_id_cnt[c][pid])
            for pid in per_id_sum[c].keys()
            if per_id_cnt[c][pid] > 0
        }
        for c in numeric_cols
    }

    # 2nd pass: fill all NaN and write output.
    if args.out_csv.exists():
        args.out_csv.unlink()

    rows_written = 0
    first = True
    nulls_after_total = None

    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize, low_memory=False):
        if "ing_sum" in chunk.columns:
            chunk = chunk.drop(columns=["ing_sum"])
        ids = chunk["id"].astype(str)

        # Same semantic normalization in write pass.
        if "seg_salinity" in chunk.columns and "Min" in chunk.columns:
            sal = pd.to_numeric(chunk["seg_salinity"], errors="coerce")
            mn = pd.to_numeric(chunk["Min"], errors="coerce")
            sal = sal.where(sal.notna() & (sal > 0), mn)
            chunk["seg_salinity"] = sal
        if {"seg_comprw", "seg_temperature", "seg_p_start", "seg_salinity"}.issubset(chunk.columns):
            cw = pd.to_numeric(chunk["seg_comprw"], errors="coerce")
            need = cw.isna() | (cw <= 0)
            if need.any():
                cw_calc = _calc_comprw_kriel(
                    t_c=pd.to_numeric(chunk["seg_temperature"], errors="coerce"),
                    p_atm=pd.to_numeric(chunk["seg_p_start"], errors="coerce"),
                    salinity_ppm=pd.to_numeric(chunk["seg_salinity"], errors="coerce"),
                )
                cw = cw.where(~need, cw_calc)
                chunk["seg_comprw"] = cw

        # Fill id/date text holes too, to ensure absolutely no NaN in output.
        if "id" in chunk.columns:
            chunk["id"] = chunk["id"].astype(str).replace("nan", "")
        if "date" in chunk.columns:
            chunk["date"] = chunk["date"].astype(str).replace("nan", "")

        for c in numeric_cols:
            s = pd.to_numeric(chunk[c], errors="coerce")

            # Explicit rule from your pipeline notes: temperature can mirror seg_temperature.
            if c == "temperature" and "seg_temperature" in chunk.columns:
                s = s.fillna(pd.to_numeric(chunk["seg_temperature"], errors="coerce"))

            # Fill in order: per-id mean -> global mean -> 0.
            if s.isna().any():
                id_map = ids.map(per_id_mean[c])
                s = s.fillna(pd.to_numeric(id_map, errors="coerce"))
                s = s.fillna(global_mean[c])
                s = s.fillna(0.0)

            chunk[c] = s

        # Last guard for any object NaN.
        obj_cols = chunk.select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            chunk[c] = chunk[c].fillna("")

        # Track remaining nulls.
        nulls = chunk.isna().sum()
        if nulls_after_total is None:
            nulls_after_total = nulls.astype("int64")
        else:
            nulls_after_total = nulls_after_total.add(nulls, fill_value=0).astype("int64")

        chunk.to_csv(args.out_csv, mode="w" if first else "a", index=False, header=first)
        first = False
        rows_written += len(chunk)

    nulls_after_total = nulls_after_total if nulls_after_total is not None else pd.Series(dtype="int64")

    report = {
        "inputs": {"input_csv": str(args.input_csv)},
        "outputs": {"training_no_gaps_csv": str(args.out_csv)},
        "counts": {
            "rows_total_input": int(rows_total),
            "rows_written": int(rows_written),
            "columns_total": int(len(cols)),
            "columns_with_nulls_after": int((nulls_after_total > 0).sum()),
            "total_nulls_after": int(nulls_after_total.sum()),
        },
        "nulls_after_by_column": {k: int(v) for k, v in nulls_after_total.to_dict().items()},
        "fill_policy": "numeric: per-id mean -> global mean -> 0; temperature fallback from seg_temperature; object NaN -> empty string",
    }

    # pH hard filter and repair (interpolate, then means).
    ph_fix = apply_ph_filter_and_repair(
        csv_path=args.out_csv,
        chunksize=args.chunksize,
        ph_max=args.ph_max,
    )
    report["ph_filter_repair"] = ph_fix

    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
