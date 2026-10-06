from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")
DIGITS_RE = re.compile(r"\d+")


def clean_text(v: object) -> str:
    if pd.isna(v):
        return ""
    t = str(v)
    t = NBSP_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_id(v: object) -> str:
    t = clean_text(v)
    if not t:
        return ""
    compact = t.replace(" ", "")
    try:
        f = float(compact)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    m = "".join(DIGITS_RE.findall(t))
    if m:
        return m
    return t.upper()


def normalize_date(v: object) -> str:
    dt = pd.to_datetime(v, errors="coerce")
    if pd.isna(dt):
        return ""
    return pd.Timestamp(dt).normalize().strftime("%Y-%m-%d")


def to_num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).map(clean_text)
    t = t.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def calculate_pco2_and_mol(
    t_c: np.ndarray,
    c_co2_mg_l: np.ndarray,
    nacl_g_kg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same physics block as stage1:
    - pCO2 helper pressure from CO2 + salinity + temperature
    - output co2_mol_l for legacy pCO2 output convention (x100)
    """
    t_k = t_c + 273.15
    valid = (
        np.isfinite(t_c)
        & np.isfinite(c_co2_mg_l)
        & np.isfinite(nacl_g_kg)
        & (t_k > 0)
        & (c_co2_mg_l >= 0)
    )
    out_p = np.full_like(t_c, np.nan, dtype=float)
    out_mol = np.full_like(t_c, np.nan, dtype=float)
    if not np.any(valid):
        return out_p, out_mol

    tv = t_c[valid]
    tkv = t_k[valid]
    cv = c_co2_mg_l[valid]
    sv = nacl_g_kg[valid]

    with np.errstate(all="ignore"):
        log_h = (
            108.3865
            + 0.01985076 * tkv
            - 6919.53 / tkv
            - 40.45154 * np.log10(tkv)
            + 669365 / (tkv**2)
        )
        h = np.power(10.0, log_h)
        denom = 1000.0 - 1.005 * sv
        i_strength = np.where(np.abs(denom) > 1e-12, 19.0 * sv / denom, np.nan)
        h_ion = 0.091 + 0.021 + (-0.005 - 0.00053 * tv)
        h_corr = h / np.power(10.0, i_strength * h_ion)
        h_corr_mpa = h_corr * 10.0

        c_mol_l = cv / (44.1 * 10000.0)
        p_co2_mpa = c_mol_l / h_corr_mpa
        co2_mol_l = h_corr_mpa * p_co2_mpa

    bad = (
        ~np.isfinite(p_co2_mpa)
        | ~np.isfinite(co2_mol_l)
        | ~np.isfinite(h_corr_mpa)
        | (h_corr_mpa <= 0)
    )
    p_co2_mpa[bad] = np.nan
    co2_mol_l[bad] = np.nan

    out_p[valid] = p_co2_mpa
    out_mol[valid] = co2_mol_l
    return out_p, out_mol


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fill CO2/Min/pH chemistry coverage for variant3 outputs: "
        "interpolate inside, mean-fill edges, global curve fallback."
    )
    p.add_argument("--required-csv", type=Path, required=True)
    p.add_argument("--full-csv", type=Path, required=True)
    p.add_argument("--id-date-csv", type=Path, required=True)
    p.add_argument("--chem-daily-csv", type=Path, required=True)
    p.add_argument("--graph-edges-csv", type=Path, default=None, help="Graph edges CSV for CO2 propagation by neighboring pipes.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=250_000)
    return p.parse_args()


def build_raw_co2_lookup(chem_daily_csv: Path) -> pd.DataFrame:
    cols = ["id простого участка", "Дата контроля", "CO2"]
    df = pd.read_csv(chem_daily_csv, usecols=lambda c: c in cols, low_memory=False)
    df["id"] = df["id простого участка"].map(normalize_id)
    df["date"] = df["Дата контроля"].map(normalize_date)
    df["CO2"] = to_num(df["CO2"])
    # CO2 <= 0 for this pipeline is treated as missing source, not as physical zero.
    df["CO2"] = df["CO2"].where(df["CO2"] > 0)
    out = df[["id", "date", "CO2"]].copy().rename(columns={"CO2": "CO2 in Water Phase"})
    out = out[(out["id"] != "") & (out["date"] != "")]
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    return out


def build_global_curve(
    df: pd.DataFrame,
    value_col: str,
    all_dates: pd.Index,
) -> pd.Series:
    # Daily mean across all IDs, then interpolation, then remaining by global mean.
    g = (
        df.groupby("date", as_index=False)[value_col]
        .mean()
        .sort_values("date")
    )
    s = g.set_index("date")[value_col].reindex(all_dates)
    s = s.interpolate(method="linear", limit_area="inside")
    mean_val = s.mean(skipna=True)
    if pd.notna(mean_val):
        s = s.fillna(float(mean_val))
    return s


def fill_per_id_with_global_fallback(
    id_date_df: pd.DataFrame,
    value_col: str,
    global_curve: pd.Series,
    neighbors_by_id: dict[str, list[str]] | None = None,
    max_graph_iters: int = 6,
) -> tuple[pd.Series, dict[str, Any]]:
    out = pd.Series(index=id_date_df.index, dtype=float)
    by_id = id_date_df.groupby("id").groups
    ids_total = len(by_id)
    ids_no_source = 0

    global_mean = float(global_curve.mean(skipna=True)) if global_curve.notna().any() else np.nan

    for pid, idx in by_id.items():
        part = id_date_df.loc[idx, ["date", value_col]].copy().sort_values("date")
        s = to_num(part[value_col])

        if int(s.notna().sum()) == 0:
            ids_no_source += 1
            filled = pd.Series(np.nan, index=part.index, dtype=float)
        else:
            # 1) interpolate only internal gaps
            filled = s.interpolate(method="linear", limit_area="inside")
            # 2) fill left/right by per-id mean over available+interpolated
            mean_id = filled.mean(skipna=True)
            if pd.notna(mean_id):
                filled = filled.fillna(float(mean_id))
        out.loc[part.index] = filled.values

    rows_after_local = int(out.notna().sum())

    # 3) graph propagation by same date from neighboring IDs
    graph_filled_rows = 0
    if neighbors_by_id:
        work = id_date_df[["id", "date"]].copy()
        work["v"] = out.values

        piv = work.pivot(index="date", columns="id", values="v")
        for _ in range(max_graph_iters):
            prev = piv.copy()
            for pid in piv.columns:
                nbrs = [n for n in neighbors_by_id.get(str(pid), []) if n in piv.columns]
                if not nbrs:
                    continue
                nbr_mean = piv[nbrs].mean(axis=1, skipna=True)
                mask = piv[pid].isna() & nbr_mean.notna()
                if mask.any():
                    piv.loc[mask, pid] = nbr_mean.loc[mask]
            if int((prev.isna() & piv.notna()).sum().sum()) == 0:
                break

        piv_s = piv.stack(dropna=False)
        idx = pd.MultiIndex.from_arrays([work["date"].values, work["id"].values])
        after_graph = pd.Series(piv_s.reindex(idx).values, index=id_date_df.index, dtype=float)
        graph_filled_rows = int((out.isna() & after_graph.notna()).sum())
        out = after_graph

    rows_after_graph = int(out.notna().sum())

    # 4) global curve fallback by date
    for pid, idx in by_id.items():
        part = id_date_df.loc[idx, ["date"]].copy().sort_values("date")
        filled = out.loc[part.index]
        if filled.isna().any():
            gc = part["date"].map(global_curve).astype(float)
            filled = filled.fillna(gc)

        # 5) final fallback by global mean (if needed)
        if filled.isna().any() and pd.notna(global_mean):
            filled = filled.fillna(global_mean)

        out.loc[part.index] = filled.values

    rep = {
        "ids_total": ids_total,
        "ids_without_any_source": ids_no_source,
        "rows_before_non_null": int(id_date_df[value_col].notna().sum()),
        "rows_after_local_interp_and_edges": rows_after_local,
        "graph_filled_rows": graph_filled_rows,
        "rows_after_graph_propagation": rows_after_graph,
        "rows_after_non_null": int(out.notna().sum()),
        "rows_total": int(len(id_date_df)),
    }
    return out, rep


def load_graph_neighbors(graph_edges_csv: Path | None) -> dict[str, list[str]]:
    if graph_edges_csv is None:
        return {}
    p = Path(graph_edges_csv)
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=object)

    source_col = "source_id" if "source_id" in df.columns else ("source" if "source" in df.columns else None)
    target_col = "target_id" if "target_id" in df.columns else ("target" if "target" in df.columns else None)
    if source_col is None or target_col is None:
        return {}

    src = df[source_col].map(normalize_id)
    tgt = df[target_col].map(normalize_id)
    m = (src != "") & (tgt != "")
    src = src[m]
    tgt = tgt[m]

    nbr: dict[str, set[str]] = {}
    for s, t in zip(src.tolist(), tgt.tolist()):
        nbr.setdefault(s, set()).add(t)
        nbr.setdefault(t, set()).add(s)
    return {k: sorted(v) for k, v in nbr.items()}


def apply_filled_values_to_large_csv(
    src_csv: Path,
    dst_csv: Path,
    fill_df: pd.DataFrame,
    fill_cols: list[str],
    chunksize: int,
) -> int:
    if dst_csv.exists():
        dst_csv.unlink()

    # Small lookup table by id+date.
    lookup = fill_df[["id", "date"] + fill_cols].copy()

    rows_total = 0
    first = True
    for chunk in pd.read_csv(src_csv, chunksize=chunksize, low_memory=False):
        if "id" not in chunk.columns or "date" not in chunk.columns:
            raise ValueError(f"{src_csv} does not contain id/date columns.")
        chunk["id"] = chunk["id"].map(normalize_id)
        chunk["date"] = chunk["date"].map(normalize_date)
        merged = chunk.merge(lookup, on=["id", "date"], how="left", suffixes=("", "__fill"))

        for c in fill_cols:
            cf = f"{c}__fill"
            if c not in merged.columns:
                merged[c] = np.nan
            merged[c] = merged[cf].where(merged[cf].notna(), merged[c])
            merged.drop(columns=[cf], inplace=True)

        mode = "w" if first else "a"
        merged.to_csv(dst_csv, mode=mode, header=first, index=False)
        first = False
        rows_total += len(merged)
    return rows_total


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_id_date = out_dir / "финальный_датасет__stage1__id_date_chem_co2__filled.csv"
    out_required = out_dir / "финальный_датасет__stage1__required_columns__filled.csv"
    out_full = out_dir / "финальный_датасет__stage1__full__filled.csv"
    out_global = out_dir / "global_approx_curves__co2_min_ph.csv"
    out_report = out_dir / "fill_chem_full_coverage__report.json"

    id_date = pd.read_csv(args.id_date_csv, low_memory=False)
    id_date["id"] = id_date["id"].map(normalize_id)
    id_date["date"] = id_date["date"].map(normalize_date)
    id_date = id_date[(id_date["id"] != "") & (id_date["date"] != "")].copy()
    id_date = id_date.sort_values(["id", "date"]).reset_index(drop=True)
    neighbors_by_id = load_graph_neighbors(args.graph_edges_csv)

    raw_co2 = build_raw_co2_lookup(args.chem_daily_csv)
    id_date = id_date.merge(raw_co2, on=["id", "date"], how="left", suffixes=("", "__raw"))

    # Ensure target chemistry columns exist.
    for c in ["CO2 in Water Phase", "Min", "pH", "pCO2"]:
        if c not in id_date.columns:
            id_date[c] = np.nan
        id_date[c] = to_num(id_date[c])
    # CO2-in-water source (raw), then fill it too.
    raw_col = "CO2 in Water Phase__raw" if "CO2 in Water Phase__raw" in id_date.columns else ("CO2 in Water Phase" if "CO2 in Water Phase" in raw_co2.columns else None)
    if raw_col is not None:
        # If CO2-in-water already exists in id_date, merge creates __raw.
        # If not, merge may keep only source column itself.
        co2_base = to_num(id_date["CO2 in Water Phase"]) if "CO2 in Water Phase" in id_date.columns else pd.Series(np.nan, index=id_date.index)
        co2_raw = to_num(id_date[raw_col]) if raw_col in id_date.columns else pd.Series(np.nan, index=id_date.index)
        id_date["CO2 in Water Phase"] = co2_base.where(co2_base.notna(), co2_raw)
    for c in ["CO2 in Water Phase__raw"]:
        if c in id_date.columns:
            id_date.drop(columns=[c], inplace=True)

    all_dates = pd.Index(sorted(id_date["date"].unique()), name="date")
    fill_targets_base = ["CO2 in Water Phase", "Min", "pH"]

    global_curves = pd.DataFrame({"date": all_dates})
    per_col_report: dict[str, Any] = {}

    for c in fill_targets_base:
        gc = build_global_curve(id_date, c, all_dates)
        global_curves[f"{c}__global_curve"] = gc.values
        if c == "CO2 in Water Phase":
            filled, rep = fill_per_id_with_global_fallback(
                id_date,
                c,
                gc,
                neighbors_by_id=neighbors_by_id,
                max_graph_iters=6,
            )
        else:
            filled, rep = fill_per_id_with_global_fallback(id_date, c, gc)
        id_date[c] = filled.values
        per_col_report[c] = rep

    # Recompute pCO2 from filled CO2 + Min + temperature (same equation as stage1).
    # In stage1 user-facing pCO2 is historical alias for CO2_mol*100.
    t_col = "seg_temperature" if "seg_temperature" in id_date.columns else None
    pco2_calc_rows = 0
    if t_col is not None:
        t_c = to_num(id_date[t_col]).to_numpy(dtype=float)
        c_co2 = to_num(id_date["CO2 in Water Phase"]).to_numpy(dtype=float)
        min_mg_l = to_num(id_date["Min"]).to_numpy(dtype=float)
        nacl_g_kg = min_mg_l / 1000.0
        _p_mpa, co2_mol_l = calculate_pco2_and_mol(t_c=t_c, c_co2_mg_l=c_co2, nacl_g_kg=nacl_g_kg)
        pco2_calc = co2_mol_l * 100.0
        pco2_calc_rows = int(np.isfinite(pco2_calc).sum())
        base_pco2 = to_num(id_date["pCO2"])
        id_date["pCO2"] = base_pco2.where(base_pco2.notna(), pd.Series(pco2_calc, index=id_date.index, dtype=float))

    # Fill pCO2 finally (local interp -> graph/global fallback).
    gc_pco2 = build_global_curve(id_date, "pCO2", all_dates)
    global_curves["pCO2__global_curve"] = gc_pco2.values
    filled_pco2, rep_pco2 = fill_per_id_with_global_fallback(id_date, "pCO2", gc_pco2)
    id_date["pCO2"] = filled_pco2.values
    rep_pco2["rows_recomputed_from_co2_min_t"] = int(pco2_calc_rows)
    per_col_report["pCO2"] = rep_pco2

    # Hard guarantee: no gaps in fill targets.
    for c in ["CO2 in Water Phase", "Min", "pH", "pCO2"]:
        if id_date[c].isna().any():
            fallback = float(id_date[c].mean(skipna=True)) if id_date[c].notna().any() else 0.0
            id_date[c] = id_date[c].fillna(fallback)

    id_date.to_csv(out_id_date, index=False)
    global_curves.to_csv(out_global, index=False)

    rows_required = apply_filled_values_to_large_csv(
        src_csv=args.required_csv,
        dst_csv=out_required,
        fill_df=id_date,
        fill_cols=["CO2 in Water Phase", "Min", "pH", "pCO2"],
        chunksize=args.chunksize,
    )
    rows_full = apply_filled_values_to_large_csv(
        src_csv=args.full_csv,
        dst_csv=out_full,
        fill_df=id_date,
        fill_cols=["CO2 in Water Phase", "Min", "pH", "pCO2"],
        chunksize=args.chunksize,
    )

    rep = {
        "inputs": {
            "required_csv": str(args.required_csv),
            "full_csv": str(args.full_csv),
            "id_date_csv": str(args.id_date_csv),
            "chem_daily_csv": str(args.chem_daily_csv),
            "graph_edges_csv": str(args.graph_edges_csv) if args.graph_edges_csv else None,
        },
        "outputs": {
            "id_date_filled_csv": str(out_id_date),
            "required_filled_csv": str(out_required),
            "full_filled_csv": str(out_full),
            "global_curves_csv": str(out_global),
        },
        "counts": {
            "id_date_rows": int(len(id_date)),
            "required_rows_written": int(rows_required),
            "full_rows_written": int(rows_full),
        },
        "per_column": per_col_report,
    }
    out_report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
