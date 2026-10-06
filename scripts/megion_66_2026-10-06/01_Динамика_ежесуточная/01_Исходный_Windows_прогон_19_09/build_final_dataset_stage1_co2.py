from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

VARIANT_DIR = SCRIPT_DIR / "output_orenburg" / "variant_strict_orenburg"
SEGMENTS_CSV = VARIANT_DIR / "расчет_сегментов__все_трубы.csv"
CHEM_DAILY_CSV = SCRIPT_DIR / "input" / "chem_daily_orenburg.csv"
KVCH_DAILY_CSV = SCRIPT_DIR / "input" / "kvch_daily_orenburg.csv"
VISC_DAILY_CSV = SCRIPT_DIR / "input" / "visc_daily_orenburg.csv"
REQUESTED_JSON = SCRIPT_DIR / "input" / "graph_orenburg_requested_daily_params.json"
OUT_DIR_DEFAULT = SCRIPT_DIR / "08_stage1_orenburg"

NBSP_RE = re.compile(r"[\u00A0\u2007\u202F]")
DIGITS_RE = re.compile(r"\d+")
PSI_PER_PA = 0.00014503773773020923

ID_COL = "id"
DATE_COL = "date"


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
    # ISO must never pass through a day-first parser (YYYY-MM-DD is unambiguous).
    if isinstance(v, str):
        text = v.strip()
        if re.match(r'^\d{4}-\d{2}-\d{2}(?:$|[ T])', text):
            try:
                return datetime.fromisoformat(text.replace('Z', '+00:00')).date().isoformat()
            except ValueError:
                return ''
    dt = pd.to_datetime(v, errors="coerce", dayfirst=True)
    if pd.isna(dt):
        return ""
    return pd.Timestamp(dt).normalize().strftime("%Y-%m-%d")


def to_num(s: pd.Series) -> pd.Series:
    if s.dtype.kind in "biufc":
        return pd.to_numeric(s, errors="coerce")
    t = s.astype(str).map(clean_text)
    t = t.str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(t, errors="coerce")


def convert_mineralization_to_nacl_g_kg(mineralization_mg_l: np.ndarray) -> np.ndarray:
    return mineralization_mg_l / 1000.0


def calculate_pco2_and_mol(
    t_c: np.ndarray,
    c_co2_mg_l: np.ndarray,
    nacl_g_kg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    t_k = t_c + 273.15

    valid = (
        np.isfinite(t_c)
        & np.isfinite(c_co2_mg_l)
        & np.isfinite(nacl_g_kg)
        & (t_k > 0)
        & (c_co2_mg_l > 0)
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


def calculate_comprw_kriel(
    t_c: np.ndarray,
    p_mpa: np.ndarray,
    salinity_ppm: np.ndarray,
) -> np.ndarray:
    """
    Water compressibility by Kriel correlation (aligned with Unifloc implementation).
    Input:
      - t_c in Celsius
      - p_mpa in MPa
      - salinity_ppm in ppm
    Output:
      - comprw in 1/Pa
    """
    out = np.full_like(t_c, np.nan, dtype=float)
    valid = (
        np.isfinite(t_c)
        & np.isfinite(p_mpa)
        & np.isfinite(salinity_ppm)
        & (p_mpa > 0)
    )
    if not np.any(valid):
        return out
    tv = t_c[valid]
    pv_pa = p_mpa[valid] * 1e6
    sv = salinity_ppm[valid]
    with np.errstate(all="ignore"):
        t_f = tv * 1.8 + 32.0
        p_psi = pv_pa * PSI_PER_PA
        denom = 7.033 * p_psi + 0.5415 * sv - 537.0 * t_f + 403300.0
        cw = 0.1 * 145.04 / denom
    bad = (~np.isfinite(cw)) | (denom <= 0) | (cw <= 0)
    cw[bad] = np.nan
    out[valid] = cw
    return out


def load_chem_lookup(chem_daily_csv: Path) -> pd.DataFrame:
    usecols = [
        "id простого участка",
        "Дата контроля",
        "CO2",
        "Общая минерализация",
        "Общая минерализация, г/л",
        "pH",
    ]
    df = pd.read_csv(chem_daily_csv, usecols=lambda c: c in usecols, low_memory=False)
    df["id"] = df["id простого участка"].map(normalize_id)
    df["date"] = df["Дата контроля"].map(normalize_date)
    df["CO2"] = to_num(df["CO2"])
    df["Min_mg_l"] = to_num(df["Общая минерализация"])
    df["Min"] = to_num(df["Общая минерализация, г/л"])
    missing_min = df["Min"].isna() & df["Min_mg_l"].notna()
    df.loc[missing_min, "Min"] = df.loc[missing_min, "Min_mg_l"] / 1000.0
    df["pH"] = to_num(df["pH"])
    out = df[["id", "date", "CO2", "Min", "Min_mg_l", "pH"]].copy()
    out = out[(out["id"] != "") & (out["date"] != "")]
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    return out


def load_kvch_lookup(kvch_daily_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(kvch_daily_csv, usecols=["id", "date", "kvch_mg_l"], low_memory=False)
    df["id"] = df["id"].map(normalize_id)
    df["date"] = df["date"].map(normalize_date)
    df["kvch"] = to_num(df["kvch_mg_l"])
    out = df[["id", "date", "kvch"]].copy()
    out = out[(out["id"] != "") & (out["date"] != "")]
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    return out


def load_requested_lookup(requested_json: Path) -> pd.DataFrame:
    obj = json.loads(requested_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for pid, payload in obj.get("by_id", {}).items():
        pid_norm = normalize_id(pid)
        for r in payload.get("daily", []):
            dt = normalize_date(r.get("Дата") or r.get("date"))
            if not dt:
                continue
            rows.append(
                {
                    "id": pid_norm,
                    "date": dt,
                    "viscosity_liquid_work": r.get("Жидкости, кг/(м*с)"),
                    "oil_density_requested": r.get("Нефти, кг/м3"),
                    "water_density_requested": r.get("Жидкости, кг/м3"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["id", "date", "viscosity_liquid_work", "oil_density_requested", "water_density_requested"])
    df = pd.DataFrame(rows).sort_values(["id", "date"])
    for column in ('viscosity_liquid_work','oil_density_requested','water_density_requested'):
        df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.drop_duplicates(["id", "date"], keep="last")
    return df


def load_visc_lookup(visc_csv: Path) -> pd.DataFrame:
    if not visc_csv.exists():
        return pd.DataFrame(columns=["id", "date", "viscosity_liquid_work_shtr"])
    df = pd.read_csv(visc_csv, low_memory=False)
    if "id" not in df.columns or "date" not in df.columns:
        return pd.DataFrame(columns=["id", "date", "viscosity_liquid_work_shtr"])
    src_col = "mu_liq" if "mu_liq" in df.columns else ("viscosity_liquid_work" if "viscosity_liquid_work" in df.columns else None)
    if src_col is None:
        return pd.DataFrame(columns=["id", "date", "viscosity_liquid_work_shtr"])
    df["id"] = df["id"].map(normalize_id)
    df["date"] = df["date"].map(normalize_date)
    df["viscosity_liquid_work_shtr"] = to_num(df[src_col])
    out = df[["id", "date", "viscosity_liquid_work_shtr"]].copy()
    out = out[(out["id"] != "") & (out["date"] != "")]
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"], keep="last")
    return out


def load_dosage_lookup(dosage_files: list[Path]) -> pd.DataFrame:
    if not dosage_files:
        return pd.DataFrame(columns=["id", "date", "ing_factor"])

    all_rows: list[pd.DataFrame] = []
    for p in dosage_files:
        p = Path(p)
        if not p.exists():
            continue
        try:
            xls = pd.ExcelFile(p)
        except Exception:
            continue
        for sh in xls.sheet_names:
            try:
                df = pd.read_excel(p, sheet_name=sh, dtype=object)
            except Exception:
                continue

            required = [
                "ID простого участка",
                "Дата расчета",
                "Дозировка сегмента факт г/м3",
                "Дозировка регламент г/м3",
            ]
            if any(c not in df.columns for c in required):
                continue

            part = df[required].copy()
            part["id"] = part["ID простого участка"].map(normalize_id)
            part["date"] = part["Дата расчета"].map(normalize_date)
            part["dose_fact"] = to_num(part["Дозировка сегмента факт г/м3"])
            part["dose_reg"] = to_num(part["Дозировка регламент г/м3"])
            part = part[(part["id"] != "") & (part["date"] != "")]
            all_rows.append(part[["id", "date", "dose_fact", "dose_reg"]])

    if not all_rows:
        return pd.DataFrame(columns=["id", "date", "ing_factor"])

    out = pd.concat(all_rows, ignore_index=True)
    out = (
        out.groupby(["id", "date"], as_index=False)[["dose_fact", "dose_reg"]]
        .mean()
        .sort_values(["id", "date"])
    )
    out["ing_factor"] = np.nan
    # Formula (latest agreed): ing_factor = factual / regulatory
    ok = out["dose_reg"].notna() & (out["dose_reg"] > 0) & out["dose_fact"].notna()
    out.loc[ok, "ing_factor"] = out.loc[ok, "dose_fact"] / out.loc[ok, "dose_reg"]
    return out[["id", "date", "ing_factor"]]


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final stage1 dataset from strict segments + chemistry.")
    p.add_argument("--segments-csv", type=Path, default=SEGMENTS_CSV)
    p.add_argument("--chem-daily-csv", type=Path, default=CHEM_DAILY_CSV)
    p.add_argument("--kvch-daily-csv", type=Path, default=KVCH_DAILY_CSV)
    p.add_argument("--visc-csv", type=Path, default=VISC_DAILY_CSV)
    p.add_argument("--requested-json", type=Path, default=REQUESTED_JSON)
    p.add_argument(
        "--dosage-xls",
        type=Path,
        action="append",
        default=[],
        help="Dosage XLS with columns: ID простого участка, Дата расчета, Дозировка сегмента факт г/м3, Дозировка регламент г/м3. May be specified multiple times.",
    )
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = args.out_dir
    out_full = out_dir / "финальный_датасет__stage1__full.csv"
    out_required = out_dir / "финальный_датасет__stage1__required_columns.csv"
    out_id_date = out_dir / "финальный_датасет__stage1__id_date_chem_co2.csv"
    out_id_date_tmp = out_dir / "_tmp_id_date_chunks.csv"
    out_report = out_dir / "финальный_датасет__stage1__report.json"

    out_dir.mkdir(parents=True, exist_ok=True)

    chem = load_chem_lookup(args.chem_daily_csv)
    kvch = load_kvch_lookup(args.kvch_daily_csv)
    visc = load_visc_lookup(args.visc_csv)
    req = load_requested_lookup(args.requested_json)
    dosage = load_dosage_lookup(list(args.dosage_xls))

    merge_chem = chem.rename(columns={"CO2": "co2_src", "Min": "min_g_l_src", "Min_mg_l": "min_mg_l_src", "pH": "ph_src"})

    keep_cols = [
        "id",
        "date",
        "distance",
        "angle_deg",
        "D",
        "rho_wat",
        "q_liq",
        "watercut",
        "q_oil",
        "rho_gas",
        "q_gas",
        "v_liquid_true",
        "rho_oil",
        "v_mix",
        "t",
        "p",
        "bw",
        "muw",
        "hc_wat",
        "hc_gas",
        "rs",
        "compro",
        "bo",
        "muo",
        "hc_oil",
        "st_wat_gas",
        "st_oil_gas",
        "mu_mix",
        "salinity",
        "comprw",
        "q_gas_norm_tsd_m3d",
        "q_gas_work_m3s_polytech",
        "q_gas_work_tsd_m3d_polytech",
        "q_liq_m3s_polytech",
        "q_mix_m3s_polytech",
        "v_mix_polytech",
        "v_mix_source_techregime",
        "alpha_liq_polytech",
        "alpha_gas_polytech",
        "rho_mix_polytech",
        "mu_mix_polytech",
        "re_polytech",
        "ff_polytech",
        "knns_polytech",
    ]

    required_order = [
        "date",
        "id",
        "Qv",
        "Qzh",
        "Qn",
        "Qgas",
        "water_cut",
        "kvch",
        "CO2 in Water Phase",
        "Min",
        "pH",
        "ing_factor",
        "viscosity_liquid_work",
        "oil_viscosity",
        "oil_density",
        "water_density",
        "temperature",
        "gas_density",
        "segment_id",
        "seg_p_start",
        "seg_v_liq",
        "seg_temperature",
        "seg_bw",
        "seg_rho_wat",
        "seg_muw",
        "seg_hc_wat",
        "seg_salinity",
        "seg_hc_gas",
        "seg_rs",
        "seg_compro",
        "seg_bo",
        "seg_rho_oil",
        "seg_muo",
        "seg_hc_oil",
        "seg_st_wat_gas",
        "seg_st_oil_gas",
        "seg_comprw",
        "seg_reynolds",
        "pCO2",
        "Qgas_norm_tsd_m3d",
        "Qgas_work_m3s_polytech",
        "Qliq_m3s_polytech",
        "Qmix_m3s_polytech",
        "seg_v_mix_polytech",
        "seg_v_mix_source_techregime",
        "seg_alpha_gas_polytech",
        "seg_alpha_liq_polytech",
        "seg_reynolds_polytech",
        "seg_knns_polytech",
    ]

    if out_full.exists():
        out_full.unlink()
    if out_required.exists():
        out_required.unlink()
    if out_id_date.exists():
        out_id_date.unlink()
    if out_id_date_tmp.exists():
        out_id_date_tmp.unlink()

    chunksize = 250_000
    chunk_n = 0
    row_total = 0
    co2_calc_rows = 0
    invalid_requested_viscosity_rows = 0
    invalid_seg_muw_rows = 0
    first_written = False
    first_written_id_date = False

    for chunk in pd.read_csv(args.segments_csv, usecols=lambda c: c in keep_cols, chunksize=chunksize, low_memory=False):
        chunk_n += 1
        row_total += len(chunk)
        chunk = ensure_cols(chunk, keep_cols)

        chunk["id"] = chunk["id"].map(normalize_id)
        chunk["date"] = chunk["date"].map(normalize_date)

        for c in [
            "distance",
            "angle_deg",
            "D",
            "rho_wat",
            "q_liq",
            "watercut",
            "q_oil",
            "rho_gas",
            "q_gas",
            "v_liquid_true",
            "rho_oil",
            "v_mix",
            "t",
            "p",
            "bw",
            "muw",
            "hc_wat",
            "hc_gas",
            "rs",
            "compro",
            "bo",
            "muo",
            "hc_oil",
            "st_wat_gas",
            "st_oil_gas",
            "mu_mix",
            "salinity",
            "comprw",
            "q_gas_norm_tsd_m3d",
            "q_gas_work_m3s_polytech",
            "q_gas_work_tsd_m3d_polytech",
            "q_liq_m3s_polytech",
            "q_mix_m3s_polytech",
            "v_mix_polytech",
            "v_mix_source_techregime",
            "alpha_liq_polytech",
            "alpha_gas_polytech",
            "rho_mix_polytech",
            "mu_mix_polytech",
            "re_polytech",
            "ff_polytech",
            "knns_polytech",
        ]:
            chunk[c] = to_num(chunk[c])

        chunk = chunk.merge(merge_chem, on=["id", "date"], how="left")
        chunk = chunk.merge(kvch, on=["id", "date"], how="left")
        chunk = chunk.merge(visc, on=["id", "date"], how="left")
        chunk = chunk.merge(req, on=["id", "date"], how="left")
        chunk = chunk.merge(dosage, on=["id", "date"], how="left")

        chunk["Qv"] = chunk["q_liq"] * chunk["watercut"] / 100.0
        chunk["Qzh"] = chunk["q_liq"]
        chunk["Qn"] = chunk["q_oil"]
        chunk["Qgas"] = to_num(chunk["q_gas_work_tsd_m3d_polytech"]).where(
            to_num(chunk["q_gas_work_tsd_m3d_polytech"]).notna(),
            chunk["q_gas"],
        )
        chunk["Qgas_norm_tsd_m3d"] = chunk["q_gas_norm_tsd_m3d"]
        chunk["Qgas_work_m3s_polytech"] = chunk["q_gas_work_m3s_polytech"]
        chunk["Qliq_m3s_polytech"] = chunk["q_liq_m3s_polytech"]
        chunk["Qmix_m3s_polytech"] = chunk["q_mix_m3s_polytech"]
        chunk["water_cut"] = chunk["watercut"]

        chunk["Min"] = chunk["min_g_l_src"]
        chunk["pH"] = chunk["ph_src"]
        chunk["CO2 in Water Phase"] = to_num(chunk["co2_src"]).where(to_num(chunk["co2_src"]) > 0)

        chunk["kvch"] = to_num(chunk["kvch"])
        chunk["seg_p_start"] = chunk["p"] * 10.0  # МПа -> атм (по требованию).
        chunk["z"] = chunk["angle_deg"]

        # Reynolds by user formula:
        # re = (rho_oil + rho_wat) * v_mix * D / mul
        # where mul is taken from PVT mixture viscosity (mu_mix).
        d_m = to_num(chunk["D"])
        mul = to_num(chunk["mu_mix"])
        num_re = (to_num(chunk["rho_oil"]) + to_num(chunk["rho_wat"])) * to_num(chunk["v_mix"]) * d_m
        chunk["seg_reynolds"] = np.nan
        denom_ok = mul.notna() & (mul > 0)
        chunk.loc[denom_ok, "seg_reynolds"] = num_re[denom_ok] / mul[denom_ok]
        chunk["seg_reynolds"] = to_num(chunk["re_polytech"]).where(
            to_num(chunk["re_polytech"]).notna(),
            chunk["seg_reynolds"],
        )

        chunk["D"] = d_m * 1000.0  # м -> мм

        # CO2-derived
        nacl_g_kg = to_num(chunk["min_mg_l_src"]) / 1000.0
        p_co2_mpa, co2_mol_l = calculate_pco2_and_mol(
            t_c=to_num(chunk["t"]).to_numpy(dtype=float),
            c_co2_mg_l=to_num(chunk["co2_src"]).to_numpy(dtype=float),
            nacl_g_kg=nacl_g_kg.to_numpy(dtype=float),
        )
        chunk["P_CO2"] = p_co2_mpa
        chunk["CO2_mol"] = co2_mol_l * 100.0  # По требованию: умножаем на 100.
        chunk["pCO2"] = chunk["CO2_mol"]
        co2_calc_rows += int(np.isfinite(p_co2_mpa).sum())

        # Rename to final semantic columns
        chunk["seg_v_liq"] = to_num(chunk["v_mix_polytech"]).where(
            to_num(chunk["v_mix_polytech"]).notna(),
            chunk["v_liquid_true"],
        )
        chunk["seg_v_mix_polytech"] = chunk["v_mix_polytech"]
        chunk["seg_v_mix_source_techregime"] = chunk["v_mix_source_techregime"]
        chunk["seg_alpha_gas_polytech"] = chunk["alpha_gas_polytech"]
        chunk["seg_alpha_liq_polytech"] = chunk["alpha_liq_polytech"]
        chunk["seg_reynolds_polytech"] = chunk["re_polytech"]
        chunk["seg_knns_polytech"] = chunk["knns_polytech"]
        chunk["seg_temperature"] = chunk["t"]
        chunk["seg_bw"] = chunk["bw"]
        chunk["seg_rho_wat"] = chunk["rho_wat"]
        seg_muw = to_num(chunk["muw"])
        invalid_seg_muw_rows += int((seg_muw.notna() & (seg_muw <= 0)).sum())
        chunk["seg_muw"] = seg_muw.where(seg_muw > 0)
        chunk["seg_hc_wat"] = chunk["hc_wat"]
        chunk["seg_hc_gas"] = chunk["hc_gas"]
        chunk["seg_rs"] = chunk["rs"]
        chunk["seg_compro"] = chunk["compro"]
        chunk["seg_bo"] = chunk["bo"]
        chunk["seg_rho_oil"] = chunk["rho_oil"]
        chunk["seg_muo"] = chunk["muo"]
        chunk["seg_hc_oil"] = chunk["hc_oil"]
        chunk["seg_st_wat_gas"] = chunk["st_wat_gas"]
        chunk["seg_st_oil_gas"] = chunk["st_oil_gas"]

        # User-facing aliases
        chunk["segment_id"] = chunk["distance"]
        chunk["oil_density"] = chunk["seg_rho_oil"]
        chunk["oil_viscosity"] = chunk["seg_muo"]
        chunk["water_density"] = chunk["rho_wat"]
        chunk["gas_density"] = chunk["rho_gas"]

        # viscosity_liquid_work: из requested JSON, если есть; иначе пока NaN.
        requested_visc = to_num(chunk["viscosity_liquid_work"])
        invalid_requested_viscosity_rows += int((requested_visc.notna() & (requested_visc <= 0)).sum())
        fallback_visc = to_num(chunk["viscosity_liquid_work_shtr"])
        fallback_visc = fallback_visc.where(fallback_visc > 0)
        chunk["viscosity_liquid_work"] = requested_visc.where(requested_visc > 0, fallback_visc)

        # ing_factor from dosage sources (no placeholder column ing_sum).
        chunk["ing_factor"] = to_num(chunk["ing_factor"])
        chunk["temperature"] = np.nan
        salinity_src = to_num(chunk["salinity"])
        min_src_ppm = to_num(chunk["min_mg_l_src"])
        chunk["seg_salinity"] = salinity_src.where(salinity_src.notna(), min_src_ppm)

        comprw_src = to_num(chunk["comprw"])
        missing_comprw = comprw_src.isna()
        if missing_comprw.any():
            cw_calc = calculate_comprw_kriel(
                t_c=to_num(chunk["t"]).to_numpy(dtype=float),
                p_mpa=to_num(chunk["p"]).to_numpy(dtype=float),
                salinity_ppm=to_num(chunk["seg_salinity"]).to_numpy(dtype=float),
            )
            cw_calc_s = pd.Series(cw_calc, index=chunk.index, dtype=float)
            comprw_src = comprw_src.where(comprw_src.notna(), cw_calc_s)
        chunk["seg_comprw"] = comprw_src
        # seg_reynolds already calculated above.

        full_keep = required_order + ["z", "D", "P_CO2", "CO2_mol"]
        chunk_full = chunk[full_keep].copy()
        chunk_req = chunk[required_order].copy()

        id_date_cols = [
            "id",
            "date",
            "Qzh",
            "water_cut",
            "Qn",
            "Qgas",
            "kvch",
            "Min",
            "pH",
            "CO2 in Water Phase",
            "pCO2",
            "P_CO2",
            "CO2_mol",
            "seg_temperature",
            "seg_p_start",
            "z",
        ]
        chunk_id_date = chunk_full[id_date_cols].drop_duplicates(["id", "date"], keep="first")

        mode = "w" if not first_written else "a"
        header = not first_written
        chunk_full.to_csv(out_full, mode=mode, header=header, index=False)
        chunk_req.to_csv(out_required, mode=mode, header=header, index=False)
        first_written = True
        mode_id = "w" if not first_written_id_date else "a"
        header_id = not first_written_id_date
        chunk_id_date.to_csv(out_id_date_tmp, mode=mode_id, header=header_id, index=False)
        first_written_id_date = True

        print(f"chunk={chunk_n} rows={len(chunk)} total={row_total}", flush=True)

    # Separate id+date table after CO2/min/pH/kvch binding + CO2 formulas.
    id_date = pd.read_csv(out_id_date_tmp, low_memory=False)
    id_date = (
        id_date.sort_values(["id", "date"])
        .drop_duplicates(["id", "date"], keep="first")
        .reset_index(drop=True)
    )
    id_date.to_csv(out_id_date, index=False)
    out_id_date_tmp.unlink(missing_ok=True)

    rep = {
        "inputs": {
            "segments_csv": str(args.segments_csv),
            "chem_daily_csv": str(args.chem_daily_csv),
            "kvch_daily_csv": str(args.kvch_daily_csv),
            "visc_csv": str(args.visc_csv),
            "requested_json": str(args.requested_json),
            "dosage_xls": [str(x) for x in args.dosage_xls],
        },
        "outputs": {
            "full_csv": str(out_full),
            "required_csv": str(out_required),
            "id_date_csv": str(out_id_date),
        },
        "counts": {
            "segment_rows_processed": int(row_total),
            "co2_formula_rows_non_null": int(co2_calc_rows),
            "id_date_rows": int(len(id_date)),
            "id_count_in_id_date": int(id_date["id"].astype(str).nunique()),
            "invalid_requested_viscosity_rows_rejected": int(invalid_requested_viscosity_rows),
            "invalid_seg_muw_rows_set_nan_before_no_gaps": int(invalid_seg_muw_rows),
        },
    }
    out_report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Done")
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
