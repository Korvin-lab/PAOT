import pandas as pd
import pysand.erosion as erosion
import math
import sys
import os
import warnings
import contextlib
import io
from tqdm import tqdm

warnings.filterwarnings("ignore")

INPUT_CSV = "merged_VTD_ERR_1250000742.csv"
OUTPUT_CSV = "output_VTD_ERR_1250000742_with_erosion.csv"
SEPARATOR = ","
ENCODING = "utf-8-sig"
CHUNK_SIZE = 100_000

#Колонки
COL_FEATURE_DESC = "Feature Description"
COL_VELOCITY = "seg_v_liq"
COL_Q_LIQ = "Qzh"
COL_Q_GAS = "Qgas"
COL_DIAMETER = "D"
COL_RHO_OIL = "seg_rho_oil"
COL_RHO_WAT = "seg_rho_wat"
COL_RHO_GAS = "gas_density"
COL_WATER_CUT = "water_cut"
COL_MU_M = "viscosity_liquid_work"
COL_SAND_CONC = "kvch"

DIAMETER_SCALE = 0.001   # мм -> м
MU_SCALE = 1.0
GF = 1.0
RHO_P = 2650.0
DP_DEFAULT = 0.3
MATERIAL = "duplex"
CRUSHED = False

OUT_COL = "Ecor_DNV"
OUT_COL_METHOD = "DNV_method"   # pipe / tee для контроля

TEE_KEYWORDS = ["тройник", "отвод"] # синонимы для отвода и тройника,

######################_____________________________________________


@contextlib.contextmanager
def _suppress_output():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_out
        sys.stderr = old_err


def compute_Q_s(kvch_mg_l: float, qzh_m3_day: float) -> float:
    return kvch_mg_l * qzh_m3_day / 86400.0


def compute_mixture(q_liq, q_gas, D_m, rho_oil, rho_wat, rho_gas, water_cut_pct):
    A = math.pi * D_m ** 2 / 4.0
    v_sl = q_liq / (86400.0 * A)
    v_sg = q_gas * 1000.0 / (86400.0 * A)
    v_m = v_sl + v_sg
    if v_m <= 0:
        return float("nan"), float("nan")
    lambda_l = v_sl / v_m
    wc = water_cut_pct / 100.0
    rho_liq = rho_oil * (1.0 - wc) + rho_wat * wc
    rho_m = rho_liq * lambda_l + rho_gas * (1.0 - lambda_l)
    return v_m, rho_m


def is_tee(feature_desc) -> bool:
    if pd.isna(feature_desc):
        return False
    desc = str(feature_desc).strip().lower()
    return any(kw in desc for kw in TEE_KEYWORDS)


def calc_row(row) -> tuple:
    try:
        D    = float(row[COL_DIAMETER]) * DIAMETER_SCALE
        kvch = float(row[COL_SAND_CONC])
        q_liq = float(row[COL_Q_LIQ])

        if D <= 0 or kvch < 0 or q_liq <= 0:
            return float("nan"), None

        Q_s = compute_Q_s(kvch, q_liq)
        if Q_s <= 0:
            return float("nan"), None

        if is_tee(row[COL_FEATURE_DESC]):
            # ── Расчёт TEE ────────────────────────────────────────────────
            q_gas = float(row[COL_Q_GAS])
            rho_oil = float(row[COL_RHO_OIL])
            rho_wat = float(row[COL_RHO_WAT])
            rho_gas = float(row[COL_RHO_GAS])
            water_cut = float(row[COL_WATER_CUT])
            mu_m = float(row[COL_MU_M]) * MU_SCALE

            v_m, rho_m = compute_mixture(q_liq, q_gas, D,
                                         rho_oil, rho_wat, rho_gas, water_cut)
            if math.isnan(v_m) or math.isnan(rho_m):
                return float("nan"), "tee"

            with _suppress_output():
                E_rel = erosion.tee(v_m=v_m, rho_m=rho_m, mu_m=mu_m,
                                     GF=GF, D=D, d_p=DP_DEFAULT,
                                     material=MATERIAL, rho_p=RHO_P,
                                     crushed=CRUSHED)
                E_rate = erosion.erosion_rate(E_rel, Q_s)
            return E_rate, "tee"

        else:
            #_______STRAIGHT PIPE ________________________
            v_m = float(row[COL_VELOCITY])
            if v_m <= 0:
                return float("nan"), "pipe"

            with _suppress_output():
                E_rel = erosion.straight_pipe(v_m, D, crushed=CRUSHED)
                E_rate = erosion.erosion_rate(E_rel, Q_s)
            return E_rate, "pipe"

    except Exception:
        return float("nan"), None


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Файл не найден: {INPUT_CSV}")
        sys.exit(1)

    print(f"Файл: {INPUT_CSV} ({os.path.getsize(INPUT_CSV)/1024**2:.1f} МБ)")

    first_chunk = True
    total_ok, total_bad = 0, 0
    tee_count, pipe_count = 0, 0

    reader = pd.read_csv(INPUT_CSV, sep=SEPARATOR, encoding=ENCODING,
                         chunksize=CHUNK_SIZE, low_memory=False)

    with tqdm(desc="DNV расчёт", unit=" строк", unit_scale=True) as pbar:
        for chunk in reader:
            e_rates, methods = [], []

            for _, row in chunk.iterrows():
                val, method = calc_row(row)
                e_rates.append(val)
                methods.append(method)

            chunk[OUT_COL] = e_rates
            chunk[OUT_COL_METHOD] = methods

            chunk.to_csv(OUTPUT_CSV, sep=SEPARATOR, encoding=ENCODING,
                         index=False, mode="w" if first_chunk else "a",
                         header=first_chunk)
            first_chunk = False

            ok = sum(1 for v in e_rates if v == v)
            total_ok  += ok
            total_bad += len(e_rates) - ok
            tee_count  += sum(1 for m in methods if m == "tee")
            pipe_count += sum(1 for m in methods if m == "pipe")
            pbar.update(len(chunk))

    print(f"\nГотово!")
    print(f"спешно рассчитано: {total_ok:,}")
    print(f"Ошибок (NaN):{total_bad:,}")
    print(f"По tee:    {tee_count:,}(тройник/отвод)")
    print(f"По straight pipe: {pipe_count:,}")
    print(f"Результат: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

