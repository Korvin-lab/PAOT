import glob
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution


# ==========================
# DeWaard hard (мм/сут)
# ==========================
def dewaard_hard_mm_day(t_c, p_mpa, co2_frac, v_liquid, d_mm, params):
    scale_factor, logk0, Ea_react, n_react, Sh_coef = params

    A_Henry, B_Henry = 1088.76, 5.113
    a_fug, b_fug = 0.0031, 1.4

    T = max(t_c + 273.15, 1.0)
    d = max(d_mm / 1000.0, 1e-6)

    p_CO2 = max(p_mpa * co2_frac, 0.0)
    p_CO2_atm = p_CO2 * 9.86923

    H = 10 ** (A_Henry / T - B_Henry)
    C_H2CO3 = H * p_CO2_atm

    p_bar = p_mpa * 10.0
    phi = 10 ** (p_bar * (a_fug - b_fug / T))
    f_CO2 = max(phi * p_CO2_atm, 1e-12)

    V_react = 10 ** (logk0 - (Ea_react / T) - 1.3 + n_react * np.log10(f_CO2))

    nu = 1e-6 * np.exp(-0.024 * (t_c - 20))
    D = 2e-9 * np.exp(0.03 * (t_c - 20))

    V_mass_raw = (
        Sh_coef
        * (D ** 0.7)
        / (nu ** 0.5)
        * (max(v_liquid, 0.0) ** 0.8)
        / (d ** 0.2)
        * C_H2CO3
    )
    V_mass = scale_factor * V_mass_raw

    if V_mass > 0 and V_react > 0:
        return float(1.0 / (1.0 / V_mass + 1.0 / V_react))
    return float(max(V_mass, V_react))





# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================
def load_all_csv(folder):
    files = sorted(glob.glob(str(Path(folder) / "*.csv")))
    parts = [pd.read_csv(fp, encoding=ENC,index_col=False) for fp in files]
    return pd.concat(parts, ignore_index=True)


def compute_co2_frac(df):
    if PCO2_COL in df.columns:
        frac = df[PCO2_COL] / df[P_COL]
        return frac.clip(0, 1)

    if CO2_FRAC_COL in df.columns:
        return df[CO2_FRAC_COL].clip(0, 1)

    raise ValueError("Нет данных CO2")


def compute_vcorr_timeseries(df_seg, params):
    co2 = compute_co2_frac(df_seg)
    #print(df_seg.columns)
    d = df_seg.iloc[0]['D'] * 1000
    dmm = (
        pd.to_numeric(df_seg["d_mm"], errors="coerce").fillna(d)
        if "d_mm" in df_seg.columns
        else pd.Series(d, index=df_seg.index)
    )

    t = pd.to_numeric(df_seg[T_COL], errors="coerce")
    p = pd.to_numeric(df_seg[P_COL], errors="coerce")
    v = pd.to_numeric(df_seg[V_COL], errors="coerce")

    vcorr = pd.Series(index=df_seg.index, dtype=float)

    for i in df_seg.index:
        if any(pd.isna(x) for x in (t[i], p[i], v[i], co2[i], dmm[i])):
            vcorr[i] = np.nan
        else:
            vcorr[i] = dewaard_hard_mm_day(
                float(t[i]),
                float(p[i]),
                float(co2[i]),
                float(v[i]),
                float(dmm[i]),
                params,
            )
    return vcorr

def compute_thickness_timeseries(df_seg, vcorr, start_date, nominal_thk):
    """
    Реализация вашей логики:
    1. Рассчитываем утонение ДО первых данных по средней скорости
    2. Используем его как стартовую толщину для периода с данными
    3. Для дат с данными считаем кумулятивно
    """
    # Сортируем по дате
    df = df_seg[[DATE_COL]].copy()
    df["vcorr"] = vcorr
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if df.empty:
        return pd.Series([], dtype=float)
    
    # Первая дата с данными
    first_data_date = df[DATE_COL].iloc[0]
    start_date_ts = pd.Timestamp(start_date)
    
    # Дней до первых данных
    days_before_first = (first_data_date - start_date_ts).days
    days_before_first = max(days_before_first, 0)
    
    # Средняя скорость коррозии из всех данных
    mean_vcorr = df["vcorr"].mean()
    if pd.isna(mean_vcorr):
        mean_vcorr = 0.0
    
    # Утонение ДО первых данных
    loss_before_first = mean_vcorr * days_before_first
    
    # Толщина на первую дату с данными
    thickness_at_first_data = nominal_thk - loss_before_first
    
    # Кумулятивная сумма скоростей для дат с данными
    df["vcorr_filled"] = df["vcorr"].fillna(0)
    cumulative_loss_after_first = df["vcorr_filled"].cumsum()
    
    # Итоговая толщина для каждой даты
    df["thickness"] = thickness_at_first_data - cumulative_loss_after_first
    
    # Возвращаем в исходном порядке
    return df.set_index(df_seg.index)["thickness"]

# ==========================
# MAIN
# ==========================

# ==========================
# НАСТРОЙКИ
# ==========================
#DATA_DIR = r"E:\tmp code\testing_NN\data\done_data\de_waard\data\58507"
#OUT_CSV = r".\testing_NN\data\done_data\de_waard\output_final\debug_full_pipeline_58507_new2.csv"
ENC = "Windows-1251"

#TARGET_ID_X = 58507
#TARGET_SEG_END_M = 0

#START_DATE = "2017-06-15"
NOMINAL_THICKNESS_MM = 8.0

DATE_COL = "date"
ID_COL = "id"
SEG_COL = "distance"
THICK_COL = "thickness_min"

T_COL = "t"
P_COL = "p_start"
V_COL = "v_liquid_true"
PCO2_COL = "P_CO2_MPa"
CO2_FRAC_COL = "CO2"

#DMM_FALLBACK = 219.0

DEFAULT_PARAMS = [2.62e6, 5.8, 1543, 0.67, 0.023]
VAR_PCT = 0.40
EARLY_STOP_REL_ERR = 0.02
PIPES = {47176:"2008-01-01"}
# PIPES = {50768:"2014-10-22",53896:"2014-10-22"
#         ,45781:"2011-10-01",47176:"2008-01-01",50975:"2014-08-06"
#         ,51724:"2015-05-22",51747:"2015-05-27",51748:"2015-05-27"
#         ,51749:"2015-05-27"}
# PIPES = {51747:"2015-05-27",53896:"2014-10-22",51748:"2015-05-27"
#         ,51749:"2015-05-27"}

def main():
    for pipe_id,pipe_data in PIPES.items():
        params_rows = []
        df =pd.read_csv(f"E:\\tmp code\\testing_NN\\data\\done_data\\de_waard\\Part data\\final_tech_mod_{pipe_id}.csv",index_col=False) #load_all_csv()
        df['Min'] = df['Min']/1000
        print(len(df[df['thickness_min']!=None]))
        print(len(df[df['thickness_min'].isna()]))
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        segments = (
            df[df[ID_COL] == pipe_id][SEG_COL]
            .dropna()
            .unique()
        )
        segments = np.sort(segments)
        print(f"[INFO] Found {len(segments)} segments")

        all_results = []
        prev_best_params = None

        for seg_idx in segments:
            print(f"\n[SEGMENT] Processing segment {seg_idx}")

            df_seg = df[
                (df[ID_COL] == pipe_id)
                & (df[SEG_COL].round() == seg_idx)
            ].dropna(subset=[DATE_COL]).sort_values(DATE_COL)

            if df_seg.empty:
                print("  -> empty, skip")
                continue

            df_meas = df_seg.dropna(subset=[THICK_COL])
            if df_meas.empty:
                print("  -> no VTD, skip")
                continue

            meas_row = df_meas.iloc[0]
            #meas_date = meas_row[DATE_COL]
            thick_obs = meas_row[THICK_COL]


            start_date = pd.Timestamp(pipe_data)

            def objective(x):
                """
                Оптимизация: подбираем параметры чтобы толщина на дату замера совпала
                """
                params = list(x)
                vc = compute_vcorr_timeseries(df_seg, params)
                th = compute_thickness_timeseries(df_seg, vc, start_date, NOMINAL_THICKNESS_MM)
                
                # Находим дату замера толщины
                meas_date = meas_row[DATE_COL]
                
                # Ищем толщину на эту дату
                # (предполагаем, что meas_date есть в df_seg)
                mask = df_seg[DATE_COL] == meas_date
                if mask.any():
                    pred = th[mask].iloc[0]
                else:
                    # Если даты нет в данных, используем интерполяцию
                    # или находим ближайшую дату
                    return 1e9
                
                if not np.isfinite(pred):
                    return 1e9
                
                # Абсолютная ошибка
                abs_error = abs(pred - thick_obs)
                
                # Относительная ошибка (%)
                rel_error = abs_error / thick_obs if thick_obs != 0 else abs_error
                
                # Если достигли нужной точности - возвращаем 0
                if rel_error < EARLY_STOP_REL_ERR:
                    return 0.0  # Или очень маленькое значение
                
                return (pred - thick_obs) ** 2  # MSE
            
            if prev_best_params is None:
                start_params = DEFAULT_PARAMS
                print("  -> using DEFAULT_PARAMS")
            else:
                start_params = prev_best_params
                print("  -> using params from previous segment")

            bounds = [
            (p * (1 - VAR_PCT), p * (1 + VAR_PCT))
            for p in start_params]

            result = differential_evolution(
                objective,
                bounds=bounds,
                popsize=10,
                maxiter=100,
                seed=42,
                tol=1e-10,
            )

            best_params = list(result.x)
            prev_best_params = best_params

            vc_best = compute_vcorr_timeseries(df_seg, best_params)
            th_best = compute_thickness_timeseries(
                df_seg, vc_best, start_date, NOMINAL_THICKNESS_MM
            )
            # --- diagnostic output at VTD date ---
            meas_date = meas_row[DATE_COL]

            mask = df_seg[DATE_COL] == meas_date
            if mask.any():
                th_pred = th_best[mask].iloc[0]
                err_pct = 100 * (th_pred - thick_obs) / thick_obs

                print(
                    f"[VTD CHECK] segment={seg_idx} | date={meas_date.date()} | "
                    f"measured={thick_obs:.3f} mm | predicted={th_pred:.3f} mm | "
                    f"error={err_pct:+.2f} %"
                )
                params_row = {
                    "pipe_id": pipe_id,
                    "segment_index": seg_idx,
                    "vtd_date": meas_row[DATE_COL],
                    "thickness_measured": thick_obs,
                    "thickness_predicted": th_pred,
                    "error_pct": err_pct,
                }

                for i, val in enumerate(best_params):
                    params_row[f"param_{i+1}"] = val

                params_rows.append(params_row)
            else:
                print(
                    f"[VTD CHECK] segment={seg_idx} | date={meas_date.date()} | "
                    f"no predicted value"
                )
            df_out = df_seg.copy()
            df_out["V_corr_dewaard"] = vc_best
            df_out["thickness_calc"] = th_best
            df_out["segment_index"] = seg_idx

            all_results.append(df_out)
        if params_rows:
            df_params = pd.DataFrame(params_rows)
            df_params.to_csv("optimized_dewaard_params_50768.csv", index=False, encoding=ENC)
            print("[OK] Saved optimized parameters to optimized_dewaard_params_50768.csv")
        else:
            print("[WARN] No optimized parameters collected")

        if all_results:
            df_all = pd.concat(all_results, ignore_index=True)
            OUT_CSV = f"E:\\tmp code\\testing_NN\\data\done_data\\de_waard\\output_final\\full_pipeline_{pipe_id}_new2.csv"
            df_all.to_csv(OUT_CSV, index=False, encoding=ENC)
            print("[OK] Saved all segments to:", OUT_CSV)
        else:
            print("[WARN] No segments processed")


if __name__ == "__main__":
    main()
