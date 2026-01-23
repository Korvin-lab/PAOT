from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rough_temperature_model import RoughTemperatureModel


def norm_date(x) -> pd.Timestamp:
    dt = pd.to_datetime(x, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(x, dayfirst=True, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Не удалось распарсить дату: {x}")
    return dt.normalize()


def compute_q_heat_in_by_degree_w_per_k(
    row: pd.Series,
    *,
    q_unit: str,                     # "m3_day" | "m3_s"
    col_q_liq: str = "q",
    col_watercut: str = "watercut",  # % 
    col_rho_w: str = "rho_wat",
    col_rho_o: str = "rho_oil",
    col_cp_w: str = "hc_wat",
    col_cp_o: str = "hc_oil",
) -> float:
    """
    q_heat_in_by_degree = m_dot * cp_mix  (Вт/К)

    Используем:
    - q: расход жидкости (в единицах q_unit)
    - watercut: в процентах 
    - rho_wat / rho_oil: кг/м3
    - hc_wat / hc_oil: Дж/(кг*K)
    """
    for c in (col_q_liq, col_watercut, col_rho_w, col_rho_o, col_cp_w, col_cp_o):
        if c not in row.index:
            raise KeyError(f"Нет колонки '{c}' для расчёта q_heat_in_by_degree")

    q = float(row[col_q_liq])
    wc_percent = float(row[col_watercut])
    rho_w = float(row[col_rho_w])
    rho_o = float(row[col_rho_o])
    cp_w = float(row[col_cp_w])
    cp_o = float(row[col_cp_o])

    wc = wc_percent / 100.0
    if not (0.0 <= wc <= 1.0):
        raise ValueError(f"watercut вне диапазона [0..100]%: {wc_percent}")

    frac_w = wc
    frac_o = 1.0 - wc

    # q -> м3/с
    if q_unit == "m3_day":
        q_m3_s = q / 86400.0
    elif q_unit == "m3_s":
        q_m3_s = q
    else:
        raise ValueError(f"Неизвестный q_unit={q_unit}. Допустимо: m3_day, m3_s")

    # Смесь
    rho_mix = frac_w * rho_w + frac_o * rho_o
    cp_mix = frac_w * cp_w + frac_o * cp_o

    m_dot = q_m3_s * rho_mix     # кг/с
    return m_dot * cp_mix        # Вт/К


def main():
    # Входные
    CSV_FILE = "Hantos\\cleaned_tech_mode_pvt_corrosion.csv\\cleaned_tech_mode_pvt_corrosion.csv"
    PIPE_ID = "1250013334"
    DATE = "2023-05-26"
    OUTPUT_FILE = "temp_profile2.xlsx"  # выходной файл
    
    # Параметры
    Q_UNIT = "m3_day"               
    STEP_M = 10.0                   # шаг 
    DIRECTION = -1                  
    
    COL_ID = "id"
    COL_DATE = "date"
    COL_LENGTH = "distance"         
    COL_TBOUND = "t_mix"            
    
    # Физические параметры
    TID_MM = 426.0                  # внутренний диаметр (мм)
    TIR = 5e-5                      # шероховатость (м)
    HTC = 15                        #тепловой коэф
    T_SURR = 5.0                    # температура окружающей среды




    df = pd.read_csv(CSV_FILE)
    if COL_DATE not in df.columns:
        raise KeyError(f"В CSV нет колонки даты '{COL_DATE}'. Колонки: {list(df.columns)}")

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce").dt.normalize()
    target_date = norm_date(DATE)

    sub = df[
        (df[COL_ID].astype(str) == str(PIPE_ID)) &
        (df[COL_DATE] == target_date)
    ].copy()

    if sub.empty:
        raise ValueError(f"Не найдена строка для id={PIPE_ID}, date={target_date.date()}")

    # Если ВДРУГ несколько строк — берём первую
    row = sub.iloc[0]

    # Параметры из CSV
    if COL_LENGTH not in row.index:
        raise KeyError(f"В CSV нет колонки длины '{COL_LENGTH}'")
    if COL_TBOUND not in row.index:
        raise KeyError(f"В CSV нет колонки температуры потока '{COL_TBOUND}'")

    length = float(row[COL_LENGTH])     
    t_bound = float(row[COL_TBOUND])    

    print(f"\nПараметры:")
    print(f"  ID трубы: {PIPE_ID}")
    print(f"  Дата: {DATE}")
    print(f"  Длина: {length:.1f} м")
    print(f"  T_вход: {t_bound:.2f}°C")
    print(f"  T_окружающей среды: {T_SURR}°C")
    print(f"  Диаметр: {TID_MM} мм")
    print(f"  HTC: {HTC} Вт/(м²·К)")
    print(f"  Направление: {DIRECTION}")

    # Константы
    tid_m = float(TID_MM) / 1000.0      # мм -> м
    tir = float(TIR)
    htc = float(HTC)
    t_surr = float(T_SURR)

    # Расчёт теплового потока
    q_heat = compute_q_heat_in_by_degree_w_per_k(row, q_unit=Q_UNIT)
    print(f"  q_heat: {q_heat:.1f} Вт/К")

    # Запуск
    print(f"\nРасчёт модели температуры...")
    model = RoughTemperatureModel(
        length=length,
        tid=tid_m,
        tir=tir,
        q_heat_in_by_degree=q_heat,
        surrounding_temperature=t_surr,
        htc=htc,
        t_bound=t_bound,
        direction=int(DIRECTION),
    )

    sol = model.run()

    
    xs = np.arange(0.0, length + 1e-9, float(STEP_M))
    if xs[-1] < length:
        xs = np.append(xs, length)

    print(f"\nРассчитано точек: {len(xs)}")
    

    temps = [float(sol(length - x)) for x in xs]
    

    out = pd.DataFrame({
        "id": str(PIPE_ID),
        "date": target_date.date().isoformat(),
        "x_m": xs,
        "temperature_c": temps,
    })

    out['delta_T_c'] = out['temperature_c'].diff()

    print(f"\nПредпросмотр результатов:")
    print(out.head(10).to_string(index=False))
    print("...")
    print(out.tail(5).to_string(index=False))

    print(f"\nИтоговое изменение температуры:")
    print(f"  T(0м) = {temps[0]:.2f}°C")
    print(f"  T({length:.0f}м) = {temps[-1]:.2f}°C")
    print(f"  ΔT = {temps[-1] - temps[0]:.2f}°C")


    out_path = Path(OUTPUT_FILE)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="temperature_profile", index=False)

        # Мета
        meta = pd.DataFrame([{
            "id": str(PIPE_ID),
            "date": target_date.date().isoformat(),
            "length_m": length,
            "tid_m": tid_m,
            "tir_m": tir,
            "htc_w_m2_k": htc,
            "surrounding_temperature_c": t_surr,
            "t_bound_c": t_bound,
            "q_heat_in_by_degree_w_per_k": q_heat,
            "step_m": float(STEP_M),
            "direction": int(DIRECTION),
            "q_unit": Q_UNIT,
        }])
        meta.to_excel(w, sheet_name="inputs", index=False)

    print(f"\n{'='*70}")
    print(f"✓ Файл сохранён: {out_path}")
    print(f"✓ Рассчитано {len(xs)} точек с шагом {STEP_M} м")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
