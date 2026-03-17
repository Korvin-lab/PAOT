import numpy as np
import pandas as pd

def norsok_fanning_friction_factor(Re: float) -> float:
    if Re <= 0:
        return 0.0
    if Re < 2100:
        return 16.0 / Re
    elif Re > 4000:
        return 0.079 * Re ** (-0.25)
    else:
        # Линейная интерполяция в переходной зоне
        f_lam = 16.0 / 2100
        f_turb = 0.079 * 4000 ** (-0.25)
        alpha = (Re - 2100.0) / (4000.0 - 2100.0)
        return f_lam + alpha * (f_turb - f_lam)


def corr_(density_liquid_work, v_liquid_true, T, M, P, pH, Re):
    C = [0.155039297886972,
    0.002274142231553,
    1.285337320903888,
    -0.004985773989267,
    0.003187963937826,
    -0.004342338910411,
    -0.009554205614705,
    0.000015061725195,
    0.000096194098987,
    0.000098281893720,
    0.077862218607460,
    -0.047167089992578,
    0.093224937102132,
    0.000301374720830,
    0.000167047079814,
    0.000222045051777,
    -0.000071462072302,
    0.000015094110076,
    0.001941338650808,
    -0.000001035505597,
    -0.000002596083890,
    0.000000515237464,
    -0.000198914658621,
    -0.011833616220902,
    0.008673800083278,
    -0.000008005769895
    ]
    f = norsok_fanning_friction_factor(Re)
    S = 0.5 * f * density_liquid_work * (v_liquid_true ** 2)
    #df_table = pd.read_excel('clear_data.xlsx')
    # S = float(input("Введите значение КННС (Па)")) #КННС изменить на WSS
    # T = float(input("Введите значение Температуры (С)"))
    # M = float(input("Введите значение Минерализации (г/л)"))
    # P = float(input("Введите значение Давления (МПа)"))
    # pH = float(input("Введите значение pH (моль/л)"))
    Vcorr = C[0] + C[1]*T + C[2]*P + C[3]*M + C[4]*S + C[5]*pH + C[6]*T*P + C[7]*T*M + C[8]*T*S + C[9]*T*pH + \
             + C[10]*P*M + C[11]*P*S + C[12]*P*pH + C[13]*M*S + C[14]*M*pH + C[15]*S*pH + C[16]*T*P*M \
             + C[17]*T*P*S+C[18]*T*P*pH + C[19]*T*M*S + C[20]*T*M*pH + C[21]*T*S*pH+C[22]*P*M*S + \
             + C[23]*P*M*pH + C[24]*P*S*pH + C[25]*M*S*pH

    return Vcorr

def V_kor_de_vaard_easy(t, p, CO2):
    """
    Если расчет невозможен — возвращает 0
    Работает и для scalar, и для numpy array
    """
    try:
        t = np.asarray(t, dtype=float)
        p_CO2 = p * CO2
        p_CO2 = np.asarray(p_CO2, dtype=float)

        # безопасная защита для log10
        p_CO2 = np.where(p_CO2 <= 0, 1e-12, p_CO2)

        logV = 7.96 - 2320.0 / (273.0 + t) - 0.0055 * t + 0.67 * np.log10(p_CO2)
        V = 10 ** logV

        return V

    except Exception as e:
        # Если вход был массив — вернуть массив нулей той же формы
        if hasattr(t, "shape"):
            return np.zeros_like(t, dtype=float), np.zeros_like(t, dtype=float)
        # Если скаляр — вернуть 0
        return np.nan

def to_kelvin(t_celsius: float) -> float:
    return t_celsius + 273.15

def kinematic_viscosity(T: float) -> float:
    return 10 ** (((1.3272 * (20-T) - 0.001053 * (T-10)) / (T+105)) - 6)

def dif_koef(T: float, V: float) -> float:
    return T / V * 10 ** -17

def henry_koef(T: float) -> float:
    return 10 ** ((1088.76 / T) - 5.113)

def parc_CO2(p: float, CO2: float) -> float:
    return p * CO2

def carb_acid_concentration(H: float, pCO2: float) -> float:
    return H * pCO2

def fugitive_koef(p: float, T: float) -> float:
    if p >= 250:
        return 10 ** (250 * (0.0031 - 1.4 / T))
    else:
        return 10 ** (p * (0.0031 - 1.4 / T))

def fugitivity(a: float, pCO2: float) -> float:
    return a * pCO2

def v_react(T: float, fCO2: float) -> float:
    return 10 ** (5.8 - 1543/T + 0.67 * np.log10(fCO2))

def v_mass(D, U, v, d, H2CO3) -> float:
    return 0.023 * (D ** 0.7) / (v ** 0.5) * (U ** 0.8) / (d ** 0.2) * H2CO3

def V_kor_de_vaard_hard(t, p, CO2, v_liquid_true, d_mm):
    try:
        T = to_kelvin(t)
        V = kinematic_viscosity(T)
        D = dif_koef(T, V)
        H = henry_koef(T)
        pCO2 = parc_CO2(p, CO2)
        a = fugitive_koef(p, T)
        fCO2 = fugitivity(a, pCO2)
        H2CO3 = carb_acid_concentration(H, pCO2)

        return 1 / (1 / ((2.62 * 10**6 * v_mass(D, v_liquid_true, V, d_mm, H2CO3))) + 1 / v_react(T, fCO2))
    except Exception as e:
        return np.nan
    
if __name__ == "__main__":

    df = pd.read_csv('./data/raw/tech_modes_25.11.2024.csv')
    df['calc_date'] = pd.to_datetime(df['calc_date']).dt.strftime('%Y-%m-%d')

    old_tech_mode = pd.read_csv('./data/raw/tech_mode_res.csv')
    old_tech_mode = old_tech_mode.rename(columns={'id':'simple_section_id','date': 'calc_date'})

    df = df.merge(
        old_tech_mode[['simple_section_id', 'calc_date', 'CO2', 'Min', 'pH', 'reynolds', 'd_mm']],
        on=['simple_section_id', 'calc_date'],
        how='left'
    )

    df["Vcorr_politech"] = df.apply(
        lambda row: corr_(row["density_liquid_work"], row["v_liquid_true"], row["t_mix"], row["Min"], row["p_start"], row["pH"], row["reynolds"]),
        axis=1
    )
    df["Vcorr_de_vaard1"] = df.apply(
        lambda row: V_kor_de_vaard_easy(row["t_mix"], row["p_start"], row["CO2"]),
        axis=1
    )
    df["Vcorr_de_vaard2"] = df.apply(
        lambda row: V_kor_de_vaard_hard(row["t_mix"], row["p_start"], row["CO2"], row["v_liquid_true"], row["d_mm"]),
        axis=1
    )