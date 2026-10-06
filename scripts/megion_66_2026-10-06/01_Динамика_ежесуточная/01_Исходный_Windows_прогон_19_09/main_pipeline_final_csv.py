from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MASTER_JSON = BASE_DIR / 'input/graph_orenburg_master.json'
DEFAULT_REQUESTED_JSON = BASE_DIR / 'input/graph_orenburg_requested_daily_params.json'
DEFAULT_OUT_DIR = BASE_DIR / 'output_orenburg/variant_strict_orenburg'
DEFAULT_PE2_DLL = BASE_DIR / 'deps/pe_2_main.dll'

EPS_M = 5e-5
PSI_PER_PA = 0.00014503773773020923
# For naporny lines we allow missing/zero q_gas in mask and replace with a
# tiny strictly-positive value to keep PVT/PE2 numerically stable.
# Unit here is thousand m3/day, so 1e-7 ~= 0.1 m3/day (physically near-zero).
NAPORNY_Q_GAS_FALLBACK_TSD_M3D = 1e-7
SECONDS_PER_DAY = 86400.0
POLYTECH_PRESSURE_MPA_TO_ATM = 10.0
POLYTECH_ROUGHNESS_M = 0.00005
MIN_PRESSURE_MPA = 0.01
MIN_TEMPERATURE_C = 5.0
MAX_TEMPERATURE_C = 90.0
SURROUNDING_TEMPERATURE_C = 5.0
WINDOWS_RESERVED_DEVICES = {"NUL", "CON", "PRN", "AUX"}
try:
    _run_config = json.loads((BASE_DIR / 'run_config.json').read_text(encoding='utf-8'))
    SURROUNDING_TEMPERATURE_C = float(_run_config.get('temperature_model_surrounding_c', SURROUNDING_TEMPERATURE_C))
except Exception:
    pass

FIELD_MAP = {
    'Жидкости, кг/м3': 'rho_wat',
    'Жидкости, м3/сут (дебит)': 'q_liq',
    'Обводненность, %': 'watercut',
    'Нефти, т/сут (дебит)': 'q_oil',
    'Газа, кг/м3': 'rho_gas',
    'Смеси, кг/м3': 'mix_rho',
    'Газа, кг/(м*с)*1000': 'mug',
    'Жидкости, кг/(м*с)': 'muw',
    'Общего газа, тыс.м3/сут (дебит)': 'q_gas',
    'Истинная жидкости, м/с': 'v_liquid_true',
    'Нефти, кг/м3': 'rho_oil',
    'Смеси, м/с': 'v_mix',
    't': 't',
    'p': 'p',
    'L': 'L',
    'S': 'S',
    'D': 'D',
}

REQUIRED_BASE_COLS = ['q_liq', 'q_oil', 'q_gas', 'rho_oil', 'rho_wat', 'rho_gas', 'watercut', 'D', 't', 'p']

PE2_TRACE_COLUMNS = [
    'trace_seq',
    'calc_mode',
    'stage',
    'source_fn',
    'pipe_id',
    'date',
    'distance_m',
    'segment_step_m',
    'selected_gzu_for_tp',
    'tp_pick_rule',
    'temperature_c',
    'pressure_mpa_in',
    'pressure_bar_in',
    'input_q_oil',
    'input_q_wat',
    'input_q_gas',
    'input_rhoo',
    'input_rhow',
    'input_rhog',
    'input_wct',
    'input_muo',
    'input_muw',
    'input_mug',
    'input_gl_ift',
    'input_diameter',
    'input_eps',
    'input_angle',
    'input_flow_direction_coef',
    'input_p1',
    'input_p2',
    'output_holdup',
    'output_gravity_gradient',
    'output_friction_gradient',
    'output_acceleration_gradient',
    'output_total_gradient',
    'output_regime',
    'output_regime_desc',
    'output_next_pressure_bar',
    'output_next_pressure_mpa',
    'error',
]

PE2_TRACE_ENABLED = False
PE2_TRACE_PATH: Path | None = None
PE2_TRACE_HEADER_WRITTEN = False
PE2_TRACE_ROWS_WRITTEN = 0
PE2_TRACE_SEQ = 0
PE2_TRACE_FLUSH_EVERY = 5000
PE2_TRACE_BUFFER: list[dict[str, Any]] = []


def bootstrap_runtime_imports(pe2_dll_path: Path) -> None:
    """Import runtime dependencies and require PE2 DLL to be available."""
    fallback_dirs = [BASE_DIR]

    for d in fallback_dirs:
        d_str = str(d)
        if d.exists() and d_str not in sys.path:
            sys.path.insert(0, d_str)

    global RoughTemperatureModel, pe_2_correlation, FluidFlow, PE2_SOURCE
    from rough_temperature_model import RoughTemperatureModel  # type: ignore
    from unifloc.pvt.fluid_flow import FluidFlow  # type: ignore

    pe2_dll_path = Path(pe2_dll_path)
    if not pe2_dll_path.exists():
        raise FileNotFoundError(
            f'PE2 DLL не найден: {pe2_dll_path}. Без DLL расчет давления недопустим.'
        )

    class Result(ctypes.Structure):
        _fields_ = [
            ('holdup', ctypes.c_double),
            ('gravity_gradient', ctypes.c_double),
            ('friction_gradient', ctypes.c_double),
            ('acceleration_gradient', ctypes.c_double),
            ('total_gradient', ctypes.c_double),
            ('regime', ctypes.c_int),
            ('regime_desc', ctypes.c_char_p),
        ]

    try:
        pe2_lib = ctypes.CDLL(str(pe2_dll_path))
    except OSError as exc:
        raise RuntimeError(
            f'PE2 DLL не удалось загрузить: {pe2_dll_path}. '
            f'Проверьте ОС/разрядность и совместимость библиотеки. Исходная ошибка: {exc}'
        ) from exc

    pe2_lib.run.argtypes = [ctypes.c_double] * 17
    pe2_lib.run.restype = Result

    def pe2_dll_wrapper(
        q_oil: float,
        q_wat: float,
        q_gas: float,
        rhoo: float,
        rhow: float,
        rhog: float,
        wct: float,
        muo: float,
        muw: float,
        mug: float,
        gl_ift: float,
        diameter: float,
        eps: float,
        angle: float,
        pressure: float,
        p1: float,
        p2: float,
    ) -> list[Any]:
        res = pe2_lib.run(
            q_oil,
            q_wat,
            q_gas,
            rhoo,
            rhow,
            rhog,
            wct,
            muo,
            muw,
            mug,
            gl_ift,
            diameter,
            eps,
            angle,
            pressure,
            p1,
            p2,
        )
        return [
            res.holdup,
            res.gravity_gradient,
            res.friction_gradient,
            res.acceleration_gradient,
            res.total_gradient,
            res.regime,
            res.regime_desc,
        ]

    pe_2_correlation = pe2_dll_wrapper
    PE2_SOURCE = f'dll_pe2:{pe2_dll_path}'


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_id(value: object) -> str:
    txt = clean_text(value)
    if not txt:
        return ''
    txt = txt.replace(' ', '')
    if re.fullmatch(r'\d+\.0+', txt):
        txt = txt.split('.', 1)[0]
    digits = ''.join(re.findall(r'\d+', txt))
    return digits if digits else txt.upper()


def get_rough_direction() -> int:
    """
    Runtime toggle for temperature model direction.
    ROUGH_DIRECTION=-1 or 1 (default 1).
    """
    raw = os.getenv('ROUGH_DIRECTION', '1')
    try:
        val = int(str(raw).strip())
    except Exception:
        val = 1
    return -1 if val < 0 else 1


def build_segment_distances(length_m: float, step_m: int) -> np.ndarray:
    """Build full steps and one exact final point without exceeding pipe length."""
    length = float(length_m)
    step = float(step_m)
    if not np.isfinite(length) or length <= 0 or not np.isfinite(step) or step <= 0:
        return np.array([], dtype=float)
    full_steps = int(np.floor(length / step))
    points = np.arange(full_steps + 1, dtype=float) * step
    if points.size == 0 or not np.isclose(points[-1], length, atol=1e-9, rtol=0.0):
        points = np.append(points, length)
    else:
        points[-1] = length
    return points


def classify_pressure_boundary(source_p: Any) -> str:
    """Return the physical coordinate where the supplied pressure is defined."""
    source = clean_text(source_p).lower().replace('ё', 'е')
    if re.search(r'(^|\s)(p|р)?\s*(факт|расчет|расч)[^;]*кон(ец|ца)', source):
        return 'end'
    if re.search(r'(^|\s)(p|р)?\s*(факт|расчет|расч)[^;]*начал', source):
        return 'start'
    if 'штр' in source or 'скважин' in source or 'агзу' in source or 'граф от предшественник' in source:
        return 'start'
    return ''


def assign_pressure_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Assign start/end pressure per date; synthetic dates inherit nearest known side."""
    out = df.sort_values('date').reset_index(drop=True).copy()
    if 'source_p' not in out.columns:
        out['source_p'] = ''
    direct = out['source_p'].map(classify_pressure_boundary)
    dates = pd.to_datetime(out['date'], errors='coerce')
    known = [i for i, side in enumerate(direct.tolist()) if side]
    known_dates_ns = np.array([dates.iloc[i].value for i in known], dtype=np.int64) if known else np.array([], dtype=np.int64)
    sides: list[str] = []
    rules: list[str] = []
    for i, side in enumerate(direct.tolist()):
        if side:
            sides.append(side)
            rules.append('direct_source_p')
            continue
        if known and pd.notna(dates.iloc[i]):
            position = int(np.searchsorted(known_dates_ns, dates.iloc[i].value, side='left'))
            candidate_positions = [p for p in [position - 1, position] if 0 <= p < len(known)]
            nearest = min(
                (known[p] for p in candidate_positions),
                key=lambda j: (abs((dates.iloc[j] - dates.iloc[i]).days), 0 if dates.iloc[j] <= dates.iloc[i] else 1, j),
            )
            sides.append(direct.iloc[nearest])
            rules.append(f'nearest_known_source_p:{dates.iloc[nearest]:%Y-%m-%d}')
        else:
            sides.append('start')
            rules.append('default_start_no_known_boundary')
    out['pressure_boundary_side'] = sides
    out['pressure_boundary_rule'] = rules
    out['flow_direction_coef'] = np.where(out['pressure_boundary_side'].eq('end'), -1.0, 1.0)
    return out


def classify_temperature_boundary(source_t: Any) -> str:
    """Return the physical coordinate where the supplied temperature is defined."""
    source = clean_text(source_t).lower().replace('ё', 'е')
    if '_end' in source or 't_end' in source or 'конец' in source or 'конца' in source:
        return 'end'
    if any(token in source for token in ('начал', 'средн', 'штр', 'скважин', 'агзу', 'граф от предшественник', 'дозаполнение', 'interpolation', 'nearest', 'seasonal', 'восстановление')):
        return 'start'
    return ''


def assign_temperature_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a temperature boundary; temporal fills inherit the nearest direct side."""
    out = df.copy()
    if 'date' in out.columns:
        out = out.sort_values('date').reset_index(drop=True)
    if 'source_t' not in out.columns:
        out['source_t'] = ''
    explicit = (
        out.get('temperature_boundary_side', pd.Series('', index=out.index))
        .astype(str)
        .str.strip()
        .str.lower()
    )
    inferred = out['source_t'].map(classify_temperature_boundary)
    direct = explicit.where(explicit.isin({'start', 'end'}), inferred)
    dates = pd.to_datetime(out.get('date', pd.Series(pd.NaT, index=out.index)), errors='coerce')
    known = [i for i, side in enumerate(direct.tolist()) if side]
    known_dates_ns = np.array([dates.iloc[i].value for i in known], dtype=np.int64) if known else np.array([], dtype=np.int64)
    known_dates_complete = bool(known) and bool(dates.iloc[known].notna().all())
    sides: list[str] = []
    rules: list[str] = []
    for i, side in enumerate(direct.tolist()):
        if side:
            sides.append(side)
            rules.append('direct_source_t')
            continue
        if known_dates_complete and pd.notna(dates.iloc[i]):
            position = int(np.searchsorted(known_dates_ns, dates.iloc[i].value, side='left'))
            candidate_positions = [p for p in [position - 1, position] if 0 <= p < len(known)]
            nearest = min(
                (known[p] for p in candidate_positions),
                key=lambda j: (abs((dates.iloc[j] - dates.iloc[i]).days), 0 if dates.iloc[j] <= dates.iloc[i] else 1, j),
            )
            sides.append(direct.iloc[nearest])
            rules.append(f'nearest_known_source_t:{dates.iloc[nearest]:%Y-%m-%d}')
        else:
            sides.append('start')
            rules.append('default_start_no_known_boundary')
    out['temperature_boundary_side'] = sides
    out['temperature_boundary_rule'] = rules
    out['temperature_direction_coef'] = np.where(out['temperature_boundary_side'].eq('end'), -1.0, 1.0)
    return out


def to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')


def to_float_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        if pd.isna(value):
            return None
        return float(value)
    txt = str(value).strip().replace(',', '.')
    if not txt:
        return None
    try:
        val = float(txt)
    except Exception:
        return None
    if pd.isna(val):
        return None
    return float(val)


def polytech_inner_diameter_m(d_m: Any, wall_mm: Any) -> float | None:
    d_val = to_float_value(d_m)
    if d_val is None or d_val <= 0:
        return None
    s_val = to_float_value(wall_mm)
    if s_val is None or s_val <= 0:
        return d_val
    d_inner = d_val - 2.0 * s_val / 1000.0
    return d_inner if d_inner > 0 else d_val


def polytech_q_gas_work_m3s(
    q_gas_norm_tsd_m3d: Any,
    pressure_mpa: Any,
    temperature_c: Any,
) -> float | None:
    """
    Polytech gas conversion from normal to working conditions.

    Source workbook formula:
      Qgas_work = (Qgas_norm_tsd_m3d * 1000 / 86400) * T_K / (P_atm * 273.15)

    Despite the workbook column title "тыс.м3/сут", the formula returns m3/s.
    PE2 still expects q_gas in тыс.м3/сут, so callers convert m3/s back when needed.
    """
    q_norm = to_float_value(q_gas_norm_tsd_m3d)
    p_mpa = to_float_value(pressure_mpa)
    t_c = to_float_value(temperature_c)
    if q_norm is None or p_mpa is None or t_c is None:
        return None
    p_atm = p_mpa * POLYTECH_PRESSURE_MPA_TO_ATM
    t_k = t_c + 273.15
    if q_norm < 0 or p_atm <= 0 or t_k <= 0:
        return None
    out = (q_norm * 1000.0 / SECONDS_PER_DAY) * t_k / (p_atm * 273.15)
    if not np.isfinite(out) or out < 0:
        return None
    return float(out)


def polytech_q_gas_work_tsd_m3d(
    q_gas_norm_tsd_m3d: Any,
    pressure_mpa: Any,
    temperature_c: Any,
) -> float | None:
    q_m3s = polytech_q_gas_work_m3s(q_gas_norm_tsd_m3d, pressure_mpa, temperature_c)
    if q_m3s is None:
        return None
    return float(q_m3s * SECONDS_PER_DAY / 1000.0)


def add_polytech_segment_flow_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Polytech working gas, flow and velocity columns per segment.
    The public q_gas column is replaced with working-condition gas in PE2 units
    (тыс.м3/сут); the original normal-condition gas is kept separately.
    """
    out = df.copy()
    for col in ['q_gas', 'q_liq', 'D', 'S', 'p', 't', 'rho_oil', 'rho_wat', 'rho_gas', 'watercut', 'mug', 'mu_mix']:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors='coerce')

    if 'q_gas_norm_tsd_m3d' not in out.columns:
        out['q_gas_norm_tsd_m3d'] = out['q_gas']
    else:
        out['q_gas_norm_tsd_m3d'] = pd.to_numeric(out['q_gas_norm_tsd_m3d'], errors='coerce').where(
            pd.to_numeric(out['q_gas_norm_tsd_m3d'], errors='coerce').notna(),
            out['q_gas'],
        )

    if 'v_mix_source_techregime' not in out.columns:
        out['v_mix_source_techregime'] = pd.to_numeric(out.get('v_mix', np.nan), errors='coerce')
    if 'q_gas_source_norm_tsd_m3d' not in out.columns:
        out['q_gas_source_norm_tsd_m3d'] = out['q_gas_norm_tsd_m3d']

    p_atm = out['p'] * POLYTECH_PRESSURE_MPA_TO_ATM
    t_k = out['t'] + 273.15
    q_norm = out['q_gas_norm_tsd_m3d']

    q_gas_work_m3s = pd.Series(np.nan, index=out.index, dtype=float)
    ok_gas = q_norm.notna() & (q_norm >= 0) & p_atm.notna() & (p_atm > 0) & t_k.notna() & (t_k > 0)
    q_gas_work_m3s.loc[ok_gas] = (
        q_norm.loc[ok_gas] * 1000.0 / SECONDS_PER_DAY
        * t_k.loc[ok_gas] / (p_atm.loc[ok_gas] * 273.15)
    )

    s_m = out['S'] / 1000.0
    d_inner = out['D'] - 2.0 * s_m
    d_inner = d_inner.where(d_inner.notna() & (d_inner > 0), out['D'])
    area = np.pi * d_inner * d_inner / 4.0

    q_liq_m3s = out['q_liq'] / SECONDS_PER_DAY
    q_mix_m3s = q_liq_m3s + q_gas_work_m3s
    v_mix = q_mix_m3s / area

    alpha_gas = q_gas_work_m3s / q_mix_m3s
    alpha_gas = alpha_gas.where(q_mix_m3s.notna() & (q_mix_m3s > 0))
    alpha_gas = alpha_gas.clip(lower=0.0, upper=1.0)
    alpha_liq = 1.0 - alpha_gas

    wc = (out['watercut'] / 100.0).clip(lower=0.0, upper=1.0)
    rho_liq = out['rho_oil'] * (1.0 - wc) + out['rho_wat'] * wc
    rho_mix = rho_liq * alpha_liq + out['rho_gas'] * alpha_gas

    # Workbook logic uses liquid viscosity plus gas viscosity converted from "*1000" notation.
    mu_liq = out['mu_mix']
    mu_gas_pas = out['mug'] * 0.001
    mu_mix = mu_liq * alpha_liq + mu_gas_pas * alpha_gas

    re_polytech = pd.Series(np.nan, index=out.index, dtype=float)
    ok_re = v_mix.notna() & d_inner.notna() & rho_mix.notna() & mu_mix.notna() & (mu_mix > 0)
    re_polytech.loc[ok_re] = v_mix.loc[ok_re] * d_inner.loc[ok_re] * rho_mix.loc[ok_re] / mu_mix.loc[ok_re]

    rough_rel = POLYTECH_ROUGHNESS_M / d_inner
    ff = pd.Series(np.nan, index=out.index, dtype=float)
    lam = re_polytech.notna() & (re_polytech > 0) & (re_polytech <= 5000)
    turb = re_polytech.notna() & (re_polytech > 5000)
    ff.loc[lam] = 64.0 / re_polytech.loc[lam]
    ff.loc[turb] = 0.001375 * (
        1.0 + (20000.0 * rough_rel.loc[turb] + 1000000.0 / re_polytech.loc[turb]) ** 0.33
    )
    knns = 0.5 * ff * rho_mix * v_mix * v_mix

    out['D_inner_polytech_m'] = d_inner
    out['q_gas_work_m3s_polytech'] = q_gas_work_m3s
    out['q_gas_work_tsd_m3d_polytech'] = q_gas_work_m3s * SECONDS_PER_DAY / 1000.0
    out['q_liq_m3s_polytech'] = q_liq_m3s
    out['q_mix_m3s_polytech'] = q_mix_m3s
    out['alpha_liq_polytech'] = alpha_liq
    out['alpha_gas_polytech'] = alpha_gas
    out['rho_liq_polytech'] = rho_liq
    out['rho_mix_polytech'] = rho_mix
    out['mu_mix_polytech'] = mu_mix
    out['v_mix_polytech'] = v_mix
    out['re_polytech'] = re_polytech
    out['ff_polytech'] = ff
    out['knns_polytech'] = knns
    out['q_gas'] = out['q_gas_work_tsd_m3d_polytech']
    out['v_mix'] = out['v_mix_polytech']
    return out


def normalize_distance_key(value: Any) -> float | None:
    dist = to_float_value(value)
    if dist is None:
        return None
    if abs(dist) < 1e-12:
        dist = 0.0
    return round(float(dist), 6)


def load_altitudes_map(altitudes_csv: Path | None) -> tuple[dict[str, dict[float, float]], dict[str, Any]]:
    meta: dict[str, Any] = {
        'enabled': bool(altitudes_csv),
        'path': str(altitudes_csv) if altitudes_csv else '',
        'rows_raw': 0,
        'rows_valid': 0,
        'ids_total': 0,
        'angle_col': '',
    }
    if altitudes_csv is None:
        return {}, meta

    altitudes_csv = Path(altitudes_csv)
    if not altitudes_csv.exists():
        raise FileNotFoundError(f'Файл altitudes.csv не найден: {altitudes_csv}')

    raw_df = pd.read_csv(altitudes_csv, low_memory=False)
    meta['rows_raw'] = int(len(raw_df))

    if 'id' not in raw_df.columns or 'segment_start_distance' not in raw_df.columns:
        raise ValueError('altitudes.csv должен содержать колонки id и segment_start_distance.')

    angle_col = ''
    for candidate in ['slope_deg', 'angle_deg', 'slope']:
        if candidate in raw_df.columns:
            angle_col = candidate
            break
    if not angle_col:
        raise ValueError('altitudes.csv не содержит колонку угла (slope_deg/angle_deg/slope).')
    meta['angle_col'] = angle_col

    df = raw_df[['id', 'segment_start_distance', angle_col]].copy()
    df['id'] = df['id'].map(normalize_id)
    df['distance_key'] = df['segment_start_distance'].map(normalize_distance_key)
    df['angle_deg'] = to_float(df[angle_col])
    df = df[df['id'].astype(bool) & df['distance_key'].notna() & df['angle_deg'].notna()].copy()
    meta['rows_valid'] = int(len(df))

    if df.empty:
        return {}, meta

    agg = (
        df.groupby(['id', 'distance_key'], as_index=False)['angle_deg']
        .median()
        .sort_values(['id', 'distance_key'])
        .reset_index(drop=True)
    )

    out: dict[str, dict[float, float]] = defaultdict(dict)
    for r in agg.itertuples(index=False):
        out[str(r.id)][float(r.distance_key)] = float(r.angle_deg)

    meta['ids_total'] = int(len(out))
    return dict(out), meta


def init_pe2_trace(path: Path | None) -> None:
    global PE2_TRACE_ENABLED, PE2_TRACE_PATH, PE2_TRACE_HEADER_WRITTEN
    global PE2_TRACE_ROWS_WRITTEN, PE2_TRACE_SEQ, PE2_TRACE_BUFFER

    PE2_TRACE_ENABLED = path is not None
    PE2_TRACE_PATH = Path(path) if path is not None else None
    PE2_TRACE_HEADER_WRITTEN = False
    PE2_TRACE_ROWS_WRITTEN = 0
    PE2_TRACE_SEQ = 0
    PE2_TRACE_BUFFER = []

    if not PE2_TRACE_ENABLED or PE2_TRACE_PATH is None:
        return

    if str(PE2_TRACE_PATH).strip().upper() in WINDOWS_RESERVED_DEVICES:
        PE2_TRACE_ENABLED = False
        PE2_TRACE_PATH = None
        return
    PE2_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PE2_TRACE_PATH.exists():
        PE2_TRACE_PATH.unlink()


def _serialize_trace_value(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode('utf-8', errors='replace')
        except Exception:
            return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime('%Y-%m-%d')
    return value


def flush_pe2_trace() -> None:
    global PE2_TRACE_BUFFER, PE2_TRACE_HEADER_WRITTEN, PE2_TRACE_ROWS_WRITTEN

    if not PE2_TRACE_ENABLED or PE2_TRACE_PATH is None or not PE2_TRACE_BUFFER:
        return

    with PE2_TRACE_PATH.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=PE2_TRACE_COLUMNS)
        if not PE2_TRACE_HEADER_WRITTEN:
            writer.writeheader()
            PE2_TRACE_HEADER_WRITTEN = True
        for row in PE2_TRACE_BUFFER:
            writer.writerow({k: _serialize_trace_value(row.get(k)) for k in PE2_TRACE_COLUMNS})

    PE2_TRACE_ROWS_WRITTEN += len(PE2_TRACE_BUFFER)
    PE2_TRACE_BUFFER = []


def append_pe2_trace(row: dict[str, Any]) -> None:
    global PE2_TRACE_BUFFER, PE2_TRACE_SEQ
    if not PE2_TRACE_ENABLED:
        return
    PE2_TRACE_SEQ += 1
    record = {k: row.get(k) for k in PE2_TRACE_COLUMNS}
    record['trace_seq'] = PE2_TRACE_SEQ
    PE2_TRACE_BUFFER.append(record)
    if len(PE2_TRACE_BUFFER) >= PE2_TRACE_FLUSH_EVERY:
        flush_pe2_trace()


def finalize_pe2_trace() -> None:
    flush_pe2_trace()
    if PE2_TRACE_ENABLED and PE2_TRACE_PATH is not None and not PE2_TRACE_PATH.exists():
        with PE2_TRACE_PATH.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=PE2_TRACE_COLUMNS)
            writer.writeheader()


def choose_first_kust_with_tp(
    t_points: list[dict[str, Any]] | None,
    p_points: list[dict[str, Any]] | None,
) -> tuple[float | None, float | None, str]:
    """
    Business rule:
    - for multi-kust rows choose the first encountered KUST/GZU that has both T and P values.
    - if no such kust exists, return (None, None, '') and keep original row values.
    """
    t_points = t_points or []
    p_points = p_points or []

    ordered_kust: list[str] = []
    for pt in t_points + p_points:
        gzu = clean_text(pt.get('gzu'))
        if gzu and gzu not in ordered_kust:
            ordered_kust.append(gzu)

    def first_valid(points: list[dict[str, Any]], gzu: str) -> float | None:
        for pt in points:
            if clean_text(pt.get('gzu')) != gzu:
                continue
            val = to_float_value(pt.get('value'))
            if val is not None:
                return val
        return None

    for gzu in ordered_kust:
        t_val = first_valid(t_points, gzu)
        p_val = first_valid(p_points, gzu)
        if t_val is not None and p_val is not None:
            return t_val, p_val, gzu

    return None, None, ''


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def export_json_to_csv(master_json: Path, requested_json: Path, csv_dir: Path) -> tuple[Path, Path, Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)

    master = load_json(master_json)
    requested = load_json(requested_json)

    node_rows: list[dict[str, Any]] = []
    for n in master.get('nodes', []):
        g = n.get('graph', {})
        k = n.get('kust_binding', {})
        node_rows.append(
            {
                'id': normalize_id(n.get('id')),
                'found_in_graph': bool(g.get('found_in_graph', False)),
                'main_id': g.get('main_id'),
                'simple_name': g.get('simple_name'),
                'object_name': g.get('object_name'),
                'start_node_norm': g.get('start_node_norm'),
                'end_node_norm': g.get('end_node_norm'),
                'L_m_graph': g.get('L_m'),
                'D_mm_graph': g.get('D_mm'),
                'S_mm_graph': g.get('S_mm'),
                'eligible_one_kust': bool(k.get('eligible_one_kust', False)),
                'kust': k.get('kust'),
                'kust_count_in_sources': int(k.get('kust_count_in_sources', 0) or 0),
                'location_by_sources': k.get('location_by_sources'),
            }
        )

    edge_rows: list[dict[str, Any]] = []
    for e in master.get('edges', []):
        edge_rows.append(
            {
                'source': normalize_id(e.get('source')),
                'target': normalize_id(e.get('target')),
                'main_id': e.get('main_id'),
                'via_node': e.get('via_node'),
                'edge_type': e.get('edge_type', ''),
            }
        )

    daily_rows: list[dict[str, Any]] = []
    by_id = requested.get('by_id', {})
    for pipe_id, payload in by_id.items():
        pid = normalize_id(pipe_id)
        for rec in payload.get('daily', []):
            t_points = rec.get('t_points') or []
            p_points = rec.get('p_points') or []
            chosen_t, chosen_p, chosen_gzu = choose_first_kust_with_tp(t_points, p_points)

            row: dict[str, Any] = {'id': pid, 'date': rec.get('Дата', rec.get('date'))}
            for k, v in rec.items():
                if k in {'Дата', 'date'}:
                    continue
                if k in {'t_points', 'p_points'}:
                    sources = sorted({clean_text(x.get('source')) for x in (v or []) if clean_text(x.get('source'))})
                    row[f'{k}_sources'] = '; '.join(sources)
                    row[f'{k}_count'] = len(v or [])
                    continue
                row[k] = v

            # Override row-level t/p by the explicit business rule for multi-kust data.
            # If rule did not find a valid pair, keep original record values.
            if chosen_t is not None and chosen_p is not None:
                row['t'] = chosen_t
                row['p'] = chosen_p
                row['selected_gzu_for_tp'] = chosen_gzu
                row['tp_pick_rule'] = 'first_kust_with_both_t_p'
            else:
                row['selected_gzu_for_tp'] = ''
                row['tp_pick_rule'] = 'record_level_fallback'

            daily_rows.append(row)

    nodes_df = pd.DataFrame(node_rows).drop_duplicates('id').sort_values('id').reset_index(drop=True)
    edges_df = pd.DataFrame(edge_rows).drop_duplicates(['source', 'target', 'main_id', 'via_node']).reset_index(drop=True)
    daily_df = pd.DataFrame(daily_rows)

    if not daily_df.empty:
        daily_df['id'] = daily_df['id'].map(normalize_id)
        daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce').dt.normalize()
        daily_df = daily_df[daily_df['date'].notna()].copy()
        daily_df = daily_df.sort_values(['id', 'date']).reset_index(drop=True)

    nodes_csv = csv_dir / 'graph_nodes_129.csv'
    edges_csv = csv_dir / 'graph_edges_129.csv'
    daily_csv = csv_dir / 'pipe_daily_requested_129.csv'

    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)
    daily_df.to_csv(daily_csv, index=False)

    return nodes_csv, edges_csv, daily_csv


def calculate_pvt_for_segment(
    pressure_mpa: float,
    temperature_c: float,
    q_oil_tsd: float,
    q_wat_m3d: float,
    q_gas_tsd_m3d: float,
    rho_oil_kgm3: float,
    rho_wat_kgm3: float,
    rho_gas_kgm3: float,
    watercut: float,
    gamma_oil: float = 0.86,
    gamma_gas: float = 0.7,
    gamma_wat: float = 1.0,
    rp: float | None = None,
) -> dict[str, float]:
    if rp is None:
        q_oil_m3d = q_oil_tsd * 1000 / rho_oil_kgm3
        q_gas_m3d = q_gas_tsd_m3d * 1000
        rp = q_gas_m3d / q_oil_m3d if q_oil_m3d > 0 else 0.0

    p_pa = pressure_mpa * 1e6
    t_k = temperature_c + 273.15

    q_oil_m3s = (q_oil_tsd * 1000 / rho_oil_kgm3) / 86400
    q_wat_m3s = q_wat_m3d / 86400
    q_gas_m3s = (q_gas_tsd_m3d * 1000) / 86400
    q_liq_m3s = q_oil_m3s + q_wat_m3s
    q_gas_work_m3s = polytech_q_gas_work_m3s(q_gas_tsd_m3d, pressure_mpa, temperature_c)
    if q_gas_work_m3s is None:
        q_gas_work_m3s = q_gas_m3s
    q_gas_work_tsd_m3d = q_gas_work_m3s * SECONDS_PER_DAY / 1000.0
    wct = watercut

    fluid = FluidFlow(
        q_fluid=q_liq_m3s,
        wct=wct,
        pvt_model_data={
            'black_oil': {
                'gamma_oil': gamma_oil,
                'gamma_gas': gamma_gas,
                'gamma_wat': gamma_wat,
                'rp': rp,
                'fluid_type': 'liquid',
            }
        },
        fluid_type='liquid',
    )
    fluid.calc_flow(p_pa, t_k)
    salinity_ppm = to_float_value(fluid.salinity)
    comprw = calc_water_compressibility_kriel(t_k=t_k, p_pa=p_pa, salinity_ppm=salinity_ppm)

    # PB reporting mode:
    # - rs (default): derive Pb from current Rs via Standing (matches expected pipeline-level behavior)
    # - rp_legacy: keep legacy Pb from FluidFlow (derived from rp branch)
    pb_mode = os.getenv('PB_MODE', 'rs').strip().lower()
    if pb_mode == 'rp_legacy':
        pb_mpa_out = fluid.pb / 1e6
    else:
        rs_for_pb = to_float_value(fluid.rs)
        if rs_for_pb is None:
            pb_mpa_out = fluid.pb / 1e6
        else:
            # Standing correlation (same constants as calc_pb.py / unifloc oil correlations).
            rsb_min = 1.8
            rsb_old = float(rs_for_pb)
            rsb = max(rsb_min, rsb_old)
            yg = 1.2254503 + 0.001638 * t_k - 1.76875 / gamma_oil
            pb_pa = 519666.7519706273 * ((rsb / gamma_gas) ** 0.83) * (10 ** yg)
            if rsb_old < rsb_min:
                pb_pa = (pb_pa - 101325) * rsb_old / rsb_min + 101325
            pb_mpa_out = pb_pa / 1e6

    return {
        'pb': pb_mpa_out,
        'rs': fluid.rs,
        'bo': fluid.bo,
        'muo': fluid.muo,
        'compro': fluid.co,
        'hc_oil': fluid.heat_capacity_oil,
        'st_oil_gas': fluid.stog,
        'bw': fluid.bw,
        'muw': fluid.muw,
        'hc_wat': fluid.heat_capacity_wat,
        'st_wat_gas': fluid.stwg,
        'bg': fluid.bg,
        'mug': fluid.mug,
        'hc_gas': fluid.heat_capacity_gas,
        'rho_oil': fluid.ro,
        'rho_wat': fluid.rw,
        'rho_gas': fluid.rg,
        'rho_mix': fluid.rm,
        'mu_mix': fluid.mum,
        'salinity': salinity_ppm,
        'comprw': comprw,
        'q_oil': fluid.qo,
        'q_gas_unifloc_m3s': fluid.qg,
        'q_gas_unifloc_tsd_m3d': fluid.qg * SECONDS_PER_DAY / 1000.0,
        'q_gas_work_m3s_polytech': q_gas_work_m3s,
        'q_gas_work_tsd_m3d_polytech': q_gas_work_tsd_m3d,
        'q_gas_norm_tsd_m3d': q_gas_tsd_m3d,
        'q_wat': fluid.qw,
        'q_oil_m3s': q_oil_m3s,
        'q_wat_m3s': q_wat_m3s,
        'q_gas_m3s': q_gas_work_m3s,
        'q_gas_norm_m3s': q_gas_m3s,
        'q_liq_m3s': q_liq_m3s,
        'wct': wct,
        'rp': rp,
    }


def calc_water_compressibility_kriel(
    t_k: float | None,
    p_pa: float | None,
    salinity_ppm: float | None,
) -> float | None:
    """
    Water compressibility (Kriel), aligned with Unifloc water correlation.
    Returns 1/Pa. Uses:
    - t in K
    - p in Pa
    - salinity in ppm
    """
    t_val = to_float_value(t_k)
    p_val = to_float_value(p_pa)
    s_val = to_float_value(salinity_ppm)
    if t_val is None or p_val is None or s_val is None:
        return None
    if p_val <= 0:
        return None
    t_f = (t_val - 273.15) * 1.8 + 32.0
    p_psi = p_val * PSI_PER_PA
    denom = 7.033 * p_psi + 0.5415 * s_val - 537.0 * t_f + 403300.0
    if denom <= 0:
        return None
    out = 0.1 * 145.04 / denom
    if not np.isfinite(out) or out <= 0:
        return None
    return float(out)


def convert_to_kg_s(row: pd.Series) -> dict[str, float]:
    sec_per_day = 86400
    oil_kg_s = row['q_oil'] * 1000 / sec_per_day
    water_kg_s = (row['q_liq'] * row['watercut'] / 100 * 1000) / sec_per_day
    q_gas_work_m3s = polytech_q_gas_work_m3s(row['q_gas'], row['p'], row['t'])
    if q_gas_work_m3s is None:
        raise ValueError('Polytech working-condition gas rate is unavailable for heat calculation.')
    gas_kg_s = q_gas_work_m3s * row['rho_gas']
    return {'oil_kg_s': oil_kg_s, 'water_kg_s': water_kg_s, 'gas_kg_s': gas_kg_s}


def calculate_q_heat(row: pd.Series) -> float:
    kg_s = convert_to_kg_s(row)
    cp_oil = 2100
    cp_gas = 2200
    cp_water = 4200
    return kg_s['oil_kg_s'] * cp_oil + kg_s['water_kg_s'] * cp_water + kg_s['gas_kg_s'] * cp_gas


def calculate_temperature_profile(
    group: pd.DataFrame,
    length_m: float,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Calculate T from its actual boundary and enforce the model range 5..90 C."""
    side = clean_text(
        group.get('temperature_boundary_side', pd.Series(['start'])).iloc[0]
    ).lower()
    if side not in {'start', 'end'}:
        side = 'start'
    boundary_values = pd.to_numeric(group['t'], errors='coerce').dropna()
    if boundary_values.empty:
        raise ValueError('Temperature boundary is missing.')
    boundary_temperature = float(boundary_values.iloc[0])
    pressure_values = pd.to_numeric(group['p'], errors='coerce').dropna()
    if pressure_values.empty:
        raise ValueError('Pressure boundary is missing for heat calculation.')

    heat_boundary = group.sort_values('distance').iloc[0].copy()
    heat_boundary['t'] = boundary_temperature
    heat_boundary['p'] = float(pressure_values.iloc[0])
    q_heat = calculate_q_heat(heat_boundary)
    if not np.isfinite(q_heat) or q_heat <= 0:
        raise ValueError(f'Nonpositive heat-capacity flow: {q_heat}')

    model = RoughTemperatureModel(
        length=length_m,
        tid=heat_boundary['D'],
        tir=EPS_M,
        q_heat_in_by_degree=q_heat,
        surrounding_temperature=SURROUNDING_TEMPERATURE_C,
        htc=15,
        t_bound=boundary_temperature,
        direction=-1 if side == 'end' else 1,
    )
    fn = model.run()
    raw = np.array([float(fn(float(distance))) for distance in group['distance']], dtype=float)
    if not np.isfinite(raw).all():
        raise RuntimeError('Temperature profile contains non-finite values.')
    bounded = np.clip(raw, MIN_TEMPERATURE_C, MAX_TEMPERATURE_C)
    return raw, bounded, side


def estimate_reverse_temperature_start(row: pd.Series, length_m: float) -> float | None:
    """Estimate T(0) from a measured T(L); return None when inputs are unusable."""
    boundary_temperature = to_float_value(row.get('t'))
    diameter_m = to_float_value(row.get('D'))
    if boundary_temperature is None or diameter_m is None:
        return None
    if not (MIN_TEMPERATURE_C <= boundary_temperature <= MAX_TEMPERATURE_C):
        return None
    if boundary_temperature <= SURROUNDING_TEMPERATURE_C + 1e-12:
        return None
    try:
        q_heat = calculate_q_heat(row)
    except Exception:
        return None
    inner_diameter = diameter_m - 2.0 * EPS_M
    if not np.isfinite(q_heat) or q_heat <= 0 or inner_diameter <= 0:
        return None
    exponent = np.pi * 15.0 * inner_diameter * float(length_m) / q_heat
    log_delta = np.log(boundary_temperature - SURROUNDING_TEMPERATURE_C) + exponent
    max_log_delta = np.log(MAX_TEMPERATURE_C - SURROUNDING_TEMPERATURE_C)
    if not np.isfinite(log_delta) or log_delta > max_log_delta:
        return float('inf')
    return float(SURROUNDING_TEMPERATURE_C + np.exp(log_delta))


def mark_reverse_temperature_feasibility(df: pd.DataFrame, length_m: float) -> pd.DataFrame:
    """Reject end-boundary profiles that cannot stay inside the 5..90 C model range."""
    out = df.copy()
    feasible: list[bool] = []
    estimated_start: list[float] = []
    for _, row in out.iterrows():
        side = clean_text(row.get('temperature_boundary_side', 'start')).lower()
        if side != 'end':
            feasible.append(True)
            estimated_start.append(np.nan)
            continue
        estimate = estimate_reverse_temperature_start(row, length_m)
        ok = estimate is not None and np.isfinite(estimate) and estimate <= MAX_TEMPERATURE_C + 1e-9
        feasible.append(bool(ok))
        estimated_start.append(np.nan if estimate is None else float(estimate))
    out['temperature_reverse_feasible'] = feasible
    out['temperature_reverse_estimated_start_c'] = estimated_start
    return out


def add_reynolds_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
    - mul alias from mu_mix
    - re = (rho_oil + rho_wat) * v_mix * D / mul
    """
    out = df.copy()
    if 'mul' not in out.columns:
        out['mul'] = np.nan
    if 'mu_mix' in out.columns:
        out['mul'] = pd.to_numeric(out['mu_mix'], errors='coerce')

    for c in ['rho_oil', 'rho_wat', 'v_mix', 'D', 'mul']:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors='coerce')

    num = (out['rho_oil'] + out['rho_wat']) * out['v_mix'] * out['D']
    out['re'] = np.nan
    ok = out['mul'].notna() & (out['mul'] > 0)
    out.loc[ok, 're'] = num[ok] / out.loc[ok, 'mul']
    return out


def expand_with_profile(
    df_original: pd.DataFrame,
    length_m: float,
    step_m: int = 10,
    angle_default_deg: float = 0.0,
    stage_label: str = '',
    calc_mode_label: str = 'strict',
    angle_by_distance: dict[float, float] | None = None,
) -> pd.DataFrame:
    """
    Core segment recalculation (reused from original script, adapted for universal CSV pipeline).
    Per-segment angles may come from altitudes.csv (id + segment_start_distance -> slope_deg).
    If angle is absent in altitudes, default 0 deg is used.
    """
    segments = build_segment_distances(length_m, step_m)

    rows: list[dict[str, Any]] = []
    for _, row in df_original.iterrows():
        for dist in segments:
            dist_key = normalize_distance_key(dist)
            has_alt_angle = bool(angle_by_distance is not None and dist_key in angle_by_distance)
            row_angle_deg = float(angle_by_distance[dist_key]) if has_alt_angle else float(angle_default_deg)
            new_row = {
                'id': row['id'],
                'date': row['date'],
                'distance': dist,
                'angle_deg': row_angle_deg,
                'angle_from_altitudes': has_alt_angle,
            }
            for col in row.index:
                if col not in {'id', 'date', 'distance', 'angle_deg', 'angle_from_altitudes'}:
                    new_row[col] = row[col]
            rows.append(new_row)

    df_expanded = pd.DataFrame(rows)
    df_expanded = df_expanded.drop_duplicates(subset=['id', 'date', 'distance'])

    results: list[pd.DataFrame] = []

    pvt_output_cols = ['pb', 'rs', 'bo', 'compro', 'hc_oil', 'st_oil_gas', 'bw', 'hc_wat', 'st_wat_gas', 'bg', 'hc_gas', 'rp', 'salinity', 'comprw']
    pvt_all_cols = [
        'muo', 'muw', 'mug', 'rho_oil', 'rho_wat', 'rho_gas', 'rho_mix', 'mu_mix', 'salinity', 'comprw',
        'pb', 'rs', 'bo', 'compro', 'hc_oil', 'st_oil_gas', 'bw', 'hc_wat', 'st_wat_gas', 'bg', 'hc_gas', 'rp',
        'q_oil_m3s', 'q_wat_m3s', 'q_liq_m3s',
        'q_gas_work_m3s_polytech', 'q_gas_work_tsd_m3d_polytech', 'q_gas_norm_tsd_m3d',
        'q_gas_unifloc_m3s', 'q_gas_unifloc_tsd_m3d', 'q_gas_norm_m3s',
    ]

    for (pipe_id, date), group in df_expanded.groupby(['id', 'date']):
        group = group.sort_values('distance').copy()
        boundary_side = clean_text(group.get('pressure_boundary_side', pd.Series(['start'])).iloc[0]).lower()
        if boundary_side not in {'start', 'end'}:
            boundary_side = 'start'
        flow_direction_coef = -1 if boundary_side == 'end' else 1
        pressure_values = pd.to_numeric(group['p'], errors='coerce').dropna()
        boundary_pressure_mpa = np.nan if pressure_values.empty else float(pressure_values.iloc[0])
        pressure_order = list(group.index if boundary_side == 'start' else group.index[::-1])
        raw_temperature, bounded_temperature, temperature_side = calculate_temperature_profile(group, length_m)
        group['p'] = np.nan
        group['pressure_guard_failed'] = False
        group['pressure_min_mpa_config'] = float(MIN_PRESSURE_MPA)
        group['pressure_boundary_side'] = boundary_side
        group['flow_direction_coef'] = int(flow_direction_coef)
        if pressure_order and pd.notna(boundary_pressure_mpa):
            group.at[pressure_order[0], 'p'] = boundary_pressure_mpa
            if boundary_pressure_mpa <= MIN_PRESSURE_MPA:
                group['pressure_guard_failed'] = True
        for col in pvt_output_cols:
            if col not in group.columns:
                group[col] = np.nan
        # First segment uses rp calculated from input rates.
        # All next segments reuse rp returned by FluidFlow on previous segment.
        current_rp: float | None = None

        group['temperature_boundary_side'] = temperature_side
        group['temperature_direction_coef'] = -1 if temperature_side == 'end' else 1
        group['t_raw_before_bound'] = raw_temperature
        group['temperature_floor_applied'] = raw_temperature < MIN_TEMPERATURE_C
        group['temperature_ceiling_applied'] = raw_temperature > MAX_TEMPERATURE_C
        group['t'] = bounded_temperature
        group['t_kelvin'] = group['t'] + 273.15
        group['eps'] = EPS_M

        for col in ['holdup', 'dp_dl_grav', 'dp_dl_fric', 'regime_desc']:
            if col not in group.columns:
                group[col] = np.nan

        for i in range(len(pressure_order) - 1):
            if bool(group['pressure_guard_failed'].iloc[0]):
                break
            current_idx = pressure_order[i]
            next_idx = pressure_order[i + 1]
            current_row = group.loc[current_idx]
            segment_length_m = abs(float(group.at[next_idx, 'distance']) - float(current_row['distance']))

            current_pressure = current_row.get('p', np.nan)
            if pd.isna(current_pressure):
                break

            # Legacy angle mode from main_pipeline_final.py: angle passed to PE2 is (90 - angle_deg).
            angle_deg = 90.0 - current_row.get('angle_deg', angle_default_deg)
            if pd.isna(angle_deg):
                angle_deg = 90.0

            required = ['q_oil', 'q_liq', 'q_gas', 'rho_oil', 'rho_wat', 'rho_gas', 'watercut', 'D']
            missing = [x for x in required if pd.isna(current_row.get(x))]
            if missing:
                break

            try:
                pvt_params = calculate_pvt_for_segment(
                    pressure_mpa=current_pressure,
                    temperature_c=current_row['t'],
                    q_oil_tsd=current_row['q_oil'],
                    q_wat_m3d=current_row['q_liq'] * current_row['watercut'] / 100,
                    q_gas_tsd_m3d=current_row['q_gas'],
                    rho_oil_kgm3=current_row['rho_oil'],
                    rho_wat_kgm3=current_row['rho_wat'],
                    rho_gas_kgm3=current_row['rho_gas'],
                    watercut=current_row['watercut'] / 100,
                    rp=current_rp,
                )
            except Exception:
                break

            for key in pvt_all_cols:
                if key in pvt_params:
                    group.at[current_idx, key] = pvt_params[key]
            current_rp = to_float_value(pvt_params.get('rp'))

            pressure_bar = current_pressure * 10
            q_oil_pe2 = current_row['q_oil'] * 1000 / current_row['rho_oil']
            q_wat_pe2 = current_row['q_liq'] * current_row['watercut'] / 100
            q_gas_pe2 = pvt_params.get('q_gas_work_tsd_m3d_polytech')
            if q_gas_pe2 is None or pd.isna(q_gas_pe2):
                q_gas_pe2 = current_row['q_gas']

            try:
                input_payload = {
                    'input_q_oil': q_oil_pe2,
                    'input_q_wat': q_wat_pe2,
                    'input_q_gas': q_gas_pe2,
                    'input_rhoo': pvt_params['rho_oil'],
                    'input_rhow': pvt_params['rho_wat'],
                    'input_rhog': pvt_params['rho_gas'],
                    'input_wct': current_row['watercut'] / 100,
                    'input_muo': pvt_params['muo'],
                    'input_muw': pvt_params['muw'],
                    'input_mug': pvt_params['mug'],
                    'input_gl_ift': 0.5,
                    'input_diameter': current_row['D'],
                    'input_eps': EPS_M / current_row['D'],
                    'input_angle': angle_deg,
                    'input_flow_direction_coef': int(flow_direction_coef),
                    'input_p1': 1,
                    'input_p2': 1,
                }

                result = pe_2_correlation(
                    q_oil=q_oil_pe2,
                    q_wat=q_wat_pe2,
                    q_gas=q_gas_pe2,
                    rhoo=pvt_params['rho_oil'],
                    rhow=pvt_params['rho_wat'],
                    rhog=pvt_params['rho_gas'],
                    wct=current_row['watercut'] / 100,
                    muo=pvt_params['muo'],
                    muw=pvt_params['muw'],
                    mug=pvt_params['mug'],
                    gl_ift=0.5,
                    diameter=current_row['D'],
                    eps=EPS_M / current_row['D'],
                    angle=angle_deg,
                    pressure=pressure_bar,
                    p1=1,
                    p2=1,
                )
                holdup, grav_grad, fric_grad, acc_grad, total_grad, regime, regime_desc = result
                group.at[current_idx, 'holdup'] = holdup
                group.at[current_idx, 'dp_dl_grav'] = grav_grad / 10
                group.at[current_idx, 'dp_dl_fric'] = fric_grad / 10
                regime_desc_text = ''
                if regime_desc:
                    regime_desc_text = regime_desc.decode() if isinstance(regime_desc, (bytes, bytearray)) else str(regime_desc)
                    group.at[current_idx, 'regime_desc'] = regime_desc_text

                next_pressure_bar = pressure_bar - (total_grad * segment_length_m * flow_direction_coef)
                group.at[next_idx, 'p'] = next_pressure_bar / 10
                append_pe2_trace(
                    {
                        'calc_mode': calc_mode_label,
                        'stage': stage_label,
                        'source_fn': 'expand_with_profile',
                        'pipe_id': pipe_id,
                        'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                        'distance_m': float(current_row.get('distance', np.nan)),
                        'segment_step_m': float(segment_length_m),
                        'selected_gzu_for_tp': clean_text(current_row.get('selected_gzu_for_tp')),
                        'tp_pick_rule': clean_text(current_row.get('tp_pick_rule')),
                        'temperature_c': to_float_value(current_row.get('t')),
                        'pressure_mpa_in': to_float_value(current_pressure),
                        'pressure_bar_in': to_float_value(pressure_bar),
                        **input_payload,
                        'output_holdup': to_float_value(holdup),
                        'output_gravity_gradient': to_float_value(grav_grad),
                        'output_friction_gradient': to_float_value(fric_grad),
                        'output_acceleration_gradient': to_float_value(acc_grad),
                        'output_total_gradient': to_float_value(total_grad),
                        'output_regime': regime,
                        'output_regime_desc': regime_desc_text,
                        'output_next_pressure_bar': to_float_value(next_pressure_bar),
                        'output_next_pressure_mpa': to_float_value(next_pressure_bar / 10),
                        'error': '',
                    }
                )
                if next_pressure_bar / 10.0 <= MIN_PRESSURE_MPA:
                    group['pressure_guard_failed'] = True
                    break
            except Exception as exc:
                append_pe2_trace(
                    {
                        'calc_mode': calc_mode_label,
                        'stage': stage_label,
                        'source_fn': 'expand_with_profile',
                        'pipe_id': pipe_id,
                        'date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                        'distance_m': float(current_row.get('distance', np.nan)),
                        'segment_step_m': float(segment_length_m),
                        'selected_gzu_for_tp': clean_text(current_row.get('selected_gzu_for_tp')),
                        'tp_pick_rule': clean_text(current_row.get('tp_pick_rule')),
                        'temperature_c': to_float_value(current_row.get('t')),
                        'pressure_mpa_in': to_float_value(current_pressure),
                        'pressure_bar_in': to_float_value(pressure_bar),
                        **{
                            'input_q_oil': to_float_value(q_oil_pe2),
                            'input_q_wat': to_float_value(q_wat_pe2),
                            'input_q_gas': to_float_value(q_gas_pe2),
                            'input_rhoo': to_float_value(pvt_params.get('rho_oil')),
                            'input_rhow': to_float_value(pvt_params.get('rho_wat')),
                            'input_rhog': to_float_value(pvt_params.get('rho_gas')),
                            'input_wct': to_float_value(current_row.get('watercut') / 100),
                            'input_muo': to_float_value(pvt_params.get('muo')),
                            'input_muw': to_float_value(pvt_params.get('muw')),
                            'input_mug': to_float_value(pvt_params.get('mug')),
                            'input_gl_ift': 0.5,
                            'input_diameter': to_float_value(current_row.get('D')),
                            'input_eps': to_float_value(EPS_M / current_row.get('D')),
                            'input_angle': to_float_value(angle_deg),
                            'input_flow_direction_coef': int(flow_direction_coef),
                            'input_p1': 1,
                            'input_p2': 1,
                        },
                        'error': f'{type(exc).__name__}: {exc}',
                    }
                )
                break

        # The last profile point has no next PE2 step, so legacy code can leave
        # PVT-derived fields empty there. Polytech Re/KNNS need those fields.
        for col in pvt_all_cols:
            if col in group.columns:
                group[col] = group[col].ffill().bfill()
        group = add_polytech_segment_flow_columns(group)
        group = add_reynolds_columns(group)
        results.append(group.sort_values('distance'))

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def expand_with_profile_fast(
    df_original: pd.DataFrame,
    length_m: float,
    step_m: int = 10,
    angle_default_deg: float = 0.0,
    stage_label: str = '',
    calc_mode_label: str = 'fast',
    angle_by_distance: dict[float, float] | None = None,
) -> pd.DataFrame:
    """
    Fast mode:
    - temperature profile is computed per (id, date) across all segments,
    - PVT/PE2 are evaluated once at segment start and applied to the whole profile.
    This preserves pipeline structure and 10 m segmentation while avoiding
    expensive per-segment iterative PVT updates.
    """
    segments = build_segment_distances(length_m, step_m)
    if len(segments) == 0:
        return pd.DataFrame()

    out_frames: list[pd.DataFrame] = []
    pvt_output_cols = ['pb', 'rs', 'bo', 'compro', 'hc_oil', 'st_oil_gas', 'bw', 'hc_wat', 'st_wat_gas', 'bg', 'hc_gas', 'rp', 'salinity', 'comprw']
    pvt_all_cols = [
        'muo', 'muw', 'mug', 'rho_oil', 'rho_wat', 'rho_gas', 'rho_mix', 'mu_mix', 'salinity', 'comprw',
        'pb', 'rs', 'bo', 'compro', 'hc_oil', 'st_oil_gas', 'bw', 'hc_wat', 'st_wat_gas', 'bg', 'hc_gas', 'rp',
        'q_oil_m3s', 'q_wat_m3s', 'q_liq_m3s',
        'q_gas_work_m3s_polytech', 'q_gas_work_tsd_m3d_polytech', 'q_gas_norm_tsd_m3d',
        'q_gas_unifloc_m3s', 'q_gas_unifloc_tsd_m3d', 'q_gas_norm_m3s',
    ]

    for _, row in df_original.iterrows():
        boundary_side = clean_text(row.get('pressure_boundary_side', 'start')).lower()
        if boundary_side not in {'start', 'end'}:
            boundary_side = 'start'
        flow_direction_coef = -1 if boundary_side == 'end' else 1
        q_heat = calculate_q_heat(row)

        temp_model = RoughTemperatureModel(
            length=length_m,
            tid=row['D'],
            tir=EPS_M,
            q_heat_in_by_degree=q_heat,
            surrounding_temperature=SURROUNDING_TEMPERATURE_C,
            htc=15,
            t_bound=row['t'],
            direction=get_rough_direction(),
        )
        temp_func = temp_model.run()
        temp_profile = np.array([float(temp_func(float(d))) for d in segments], dtype=float)

        raw_angle_deg = float(angle_default_deg)
        angle_from_altitudes = False
        key0 = normalize_distance_key(0.0)
        if angle_by_distance is not None and key0 in angle_by_distance:
            raw_angle_deg = float(angle_by_distance[key0])
            angle_from_altitudes = True
        elif pd.notna(row.get('angle_deg')):
            raw_angle_deg = float(row.get('angle_deg'))

        # Legacy angle mode from main_pipeline_final.py: angle passed to PE2 is (90 - angle_deg).
        angle_deg = 90.0 - raw_angle_deg
        if pd.isna(angle_deg):
            angle_deg = 90.0

        try:
            pvt_params = calculate_pvt_for_segment(
                pressure_mpa=float(row['p']),
                temperature_c=float(temp_profile[-1] if boundary_side == 'end' else temp_profile[0]),
                q_oil_tsd=float(row['q_oil']),
                q_wat_m3d=float(row['q_liq']) * float(row['watercut']) / 100.0,
                q_gas_tsd_m3d=float(row['q_gas']),
                rho_oil_kgm3=float(row['rho_oil']),
                rho_wat_kgm3=float(row['rho_wat']),
                rho_gas_kgm3=float(row['rho_gas']),
                watercut=float(row['watercut']) / 100.0,
                rp=None,
            )

            pressure_bar = float(row['p']) * 10.0
            q_oil_pe2 = float(row['q_oil']) * 1000.0 / float(row['rho_oil'])
            q_wat_pe2 = float(row['q_liq']) * float(row['watercut']) / 100.0
            q_gas_pe2 = pvt_params.get('q_gas_work_tsd_m3d_polytech')
            if q_gas_pe2 is None or pd.isna(q_gas_pe2):
                q_gas_pe2 = float(row['q_gas'])

            input_payload = {
                'input_q_oil': q_oil_pe2,
                'input_q_wat': q_wat_pe2,
                'input_q_gas': q_gas_pe2,
                'input_rhoo': float(pvt_params['rho_oil']),
                'input_rhow': float(pvt_params['rho_wat']),
                'input_rhog': float(pvt_params['rho_gas']),
                'input_wct': float(row['watercut']) / 100.0,
                'input_muo': float(pvt_params['muo']),
                'input_muw': float(pvt_params['muw']),
                'input_mug': float(pvt_params['mug']),
                'input_gl_ift': 0.5,
                'input_diameter': float(row['D']),
                'input_eps': EPS_M / float(row['D']),
                'input_angle': float(angle_deg),
                'input_flow_direction_coef': int(flow_direction_coef),
                'input_p1': 1,
                'input_p2': 1,
            }

            result = pe_2_correlation(
                q_oil=q_oil_pe2,
                q_wat=q_wat_pe2,
                q_gas=q_gas_pe2,
                rhoo=float(pvt_params['rho_oil']),
                rhow=float(pvt_params['rho_wat']),
                rhog=float(pvt_params['rho_gas']),
                wct=float(row['watercut']) / 100.0,
                muo=float(pvt_params['muo']),
                muw=float(pvt_params['muw']),
                mug=float(pvt_params['mug']),
                gl_ift=0.5,
                diameter=float(row['D']),
                eps=EPS_M / float(row['D']),
                angle=float(angle_deg),
                pressure=pressure_bar,
                p1=1,
                p2=1,
            )
            holdup, grav_grad, fric_grad, acc_grad, total_grad, regime, regime_desc = result
            regime_desc_text = regime_desc.decode() if isinstance(regime_desc, (bytes, bytearray)) else str(regime_desc)
            first_step_m = abs(float(segments[1] - segments[0])) if len(segments) > 1 else 0.0
            next_pressure_bar = pressure_bar - float(total_grad) * first_step_m * flow_direction_coef
            append_pe2_trace(
                {
                    'calc_mode': calc_mode_label,
                    'stage': stage_label,
                    'source_fn': 'expand_with_profile_fast',
                    'pipe_id': row['id'],
                    'date': pd.Timestamp(row['date']).strftime('%Y-%m-%d'),
                    'distance_m': 0.0,
                    'segment_step_m': float(step_m),
                    'selected_gzu_for_tp': clean_text(row.get('selected_gzu_for_tp')),
                    'tp_pick_rule': clean_text(row.get('tp_pick_rule')),
                    'temperature_c': to_float_value(temp_profile[0]),
                    'pressure_mpa_in': to_float_value(row['p']),
                    'pressure_bar_in': to_float_value(pressure_bar),
                    **input_payload,
                    'output_holdup': to_float_value(holdup),
                    'output_gravity_gradient': to_float_value(grav_grad),
                    'output_friction_gradient': to_float_value(fric_grad),
                    'output_acceleration_gradient': to_float_value(acc_grad),
                    'output_total_gradient': to_float_value(total_grad),
                    'output_regime': regime,
                    'output_regime_desc': regime_desc_text,
                    'output_next_pressure_bar': to_float_value(next_pressure_bar),
                    'output_next_pressure_mpa': to_float_value(next_pressure_bar / 10.0),
                    'error': '',
                }
            )
        except Exception as exc:
            append_pe2_trace(
                {
                    'calc_mode': calc_mode_label,
                    'stage': stage_label,
                    'source_fn': 'expand_with_profile_fast',
                    'pipe_id': row.get('id'),
                    'date': pd.Timestamp(row['date']).strftime('%Y-%m-%d') if pd.notna(row.get('date')) else '',
                    'distance_m': 0.0,
                    'segment_step_m': float(step_m),
                    'selected_gzu_for_tp': clean_text(row.get('selected_gzu_for_tp')),
                    'tp_pick_rule': clean_text(row.get('tp_pick_rule')),
                    'temperature_c': to_float_value(temp_profile[0]) if len(temp_profile) > 0 else None,
                    'pressure_mpa_in': to_float_value(row.get('p')),
                    'pressure_bar_in': to_float_value((to_float_value(row.get('p')) or 0.0) * 10.0),
                    'error': f'{type(exc).__name__}: {exc}',
                }
            )
            continue

        if boundary_side == 'end':
            p_bar_profile = pressure_bar + float(total_grad) * (float(length_m) - segments)
        else:
            p_bar_profile = pressure_bar - float(total_grad) * segments
        p_mpa_profile = p_bar_profile / 10.0

        base_data: dict[str, Any] = {}
        for col in df_original.columns:
            if col in {'id', 'date', 'distance'}:
                continue
            base_data[col] = row[col]
        base_data['angle_deg'] = float(raw_angle_deg)
        base_data['flow_direction_coef'] = int(flow_direction_coef)
        base_data['angle_from_altitudes'] = bool(angle_from_altitudes)
        base_data['eps'] = EPS_M
        base_data['t_kelvin'] = np.nan
        base_data['holdup'] = float(holdup)
        base_data['dp_dl_grav'] = float(grav_grad) / 10.0
        base_data['dp_dl_fric'] = float(fric_grad) / 10.0
        base_data['regime_desc'] = regime_desc_text
        for key in pvt_all_cols:
            if key in pvt_params:
                base_data[key] = pvt_params[key]
        for col in pvt_output_cols:
            if col not in base_data:
                base_data[col] = np.nan

        frame = pd.DataFrame(
            {
                'id': [row['id']] * len(segments),
                'date': [row['date']] * len(segments),
                'distance': segments,
                't': temp_profile,
                'p': p_mpa_profile,
            }
        )
        for k, v in base_data.items():
            frame[k] = v
        frame['t_kelvin'] = frame['t'] + 273.15
        frame['pressure_boundary_side'] = boundary_side
        frame['pressure_guard_failed'] = bool((p_mpa_profile <= MIN_PRESSURE_MPA).any())
        frame['pressure_min_mpa_config'] = float(MIN_PRESSURE_MPA)
        frame = add_polytech_segment_flow_columns(frame)
        frame = add_reynolds_columns(frame)
        out_frames.append(frame)

    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


def build_adjacency(edges_df: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    children: dict[str, list[str]] = defaultdict(list)
    parents: dict[str, list[str]] = defaultdict(list)
    for r in edges_df.itertuples(index=False):
        s = normalize_id(r.source)
        t = normalize_id(r.target)
        if not s or not t:
            continue
        children[s].append(t)
        parents[t].append(s)
    for k in list(children.keys()):
        children[k] = sorted(set(children[k]))
    for k in list(parents.keys()):
        parents[k] = sorted(set(parents[k]))
    return dict(children), dict(parents)


def choose_length_m(pipe_df: pd.DataFrame, node_row: pd.Series | None) -> float | None:
    length_vals = pd.to_numeric(pipe_df.get('L', pd.Series(dtype=float)), errors='coerce')
    length_vals = length_vals[length_vals > 0]
    if len(length_vals) > 0:
        return float(length_vals.median())
    if node_row is not None and pd.notna(node_row.get('L_m_graph')):
        lv = float(node_row['L_m_graph'])
        return lv if lv > 0 else None
    return None


def prepare_pipe_df(pipe_id: str, daily_df: pd.DataFrame, node_row: pd.Series | None) -> pd.DataFrame:
    df = daily_df[daily_df['id'] == pipe_id].copy()
    if df.empty:
        return df

    # Rename to internal calc fields.
    for src_col, dst_col in FIELD_MAP.items():
        if src_col in df.columns and dst_col not in df.columns:
            df = df.rename(columns={src_col: dst_col})

    # Ensure key cols exist.
    for col in ['id', 'date'] + list(FIELD_MAP.values()):
        if col not in df.columns:
            df[col] = np.nan

    df['id'] = df['id'].map(normalize_id)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df = df[df['date'].notna()].copy()

    # Numeric conversion.
    num_cols = ['q_liq', 'q_oil', 'q_gas', 'rho_oil', 'rho_wat', 'rho_gas', 'watercut', 't', 'p', 'D', 'L', 'S', 'flow_direction_coef']
    for c in num_cols:
        if c in df.columns:
            df[c] = to_float(df[c])

    # Diameter: read from column, convert mm -> m if looks like mm.
    if 'D' in df.columns:
        d_mask = df['D'].notna() & (df['D'] > 1.5)
        df.loc[d_mask, 'D'] = df.loc[d_mask, 'D'] / 1000.0

    # Fallback from node metadata if D is missing.
    if 'D' in df.columns and df['D'].isna().all() and node_row is not None:
        d_mm = node_row.get('D_mm_graph')
        if pd.notna(d_mm):
            df['D'] = float(d_mm) / 1000.0

    # Keep input angle at 0.0 by default. At PE2 call site, legacy conversion (90 - angle_deg) is applied.
    df['angle_deg'] = 0.0
    if 'flow_direction_coef' not in df.columns:
        df['flow_direction_coef'] = 1.0
    df['flow_direction_coef'] = to_float(df['flow_direction_coef']).fillna(1.0)
    df.loc[~df['flow_direction_coef'].isin([-1.0, 1.0]), 'flow_direction_coef'] = 1.0
    if 'q_gas_fallback_naporny' not in df.columns:
        df['q_gas_fallback_naporny'] = False

    # Business rule (approved): for naporny pipes q_gas is optional in mask.
    # We replace empty/nonpositive q_gas with tiny positive fallback.
    pipe_kind_col = 'dns_paot_pipe_kind'
    if pipe_kind_col in df.columns:
        pipe_kind_norm = df[pipe_kind_col].astype(str).str.strip().str.lower()
        naporny_mask = pipe_kind_norm.eq('naporny')
        qg_bad = df['q_gas'].isna() | (df['q_gas'] <= 0)
        qg_replace = naporny_mask & qg_bad
        if qg_replace.any():
            df.loc[qg_replace, 'q_gas'] = float(NAPORNY_Q_GAS_FALLBACK_TSD_M3D)
            df.loc[qg_replace, 'q_gas_fallback_naporny'] = True

    return assign_temperature_boundaries(assign_pressure_boundaries(df))


def apply_valid_mask(df: pd.DataFrame) -> pd.DataFrame:
    pipe_kind_col = 'dns_paot_pipe_kind'
    if pipe_kind_col in df.columns:
        pipe_kind_norm = df[pipe_kind_col].astype(str).str.strip().str.lower()
        naporny_mask = pipe_kind_norm.eq('naporny')
    else:
        naporny_mask = pd.Series(False, index=df.index)

    # For naporny q_gas is not a blocking mask criterion.
    q_gas_ok = naporny_mask | (df['q_gas'].notna() & (df['q_gas'] > 0))
    temperature_feasible = df.get(
        'temperature_reverse_feasible', pd.Series(True, index=df.index)
    ).fillna(False).astype(bool)
    mask = (
        df['q_liq'].notna() & (df['q_liq'] > 0) &
        df['q_oil'].notna() & (df['q_oil'] > 0) &
        q_gas_ok &
        df['rho_oil'].notna() & (df['rho_oil'] > 0) &
        df['rho_wat'].notna() & (df['rho_wat'] > 0) &
        df['rho_gas'].notna() & (df['rho_gas'] > 0) &
        df['watercut'].notna() &
        df['D'].notna() & (df['D'] > 0) &
        df['t'].notna() &
        temperature_feasible &
        df['p'].notna() & (df['p'] > MIN_PRESSURE_MPA)
    )
    return df.loc[mask].copy()


def extract_end_map(calc_df: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if calc_df.empty:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    for date, grp in calc_df.groupby('date'):
        last = grp.sort_values('distance').iloc[-1]
        out[pd.Timestamp(date).strftime('%Y-%m-%d')] = {
            't_end': None if pd.isna(last.get('t')) else float(last['t']),
            'p_end': None if pd.isna(last.get('p')) else float(last['p']),
        }
    return out


def collect_zero_events(calc_df: pd.DataFrame, pipe_id: str, stage: str) -> pd.DataFrame:
    if calc_df.empty:
        return pd.DataFrame(columns=['id', 'date', 'distance', 'metric', 'value', 'stage'])

    rows = []
    for metric in ['t', 'p']:
        if metric not in calc_df.columns:
            continue
        threshold = MIN_PRESSURE_MPA if metric == 'p' else 0.0
        bad = calc_df[calc_df[metric].notna() & (calc_df[metric] <= threshold)].copy()
        for r in bad.itertuples(index=False):
            rows.append(
                {
                    'id': pipe_id,
                    'date': pd.Timestamp(getattr(r, 'date')).strftime('%Y-%m-%d'),
                    'distance': float(getattr(r, 'distance')),
                    'metric': metric,
                    'value': float(getattr(r, metric)),
                    'stage': stage,
                }
            )
    return pd.DataFrame(rows)


def drop_dates_with_nonpositive_pressure(calc_df: pd.DataFrame, pipe_id: str, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Business rule:
    If at least one segment for (pipe_id, date) has p <= MIN_PRESSURE_MPA,
    drop ALL segments of this pipe/date from result.
    """
    if calc_df.empty or 'p' not in calc_df.columns:
        return calc_df, pd.DataFrame(columns=['id', 'date', 'distance', 'metric', 'value', 'stage'])

    p_num = pd.to_numeric(calc_df['p'], errors='coerce')
    guard = calc_df.get('pressure_guard_failed', pd.Series(False, index=calc_df.index)).fillna(False).astype(bool)
    bad = calc_df[(p_num.notna() & (p_num <= MIN_PRESSURE_MPA)) | guard].copy()
    if bad.empty:
        return calc_df, pd.DataFrame(columns=['id', 'date', 'distance', 'metric', 'value', 'stage'])

    bad_dates = sorted(pd.to_datetime(bad['date'], errors='coerce').dropna().dt.normalize().unique())
    if not bad_dates:
        return calc_df, pd.DataFrame(columns=['id', 'date', 'distance', 'metric', 'value', 'stage'])

    bad_dates_idx = pd.DatetimeIndex(bad_dates)
    filtered = calc_df[
        ~pd.to_datetime(calc_df['date'], errors='coerce').dt.normalize().isin(bad_dates_idx)
    ].copy()

    dropped_rows = [
        {
            'id': pipe_id,
            'date': pd.Timestamp(d).strftime('%Y-%m-%d'),
            'distance': np.nan,
            'metric': 'p_day_dropped_below_min',
            'value': float(MIN_PRESSURE_MPA),
            'stage': stage,
        }
        for d in bad_dates
    ]
    return filtered, pd.DataFrame(dropped_rows)


def fill_tp_from_predecessors(pipe_df: pd.DataFrame, pred_ids: list[str], end_maps: dict[str, dict[str, dict[str, float | None]]]) -> pd.DataFrame:
    if pipe_df.empty or not pred_ids:
        pipe_df['t_filled_from_prev'] = False
        pipe_df['p_filled_from_prev'] = False
        pipe_df['fill_sources_prev'] = ''
        return pipe_df

    df = pipe_df.copy()
    df['date_key'] = df['date'].dt.strftime('%Y-%m-%d')
    df['t_filled_from_prev'] = False
    df['p_filled_from_prev'] = False
    df['fill_sources_prev'] = ''

    for idx, row in df.iterrows():
        d = row['date_key']
        t_candidates: list[float] = []
        p_candidates: list[float] = []
        src_used: list[str] = []

        for pid in pred_ids:
            end_map = end_maps.get(pid, {})
            if d not in end_map:
                continue
            end_vals = end_map[d]
            t_end = end_vals.get('t_end')
            p_end = end_vals.get('p_end')
            if t_end is not None:
                t_candidates.append(float(t_end))
                src_used.append(pid)
            if p_end is not None:
                p_candidates.append(float(p_end))
                if pid not in src_used:
                    src_used.append(pid)

        if pd.isna(row['t']) and t_candidates:
            df.at[idx, 't'] = float(np.mean(t_candidates))
            df.at[idx, 't_filled_from_prev'] = True

        if (pd.isna(row['p']) or row['p'] <= 0) and p_candidates:
            df.at[idx, 'p'] = float(np.mean(p_candidates))
            df.at[idx, 'p_filled_from_prev'] = True

        if src_used:
            df.at[idx, 'fill_sources_prev'] = ';'.join(sorted(set(src_used)))

    df = df.drop(columns=['date_key'])
    return df


def run_one_pipe(
    pipe_id: str,
    stage: str,
    daily_df: pd.DataFrame,
    nodes_idx: dict[str, pd.Series],
    step_m: int,
    calc_mode: str,
    end_maps: dict[str, dict[str, dict[str, float | None]]],
    pred_ids_for_fill: list[str] | None = None,
    angle_by_distance: dict[float, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    node_row = nodes_idx.get(pipe_id)
    pipe_df = prepare_pipe_df(pipe_id, daily_df, node_row)

    if pipe_df.empty:
        stats = {'id': pipe_id, 'stage': stage, 'status': 'no_rows', 'rows_input': 0}
        return pd.DataFrame(), pd.DataFrame(), stats, pd.DataFrame()

    missing_tp = pipe_df[pipe_df['t'].isna() | pipe_df['p'].isna()][['id', 'date', 't', 'p']].copy()
    missing_tp['reason'] = 'missing_t_or_p_before_calc'
    missing_tp['stage'] = stage

    # Keep output schema stable for all pipes:
    # always create predecessor-fill columns even when there are no predecessors.
    pipe_df = fill_tp_from_predecessors(pipe_df, pred_ids_for_fill or [], end_maps)

    length_m = choose_length_m(pipe_df, node_row)
    if length_m is None or length_m <= 0:
        stats = {'id': pipe_id, 'stage': stage, 'status': 'no_length', 'rows_input': int(len(pipe_df))}
        return pd.DataFrame(), pd.DataFrame(), stats, missing_tp

    pipe_df = mark_reverse_temperature_feasibility(pipe_df, float(length_m))
    rejected_temperature = pipe_df.loc[
        ~pipe_df['temperature_reverse_feasible'], ['id', 'date', 't', 'p']
    ].copy()
    if not rejected_temperature.empty:
        rejected_temperature['reason'] = 'reverse_temperature_from_end_exceeds_5_90_model_range'
        rejected_temperature['stage'] = stage
        missing_tp = pd.concat([missing_tp, rejected_temperature], ignore_index=True)

    valid_df = apply_valid_mask(pipe_df)
    if valid_df.empty:
        stats = {
            'id': pipe_id,
            'stage': stage,
            'status': 'no_valid_rows_after_mask',
            'rows_input': int(len(pipe_df)),
            'rows_valid': 0,
            'rows_rejected_infeasible_reverse_temperature': int((~pipe_df['temperature_reverse_feasible']).sum()),
            'length_m': float(length_m),
            'q_gas_fallback_naporny_rows_input': int(pipe_df.get('q_gas_fallback_naporny', pd.Series(dtype=bool)).fillna(False).sum()) if 'q_gas_fallback_naporny' in pipe_df.columns else 0,
        }
        return pd.DataFrame(), pd.DataFrame(), stats, missing_tp

    if calc_mode != 'strict':
        raise ValueError('Разрешен только strict режим (без упрощений).')

    calc_df = expand_with_profile(
        valid_df,
        length_m=float(length_m),
        step_m=step_m,
        angle_default_deg=0.0,
        stage_label=stage,
        calc_mode_label=calc_mode,
        angle_by_distance=angle_by_distance,
    )
    if calc_df.empty:
        stats = {
            'id': pipe_id,
            'stage': stage,
            'status': 'calc_empty',
            'rows_input': int(len(pipe_df)),
            'rows_valid': int(len(valid_df)),
            'rows_rejected_infeasible_reverse_temperature': int((~pipe_df['temperature_reverse_feasible']).sum()),
            'length_m': float(length_m),
        }
        return pd.DataFrame(), pd.DataFrame(), stats, missing_tp

    calc_df['stage'] = stage
    calc_df['length_m'] = float(length_m)

    zeros_df = collect_zero_events(calc_df, pipe_id, stage)
    calc_df, dropped_day_df = drop_dates_with_nonpositive_pressure(calc_df, pipe_id, stage)
    if not dropped_day_df.empty:
        zeros_df = pd.concat([zeros_df, dropped_day_df], ignore_index=True)

    if calc_df.empty:
        stats = {
            'id': pipe_id,
            'stage': stage,
            'status': 'all_days_dropped_by_nonpositive_p',
            'rows_input': int(len(pipe_df)),
            'rows_valid': int(len(valid_df)),
            'rows_segments': 0,
            'dates_count': 0,
            'rows_rejected_infeasible_reverse_temperature': int((~pipe_df['temperature_reverse_feasible']).sum()),
            'length_m': float(length_m),
            'zero_events': int(len(zeros_df)),
            'dropped_bad_days': int(len(dropped_day_df)),
            'filled_from_prev_rows_t': 0,
            'filled_from_prev_rows_p': 0,
            'q_gas_fallback_naporny_rows_input': int(pipe_df.get('q_gas_fallback_naporny', pd.Series(dtype=bool)).fillna(False).sum()) if 'q_gas_fallback_naporny' in pipe_df.columns else 0,
            'q_gas_fallback_naporny_rows_valid': int(valid_df.get('q_gas_fallback_naporny', pd.Series(dtype=bool)).fillna(False).sum()) if 'q_gas_fallback_naporny' in valid_df.columns else 0,
        }
        end_maps.pop(pipe_id, None)
        return pd.DataFrame(), zeros_df, stats, missing_tp

    end_map = extract_end_map(calc_df)
    end_maps[pipe_id] = end_map

    stats = {
        'id': pipe_id,
        'stage': stage,
        'status': 'ok',
        'rows_input': int(len(pipe_df)),
        'rows_valid': int(len(valid_df)),
        'rows_rejected_infeasible_reverse_temperature': int((~pipe_df['temperature_reverse_feasible']).sum()),
        'rows_segments': int(len(calc_df)),
        'dates_count': int(calc_df['date'].nunique()),
        'length_m': float(length_m),
        'zero_events': int(len(zeros_df)),
        'dropped_bad_days': int(len(dropped_day_df)),
        'filled_from_prev_rows_t': int(calc_df.get('t_filled_from_prev', pd.Series(dtype=bool)).fillna(False).sum()) if 't_filled_from_prev' in calc_df.columns else 0,
        'filled_from_prev_rows_p': int(calc_df.get('p_filled_from_prev', pd.Series(dtype=bool)).fillna(False).sum()) if 'p_filled_from_prev' in calc_df.columns else 0,
        'q_gas_fallback_naporny_rows_input': int(pipe_df.get('q_gas_fallback_naporny', pd.Series(dtype=bool)).fillna(False).sum()) if 'q_gas_fallback_naporny' in pipe_df.columns else 0,
        'q_gas_fallback_naporny_rows_valid': int(valid_df.get('q_gas_fallback_naporny', pd.Series(dtype=bool)).fillna(False).sum()) if 'q_gas_fallback_naporny' in valid_df.columns else 0,
        'q_gas_fallback_naporny_rows_calc': int(calc_df.get('q_gas_fallback_naporny', pd.Series(dtype=bool)).fillna(False).sum()) if 'q_gas_fallback_naporny' in calc_df.columns else 0,
    }

    return calc_df, zeros_df, stats, missing_tp


def collect_reachable(children: dict[str, list[str]], start_ids: list[str]) -> set[str]:
    q = deque(start_ids)
    seen = set(start_ids)
    while q:
        cur = q.popleft()
        for nxt in children.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def build_processing_order(
    pipe_ids: list[str],
    children: dict[str, list[str]],
    parents: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """
    Build deterministic graph-aware processing order:
    - primary: Kahn topological order (parents before children),
    - fallback: sorted leftover ids (for cycles/disconnected anomalies).
    """
    if not pipe_ids:
        return [], []

    allowed = set(pipe_ids)
    indegree: dict[str, int] = {pid: 0 for pid in pipe_ids}
    for pid in pipe_ids:
        indegree[pid] = int(sum(1 for pr in parents.get(pid, []) if pr in allowed))

    ready = deque(sorted([pid for pid in pipe_ids if indegree.get(pid, 0) == 0]))
    order: list[str] = []
    in_order: set[str] = set()

    while ready:
        cur = ready.popleft()
        if cur in in_order:
            continue
        order.append(cur)
        in_order.add(cur)

        for nxt in children.get(cur, []):
            if nxt not in allowed:
                continue
            indegree[nxt] = indegree.get(nxt, 0) - 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    remaining = sorted([pid for pid in pipe_ids if pid not in in_order])
    if remaining:
        order.extend(remaining)

    return order, remaining


def run_pipeline(
    master_json: Path,
    requested_json: Path,
    out_dir: Path,
    step_m: int = 10,
    max_pipes: int | None = None,
    calc_mode: str = 'strict',
    only_pipe_id: str | None = None,
    pe2_trace_csv: Path | None = None,
    altitudes_csv: Path | None = None,
) -> dict[str, Any]:
    csv_dir = out_dir / 'csv_inputs'
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_csv, edges_csv, daily_csv = export_json_to_csv(master_json, requested_json, csv_dir)

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    daily_df = pd.read_csv(daily_csv, low_memory=False)

    nodes_df['id'] = nodes_df['id'].map(normalize_id)
    edges_df['source'] = edges_df['source'].map(normalize_id)
    edges_df['target'] = edges_df['target'].map(normalize_id)
    daily_df['id'] = daily_df['id'].map(normalize_id)
    daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce').dt.normalize()
    daily_df = daily_df[daily_df['date'].notna()].copy()

    nodes_idx = {r.id: r for _, r in nodes_df.iterrows()}

    all_129_ids = sorted(set(nodes_df['id'].astype(str).tolist()))
    green_set = set(nodes_df[(nodes_df['found_in_graph']) & (nodes_df['eligible_one_kust'])]['id'].astype(str).tolist())
    orange_set = set(nodes_df[(nodes_df['found_in_graph']) & (nodes_df['kust_count_in_sources'] > 1)]['id'].astype(str).tolist())
    children, parents = build_adjacency(edges_df)

    if calc_mode != 'strict':
        raise ValueError('Разрешен только strict режим (без упрощений).')

    altitudes_by_pipe, altitudes_meta = load_altitudes_map(altitudes_csv)

    if only_pipe_id:
        only_pipe_id = normalize_id(only_pipe_id)
        if not only_pipe_id:
            raise ValueError('Некорректный only-pipe-id.')
        if only_pipe_id not in set(all_129_ids):
            raise ValueError(f'Труба {only_pipe_id} отсутствует в 129 id.')
        all_129_ids = [only_pipe_id]

    if max_pipes is not None:
        all_129_ids = all_129_ids[:max_pipes]

    processing_order, cycle_or_unordered_ids = build_processing_order(all_129_ids, children, parents)
    selected_ids_set = set(all_129_ids)

    stats_rows: list[dict[str, Any]] = []
    missing_tp_rows: list[pd.DataFrame] = []
    missing_no_data_pipes: list[dict[str, Any]] = []

    end_maps: dict[str, dict[str, dict[str, float | None]]] = {}
    calculated_ids: set[str] = set()

    # Streaming outputs (avoid keeping huge segment tables in memory).
    result_csv = out_dir / 'расчет_сегментов__все_трубы.csv'
    zero_csv = out_dir / 'расчет_сегментов__уход_в_ноль.csv'
    stats_csv = out_dir / 'расчет_сегментов__статистика_по_трубам.csv'
    missing_tp_csv = out_dir / 'зеленые_и_цепочка__пропуски_t_p_по_датам.csv'
    missing_pipes_csv = out_dir / 'трубы_без_данных_ни_в_новых_ни_в_7_8__и_нерассчитанные.csv'
    zero_days_csv = out_dir / 'трубы_и_даты__исключены_из_за_p_le_0_01MPa.csv'
    angle_report_csv = out_dir / 'углы__проверка.csv'
    pe2_trace_csv = pe2_trace_csv or (out_dir / 'трейс_pe2__входы_и_выходы.csv')

    for p in [result_csv, zero_csv, stats_csv, missing_tp_csv, missing_pipes_csv, zero_days_csv, angle_report_csv, pe2_trace_csv]:
        if str(p).strip().upper() in WINDOWS_RESERVED_DEVICES:
            continue
        if p.exists():
            p.unlink()

    init_pe2_trace(pe2_trace_csv)

    result_header_written = False
    zero_header_written = False
    result_fixed_columns: list[str] | None = None
    zero_fixed_columns: list[str] | None = None
    segment_rows_total = 0
    zero_events_total = 0
    dropped_days_total = 0
    dropped_days_rows: list[pd.DataFrame] = []
    angle_rows_total = 0
    angle_rows_nonzero_total = 0
    angle_rows_from_alt_total = 0
    angle_rows_nonzero_from_alt_total = 0
    angle_stats_rows: list[dict[str, Any]] = []

    def append_frame(
        df: pd.DataFrame,
        path: Path,
        header_written: bool,
        fixed_columns: list[str] | None = None,
    ) -> tuple[bool, list[str] | None]:
        if df.empty:
            return header_written, fixed_columns
        if fixed_columns is None:
            fixed_columns = list(df.columns)
        frame = df.reindex(columns=fixed_columns)
        frame.to_csv(path, mode='a', index=False, header=not header_written)
        return True, fixed_columns

    # Stage: all requested pipes.
    # Important: predecessor end-values are available only after predecessor calculation,
    # so we iterate in graph-aware order (parents before children).
    for pid in processing_order:
        pred_ids = [pr for pr in parents.get(pid, []) if pr in selected_ids_set]
        calc_df, zeros_df, stats, miss_df = run_one_pipe(
            pipe_id=pid,
            stage='all_129',
            daily_df=daily_df,
            nodes_idx=nodes_idx,
            step_m=step_m,
            calc_mode=calc_mode,
            end_maps=end_maps,
            pred_ids_for_fill=pred_ids if pred_ids else None,
            angle_by_distance=altitudes_by_pipe.get(pid),
        )
        stats_rows.append(stats)
        if not miss_df.empty:
            missing_tp_rows.append(miss_df)
        if stats.get('status') == 'ok':
            result_header_written, result_fixed_columns = append_frame(
                calc_df, result_csv, result_header_written, result_fixed_columns
            )
            segment_rows_total += int(len(calc_df))
            calculated_ids.add(pid)
            angle_deg_series = pd.to_numeric(calc_df.get('angle_deg', pd.Series(dtype=float)), errors='coerce')
            angle_nonzero_mask = angle_deg_series.fillna(0).abs() > 1e-12
            angle_from_alt_mask = calc_df.get('angle_from_altitudes', pd.Series(False, index=calc_df.index)).fillna(False).astype(bool)
            angle_rows = int(len(calc_df))
            angle_nonzero_rows = int(angle_nonzero_mask.sum())
            angle_from_alt_rows = int(angle_from_alt_mask.sum())
            angle_nonzero_from_alt_rows = int((angle_nonzero_mask & angle_from_alt_mask).sum())

            angle_rows_total += angle_rows
            angle_rows_nonzero_total += angle_nonzero_rows
            angle_rows_from_alt_total += angle_from_alt_rows
            angle_rows_nonzero_from_alt_total += angle_nonzero_from_alt_rows

            angle_stats_rows.append(
                {
                    'id': pid,
                    'rows_total': angle_rows,
                    'rows_nonzero_angle': angle_nonzero_rows,
                    'rows_from_altitudes': angle_from_alt_rows,
                    'rows_nonzero_from_altitudes': angle_nonzero_from_alt_rows,
                    'angle_min_deg': None if angle_deg_series.dropna().empty else float(angle_deg_series.min()),
                    'angle_max_deg': None if angle_deg_series.dropna().empty else float(angle_deg_series.max()),
                }
            )
        else:
            missing_no_data_pipes.append(
                {
                    'id': pid,
                    'stage': 'all_129',
                    'status': stats.get('status'),
                    'reason': 'нет данных t/p или нет валидных параметров после fallback новых+7/8',
                }
            )
        if not zeros_df.empty:
            zero_header_written, zero_fixed_columns = append_frame(
                zeros_df, zero_csv, zero_header_written, zero_fixed_columns
            )
            zero_events_total += int(len(zeros_df))
            dropped = zeros_df[zeros_df['metric'] == 'p_day_dropped_below_min'][['id', 'date', 'stage']].drop_duplicates()
            if not dropped.empty:
                dropped_days_rows.append(dropped)
                dropped_days_total += int(len(dropped))

    stats_df = pd.DataFrame(stats_rows)
    missing_tp_df = pd.concat(missing_tp_rows, ignore_index=True) if missing_tp_rows else pd.DataFrame(columns=['id', 'date', 't', 'p', 'reason', 'stage'])
    missing_pipes_df = pd.DataFrame(missing_no_data_pipes)
    dropped_days_df = pd.concat(dropped_days_rows, ignore_index=True).drop_duplicates() if dropped_days_rows else pd.DataFrame(columns=['id', 'date', 'stage'])
    angle_stats_df = pd.DataFrame(angle_stats_rows)

    stats_df.to_csv(stats_csv, index=False)
    missing_tp_df.to_csv(missing_tp_csv, index=False)
    missing_pipes_df.to_csv(missing_pipes_csv, index=False)
    dropped_days_df.to_csv(zero_days_csv, index=False)
    if angle_stats_df.empty:
        angle_stats_df = pd.DataFrame(
            columns=[
                'id',
                'rows_total',
                'rows_nonzero_angle',
                'rows_from_altitudes',
                'rows_nonzero_from_altitudes',
                'angle_min_deg',
                'angle_max_deg',
            ]
        )
    angle_stats_df.to_csv(angle_report_csv, index=False)
    finalize_pe2_trace()

    if not result_csv.exists():
        pd.DataFrame().to_csv(result_csv, index=False)
    if not zero_csv.exists():
        pd.DataFrame(columns=['id', 'date', 'distance', 'metric', 'value', 'stage']).to_csv(zero_csv, index=False)

    summary = {
        'pe2_engine': PE2_SOURCE,
        'calc_mode': calc_mode,
        'nodes_total': int(len(nodes_df)),
        'edges_total': int(len(edges_df)),
        'green_total': int(len(green_set)),
        'orange_total': int(len(orange_set)),
        'run_scope': 'single_pipe' if only_pipe_id else 'all_129_pipes',
        'selected_pipe_id': only_pipe_id,
        'processing_order_mode': 'graph_topological_with_sorted_fallback',
        'processing_cycle_or_unordered_ids_total': int(len(cycle_or_unordered_ids)),
        'requested_pipes_total': int(len(all_129_ids)),
        'calculated_total_pipes': int(len(calculated_ids)),
        'calculated_green_pipes': int(sum(1 for x in calculated_ids if x in green_set)),
        'calculated_non_green_pipes': int(sum(1 for x in calculated_ids if x not in green_set)),
        'segment_rows_total': int(segment_rows_total),
        'zero_events_total': int(zero_events_total),
        'dropped_bad_days_total': int(dropped_days_total),
        'dropped_bad_days_pipes_total': int(dropped_days_df['id'].nunique()) if not dropped_days_df.empty else 0,
        'missing_tp_rows_total': int(len(missing_tp_df)),
        'minimum_pressure_mpa': float(MIN_PRESSURE_MPA),
        'pressure_direction_rule': 'source_p start -> forward; source_p end -> reverse; synthetic -> nearest known side',
        'temperature_direction_rule': 'always physical 0_to_L before pressure calculation',
        'missing_or_not_calculated_pipes_total': int(len(missing_pipes_df)),
        'altitudes': {
            **altitudes_meta,
            'ids_matched_in_129': int(len(set(all_129_ids) & set(altitudes_by_pipe.keys()))),
            'rows_total_in_result': int(angle_rows_total),
            'rows_nonzero_angle': int(angle_rows_nonzero_total),
            'rows_from_altitudes': int(angle_rows_from_alt_total),
            'rows_nonzero_from_altitudes': int(angle_rows_nonzero_from_alt_total),
            'pipes_with_any_altitude_rows': int((angle_stats_df['rows_from_altitudes'] > 0).sum()) if not angle_stats_df.empty else 0,
            'pipes_with_nonzero_angle_rows': int((angle_stats_df['rows_nonzero_angle'] > 0).sum()) if not angle_stats_df.empty else 0,
        },
        'outputs': {
            'result_csv': str(result_csv),
            'stats_csv': str(stats_csv),
            'zero_csv': str(zero_csv),
            'missing_tp_csv': str(missing_tp_csv),
            'missing_pipes_csv': str(missing_pipes_csv),
            'zero_days_csv': str(zero_days_csv),
            'angles_report_csv': str(angle_report_csv),
            'pe2_trace_csv': str(pe2_trace_csv),
            'pe2_trace_rows': int(PE2_TRACE_ROWS_WRITTEN),
            'nodes_csv': str(nodes_csv),
            'edges_csv': str(edges_csv),
            'daily_csv': str(daily_csv),
        },
    }

    summary_path = out_dir / 'пересчет__сводка.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    # ASCII alias for Windows cmd stability (avoid codepage issues with Cyrillic file names in bat).
    summary_ascii_path = out_dir / 'summary.json'
    summary_ascii_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Universal pipeline: JSON -> CSV -> segment recalculation with graph propagation.')
    parser.add_argument('--master-json', type=Path, default=DEFAULT_MASTER_JSON)
    parser.add_argument('--requested-json', type=Path, default=DEFAULT_REQUESTED_JSON)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--pe2-dll', type=Path, default=DEFAULT_PE2_DLL, help='Path to mandatory PE2 DLL.')
    parser.add_argument('--altitudes-csv', type=Path, default=None, help='Optional altitudes CSV (id, segment_start_distance, slope_deg).')
    parser.add_argument('--pe2-trace-csv', type=Path, default=None, help='Optional path for PE2 input/output trace CSV.')
    parser.add_argument('--step-m', type=int, default=10)
    parser.add_argument('--max-pipes', type=int, default=None, help='Optional debug limiter.')
    parser.add_argument('--only-pipe-id', type=str, default=None, help='Optional debug: run only one specific ID from the 129 list.')
    parser.add_argument('--calc-mode', type=str, choices=['strict'], default='strict', help='Only strict mode is allowed.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bootstrap_runtime_imports(args.pe2_dll)

    summary = run_pipeline(
        master_json=args.master_json,
        requested_json=args.requested_json,
        out_dir=args.out_dir,
        step_m=args.step_m,
        max_pipes=args.max_pipes,
        calc_mode=args.calc_mode,
        only_pipe_id=args.only_pipe_id,
        pe2_trace_csv=args.pe2_trace_csv,
        altitudes_csv=args.altitudes_csv,
    )
    print('Done')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
