from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

Q_LIQ_KEY = 'Жидкости, м3/сут (дебит)'
Q_OIL_KEY = 'Нефти, т/сут (дебит)'
Q_GAS_KEY = 'Общего газа, тыс.м3/сут (дебит)'
WATERCUT_KEY = 'Обводненность, %'
RHO_OIL_KEY = 'Нефти, кг/м3'


def to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if math.isnan(f):
            return None
        return f
    s = str(v).strip().replace(',', '.')
    if not s:
        return None
    try:
        f = float(s)
        if math.isnan(f):
            return None
        return f
    except Exception:
        return None


def norm_id(v: Any) -> str:
    s = '' if v is None else str(v)
    s = ''.join(ch for ch in s if ch.isdigit())
    return s or ('' if v is None else str(v).strip())


def date_key(rec: dict[str, Any]) -> pd.Timestamp | None:
    d = rec.get('Дата', rec.get('date'))
    dt = pd.to_datetime(d, errors='coerce')
    if pd.isna(dt):
        return None
    return pd.Timestamp(dt).normalize()


def fround(v: float | None) -> float | None:
    if v is None:
        return None
    return float(round(float(v), 10))


def process_requested(requested: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    by_id = requested.get('by_id', {})

    fill_rows: list[dict[str, Any]] = []
    inactive_rows: list[dict[str, Any]] = []

    total_rows = 0
    active_rows = 0
    inactive_cnt = 0

    filled_q_liq = 0
    filled_watercut = 0
    filled_q_oil = 0
    filled_q_gas = 0
    filled_rho_oil = 0
    filled_rho_oil_pipe_mean = 0

    ids_with_any_fill: set[str] = set()

    for pid_raw, payload in by_id.items():
        pid = norm_id(pid_raw)
        daily = payload.get('daily', [])

        # Keep stable order by date (invalid dates go to tail unchanged).
        indexed = []
        invalid = []
        for rec in daily:
            dk = date_key(rec)
            if dk is None:
                invalid.append(rec)
            else:
                indexed.append((dk, rec))
        indexed.sort(key=lambda x: x[0])

        positive_rho_values = [to_float(rec.get(RHO_OIL_KEY)) for _, rec in indexed]
        positive_rho_values = [v for v in positive_rho_values if v is not None and v > 0]
        pipe_mean_rho_oil = (sum(positive_rho_values) / len(positive_rho_values)) if positive_rho_values else None

        prev_q_liq = None
        prev_watercut = None
        prev_rho_oil = None
        prev_gas_factor = None  # q_gas(тыс м3/сут) / q_oil_m3d

        for dk, rec in indexed:
            total_rows += 1

            q_liq = to_float(rec.get(Q_LIQ_KEY))
            q_oil = to_float(rec.get(Q_OIL_KEY))
            q_gas = to_float(rec.get(Q_GAS_KEY))
            watercut = to_float(rec.get(WATERCUT_KEY))
            rho_oil = to_float(rec.get(RHO_OIL_KEY))

            is_active = bool((q_liq is not None and q_liq > 0) or (q_oil is not None and q_oil > 0) or (q_gas is not None and q_gas > 0))

            if not is_active:
                inactive_cnt += 1
                inactive_rows.append({'id': pid, 'date': dk.strftime('%Y-%m-%d'), 'status': 'бездействующая'})
                continue

            active_rows += 1

            f_q_liq = False
            f_wc = False
            f_q_oil = False
            f_q_gas = False
            f_rho_oil = False

            # 1) Жидкость: предыдущее значение
            if not (q_liq is not None and q_liq > 0) and (prev_q_liq is not None and prev_q_liq > 0):
                q_liq = prev_q_liq
                rec[Q_LIQ_KEY] = fround(q_liq)
                f_q_liq = True
                filled_q_liq += 1

            # watercut: текущая, иначе предыдущая
            if watercut is None and prev_watercut is not None:
                watercut = prev_watercut
                rec[WATERCUT_KEY] = fround(watercut)
                f_wc = True
                filled_watercut += 1

            # rho_oil: текущая; затем предыдущее положительное; для начального разрыва среднее по этой трубе.
            if rho_oil is None or rho_oil <= 0:
                if prev_rho_oil is not None and prev_rho_oil > 0:
                    rho_oil = prev_rho_oil
                elif pipe_mean_rho_oil is not None and pipe_mean_rho_oil > 0:
                    rho_oil = pipe_mean_rho_oil
                    filled_rho_oil_pipe_mean += 1
                if rho_oil is not None and rho_oil > 0:
                    rec[RHO_OIL_KEY] = fround(rho_oil)
                    f_rho_oil = True
                    filled_rho_oil += 1

            # q_oil_m3d для пересчета
            q_oil_m3d = None
            if (q_liq is not None and q_liq > 0) and (watercut is not None):
                q_oil_m3d = q_liq * (1.0 - watercut / 100.0)

            # 2) Нефть/вода: через watercut
            if not (q_oil is not None and q_oil > 0):
                if (q_oil_m3d is not None and q_oil_m3d > 0) and (rho_oil is not None and rho_oil > 0):
                    q_oil = q_oil_m3d * rho_oil / 1000.0
                    rec[Q_OIL_KEY] = fround(q_oil)
                    f_q_oil = True
                    filled_q_oil += 1

            # 3) Газ: через газовый фактор текущий/предыдущий
            if (q_gas is not None and q_gas > 0) and (q_oil_m3d is not None and q_oil_m3d > 0):
                prev_gas_factor = q_gas / q_oil_m3d

            if not (q_gas is not None and q_gas > 0):
                if (q_oil_m3d is not None and q_oil_m3d > 0) and (prev_gas_factor is not None and prev_gas_factor > 0):
                    q_gas = prev_gas_factor * q_oil_m3d
                    rec[Q_GAS_KEY] = fround(q_gas)
                    f_q_gas = True
                    filled_q_gas += 1

            # update prev trackers from final row values
            q_liq_after = to_float(rec.get(Q_LIQ_KEY))
            watercut_after = to_float(rec.get(WATERCUT_KEY))
            rho_oil_after = to_float(rec.get(RHO_OIL_KEY))
            q_oil_after = to_float(rec.get(Q_OIL_KEY))
            q_gas_after = to_float(rec.get(Q_GAS_KEY))

            if q_liq_after is not None and q_liq_after > 0:
                prev_q_liq = q_liq_after
            if watercut_after is not None:
                prev_watercut = watercut_after
            if rho_oil_after is not None and rho_oil_after > 0:
                prev_rho_oil = rho_oil_after

            q_oil_m3d_after = None
            if (q_liq_after is not None and q_liq_after > 0) and (watercut_after is not None):
                q_oil_m3d_after = q_liq_after * (1.0 - watercut_after / 100.0)
            if (q_gas_after is not None and q_gas_after > 0) and (q_oil_m3d_after is not None and q_oil_m3d_after > 0):
                prev_gas_factor = q_gas_after / q_oil_m3d_after

            if f_q_liq or f_wc or f_q_oil or f_q_gas or f_rho_oil:
                ids_with_any_fill.add(pid)
                fill_rows.append(
                    {
                        'id': pid,
                        'date': dk.strftime('%Y-%m-%d'),
                        'filled_q_liq_prev': f_q_liq,
                        'filled_watercut_prev': f_wc,
                        'filled_q_oil_recalc': f_q_oil,
                        'filled_q_gas_gf': f_q_gas,
                        'filled_rho_oil_prev': f_rho_oil,
                    }
                )

        # preserve invalid-date records in original tail position behavior
        payload['daily'] = [r for _, r in indexed] + invalid

    report = {
        'total_ids': int(len(by_id)),
        'total_rows': int(total_rows),
        'active_rows': int(active_rows),
        'inactive_rows': int(inactive_cnt),
        'filled_q_liq_prev': int(filled_q_liq),
        'filled_watercut_prev': int(filled_watercut),
        'filled_q_oil_recalc': int(filled_q_oil),
        'filled_q_gas_gf': int(filled_q_gas),
        'filled_rho_oil_prev_or_pipe_mean': int(filled_rho_oil),
        'filled_rho_oil_pipe_mean': int(filled_rho_oil_pipe_mean),
        'ids_with_any_fill': int(len(ids_with_any_fill)),
    }

    return requested, report, pd.DataFrame(fill_rows), pd.DataFrame(inactive_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Pre-fill active daily rows in requested JSON before strict run.')
    p.add_argument('--json', type=Path, required=True, help='Path to input requested JSON.')
    p.add_argument('--out-dir', type=Path, required=True, help='Output dir for reports.')
    p.add_argument('--write-backup', action='store_true', help='Write backup JSON before in-place update.')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested = json.loads(args.json.read_text(encoding='utf-8'))

    if args.write_backup:
        backup = args.json.with_name(args.json.stem + '__before_prefill.json')
        backup.write_text(json.dumps(requested, ensure_ascii=False, indent=2), encoding='utf-8')

    updated, report, fills_df, inactive_df = process_requested(requested)

    args.json.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding='utf-8')

    (args.out_dir / 'prefill_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    fills_df.to_csv(args.out_dir / 'prefill_fills_log.csv', index=False)
    inactive_df.to_csv(args.out_dir / 'prefill_inactive_dates.csv', index=False)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
