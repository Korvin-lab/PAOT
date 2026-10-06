#!/usr/bin/env python3
"""Audit missing calculated Megion pipe-days without modifying the source CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


Q_LIQ = "Жидкости, м3/сут (дебит)"
Q_OIL = "Нефти, т/сут (дебит)"
Q_GAS = "Общего газа, тыс.м3/сут (дебит)"


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-csv", type=Path, required=True)
    parser.add_argument("--requested-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    requested = json.loads(args.requested_json.read_text(encoding="utf-8"))["by_id"]
    expected = {}
    for pipe_id, payload in requested.items():
        for row in payload["daily"]:
            # This matches the Windows run's definition of an active date.
            flows = [number(row.get(key)) or 0.0 for key in (Q_LIQ, Q_OIL, Q_GAS)]
            if any(flow > 0.0 for flow in flows):
                expected[(str(pipe_id), row["Дата"])] = row

    # The Windows final CSV is an unquoted numeric CSV.  Read only date/id while
    # checking every row has exactly 51 fields; parsing all 51 strings would make
    # this independent audit unnecessarily slow on a 92-GB external-SSD file.
    calculated = set()
    rows = 0
    with args.final_csv.open("rb", buffering=16 * 1024 * 1024) as source:
        header = source.readline().decode("utf-8-sig").rstrip("\r\n").split(",")
        if header[:3] != ["date", "id", "Qv"] or len(header) != 51:
            raise ValueError(f"Unexpected final schema: {header[:3]} / {len(header)} columns")
        for line in source:
            rows += 1
            if line.count(b",") != 50:
                raise ValueError(f"Malformed source row {rows + 1}: expected 51 CSV fields")
            date_value, pipe_id, _ = line.split(b",", 2)
            calculated.add((pipe_id.decode("ascii"), date_value.decode("ascii")))

    missing = []
    classifications = Counter()
    by_pipe = defaultdict(lambda: Counter())
    for key, row in sorted(expected.items(), key=lambda item: (item[0][0], item[0][1])):
        if key in calculated:
            continue
        q_liq = number(row.get(Q_LIQ))
        q_oil = number(row.get(Q_OIL))
        q_gas = number(row.get(Q_GAS))
        if q_liq is not None and q_liq > 0.0:
            category = "есть расход жидкости: восстановить профиль"
        else:
            category = "нет расхода жидкости: остановка/газовый режим, давление не подставлять"
        classifications[category] += 1
        by_pipe[key[0]][category] += 1
        missing.append({
            "ID простого участка": key[0],
            "Дата": key[1],
            "Дебит жидкости, м3/сут": q_liq,
            "Дебит нефти, т/сут": q_oil,
            "Дебит газа, тыс.м3/сут": q_gas,
            "Решение": category,
            "Температура входа, C": number(row.get("t")),
            "Давление входа, МПа": number(row.get("p")),
            "Источник температуры": row.get("source_t"),
            "Источник давления": row.get("source_p"),
        })

    fields = list(missing[0]) if missing else ["ID простого участка", "Дата", "Решение"]
    with (args.out_dir / "Пропущенные_активные_даты_проверка_расходов.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(missing)

    pipe_rows = []
    for pipe_id in sorted(by_pipe):
        pipe_rows.append({"ID простого участка": pipe_id, **by_pipe[pipe_id]})
    pipe_fields = ["ID простого участка", *sorted(classifications)]
    with (args.out_dir / "Пропуски_по_трубам_после_проверки_расходов.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=pipe_fields)
        writer.writeheader()
        writer.writerows(pipe_rows)

    report = {
        "status": "PASS",
        "source_final_csv": str(args.final_csv),
        "source_final_data_rows": rows,
        "source_final_columns": len(header),
        "expected_active_pipe_dates": len(expected),
        "calculated_active_pipe_dates": len(calculated & set(expected)),
        "missing_active_pipe_dates": len(missing),
        "classification": dict(classifications),
        "pipes_with_missing_active_dates": len(by_pipe),
        "rule": "Only a missing active date with Q_liq > 0 can receive an interpolated segment profile. A date with Q_liq <= 0 remains uncalculated; no pressure is fabricated.",
    }
    (args.out_dir / "АУДИТ_ПРОПУСКОВ_ДО_ДОЗАПОЛНЕНИЯ.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
