#!/usr/bin/env python3
"""Compare complete final date profiles with prepared active Megion inputs."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

Q_LIQ = "Жидкости, м3/сут (дебит)"
Q_OIL = "Нефти, т/сут (дебит)"
Q_GAS = "Общего газа, тыс.м3/сут (дебит)"


def num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profiles", type=Path, required=True)
    p.add_argument("--requested", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    calculated = set()
    with args.profiles.open(encoding="utf-8") as handle:
        for line in handle:
            pipe, day = line.rstrip("\n").split("\t")
            calculated.add((pipe, day))

    requested = json.loads(args.requested.read_text(encoding="utf-8"))["by_id"]
    expected = {}
    for pipe, payload in requested.items():
        for record in payload["daily"]:
            rates = [num(record.get(column)) or 0.0 for column in (Q_LIQ, Q_OIL, Q_GAS)]
            if any(rate > 0.0 for rate in rates):
                expected[(str(pipe), record["Дата"])] = record
    if not calculated <= set(expected):
        unknown = sorted(calculated - set(expected))[:10]
        raise ValueError(f"Final contains non-active or unknown profiles: {unknown}")

    rows, mode_count, per_pipe = [], Counter(), defaultdict(Counter)
    for (pipe, day), rec in sorted(expected.items()):
        if (pipe, day) in calculated:
            continue
        q_liq = num(rec.get(Q_LIQ))
        if q_liq is not None and q_liq > 0:
            mode = "Восстановить: Qж > 0"
        else:
            mode = "Оставить без давления: Qж <= 0"
        mode_count[mode] += 1; per_pipe[pipe][mode] += 1
        rows.append({
            "ID простого участка": pipe, "Дата": day,
            "Дебит жидкости, м3/сут": q_liq,
            "Дебит нефти, т/сут": num(rec.get(Q_OIL)),
            "Дебит газа, тыс.м3/сут": num(rec.get(Q_GAS)),
            "Температура входа, C": num(rec.get("t")), "Давление входа, МПа": num(rec.get("p")),
            "Источник температуры": rec.get("source_t"), "Источник давления": rec.get("source_p"),
            "Решение": mode,
        })
    fields = list(rows[0])
    with (args.out_dir / "Пропущенные_активные_даты_проверка_расходов.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    pipe_rows=[]
    for pipe in sorted(per_pipe):
        pipe_rows.append({"ID простого участка":pipe, "Пропущено активных дат":sum(per_pipe[pipe].values()), **per_pipe[pipe]})
    pipe_fields=["ID простого участка", "Пропущено активных дат", *sorted(mode_count)]
    with (args.out_dir / "Пропуски_по_трубам_после_проверки_расходов.csv").open("w", encoding="utf-8-sig", newline="") as h:
        w=csv.DictWriter(h,fieldnames=pipe_fields);w.writeheader();w.writerows(pipe_rows)
    report={"status":"PASS","active_pipe_dates":len(expected),"calculated_profiles":len(calculated),"missing_active_pipe_dates":len(rows),"classification":dict(mode_count),"pipes_with_missing_dates":len(per_pipe)}
    (args.out_dir/"АУДИТ_ПРОПУСКОВ_ДО_ДОЗАПОЛНЕНИЯ.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__ == '__main__': main()
