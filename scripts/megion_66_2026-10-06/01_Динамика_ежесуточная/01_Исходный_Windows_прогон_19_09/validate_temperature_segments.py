from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


TOL = 1e-9


def normalize_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def expected_segment_count(length_m: float, step_m: float) -> int:
    full_steps = int(math.floor(length_m / step_m))
    count = full_steps + 1
    if not math.isclose(full_steps * step_m, length_m, abs_tol=TOL, rel_tol=0.0):
        count += 1
    return count


def load_lengths(master_json: Path) -> dict[str, float]:
    data = json.loads(master_json.read_text(encoding="utf-8-sig"))
    result: dict[str, float] = {}
    for node in data.get("nodes", []):
        pipe_id = normalize_id(node.get("id", ""))
        graph = node.get("graph") or {}
        value = graph.get("L_m", graph.get("L_m_graph"))
        try:
            length = float(value)
        except (TypeError, ValueError):
            continue
        if pipe_id and math.isfinite(length) and length > 0:
            result[pipe_id] = length
    if not result:
        raise ValueError("No positive pipe lengths were found in master JSON.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate every temperature profile and every segment interval before stage1."
    )
    parser.add_argument("--segments-csv", required=True, type=Path)
    parser.add_argument("--master-json", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--errors-csv", required=True, type=Path)
    parser.add_argument("--step-m", type=float, default=10.0)
    parser.add_argument("--temperature-min", type=float, default=5.0)
    parser.add_argument("--temperature-max", type=float, default=90.0)
    parser.add_argument("--floor-epsilon", type=float, default=0.05)
    parser.add_argument("--max-floor-share-per-profile", type=float, default=0.05)
    parser.add_argument("--max-floor-share-total", type=float, default=0.005)
    parser.add_argument("--fail-on-floor-like-share", action="store_true")
    return parser.parse_args()


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    lengths = load_lengths(args.master_json)
    errors: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    pipes: set[str] = set()
    completed_keys: set[tuple[str, str]] = set()
    rows_total = 0
    profiles_total = 0
    temperature_floor_rows = 0
    temperature_floor_like_rows = 0
    temperature_ceiling_rows = 0

    current_key: tuple[str, str] | None = None
    current: dict[str, Any] = {}

    def add_error(code: str, pipe_id: str, date: str, details: str) -> None:
        error_counts[code] += 1
        if len(errors) < 10000:
            errors.append({"id": pipe_id, "date": date, "error": code, "details": details})

    def finish_profile() -> None:
        nonlocal profiles_total
        if current_key is None:
            return
        profiles_total += 1
        pipe_id, date = current_key
        length = lengths.get(pipe_id)
        if length is None:
            add_error("missing_length", pipe_id, date, "Pipe length is absent in master JSON")
            return
        if not math.isclose(current["first_x"], 0.0, abs_tol=TOL, rel_tol=0.0):
            add_error("profile_does_not_start_at_zero", pipe_id, date, f"start={current['first_x']}")
        if not math.isclose(current["last_x"], length, abs_tol=1e-7, rel_tol=0.0):
            add_error("profile_does_not_end_at_length", pipe_id, date, f"end={current['last_x']}; L={length}")
        expected = expected_segment_count(length, args.step_m)
        if current["rows"] != expected:
            add_error("segment_count_mismatch", pipe_id, date, f"rows={current['rows']}; expected={expected}; L={length}")
        if current["nonpositive_steps"]:
            add_error("duplicate_or_reverse_segment", pipe_id, date, f"count={current['nonpositive_steps']}")
        if current["oversized_steps"]:
            add_error("segment_gap_gt_step", pipe_id, date, f"count={current['oversized_steps']}; max={current['max_step']}")
        if current["temperature_out_of_range"]:
            add_error("temperature_out_of_range", pipe_id, date, f"count={current['temperature_out_of_range']}")
        if current["nonfinite_temperature"]:
            add_error("nonfinite_temperature", pipe_id, date, f"count={current['nonfinite_temperature']}")
        if current["temperature_increase_steps"]:
            add_error("temperature_increases_along_pipe", pipe_id, date, f"count={current['temperature_increase_steps']}")
        if current["temperature_alias_mismatch"]:
            add_error("temperature_alias_mismatch", pipe_id, date, f"count={current['temperature_alias_mismatch']}")
        floor_share = current["temperature_floor_like_steps"] / current["rows"] if current["rows"] else 0.0
        if args.fail_on_floor_like_share and floor_share > args.max_floor_share_per_profile:
            add_error(
                "temperature_floor_plateau",
                pipe_id,
                date,
                f"floor_like_rows={current['temperature_floor_like_steps']}; rows={current['rows']}; share={floor_share:.6f}",
            )
        if (
            current["temperature_max"] - current["temperature_min"] <= TOL
            and current["temperature_min"] > args.temperature_min + TOL
        ):
            add_error("constant_profile_above_ambient", pipe_id, date, f"T={current['temperature_min']}")
        if (
            current["rows"] >= 3
            and current["second_t"] - current["first_t"] > TOL
            and current["post_first_max"] - current["post_first_min"] <= TOL
        ):
            add_error(
                "first_jump_then_flat_profile",
                pipe_id,
                date,
                f"T0={current['first_t']}; T1={current['second_t']}; T_end={current['last_t']}",
            )
        completed_keys.add(current_key)

    with args.segments_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        fields = set(header)
        id_col = "id"
        date_col = "date"
        x_col = "distance" if "distance" in fields else "segment_id"
        t_col = "t" if "t" in fields else ("seg_temperature" if "seg_temperature" in fields else "temperature")
        alias_col = "seg_temperature" if t_col != "seg_temperature" and "seg_temperature" in fields else None
        required = {id_col, date_col, x_col, t_col}
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"Segments CSV is missing required columns: {missing}")
        indexes = {name: header.index(name) for name in required}
        alias_index = header.index(alias_col) if alias_col else None
        floor_index = header.index("temperature_floor_applied") if "temperature_floor_applied" in fields else None
        ceiling_index = header.index("temperature_ceiling_applied") if "temperature_ceiling_applied" in fields else None

        for values in reader:
            rows_total += 1
            pipe_id = normalize_id(values[indexes[id_col]])
            date = str(values[indexes[date_col]]).strip()
            key = (pipe_id, date)
            try:
                x = float(values[indexes[x_col]])
            except (TypeError, ValueError):
                add_error("invalid_distance", pipe_id, date, repr(values[indexes[x_col]]))
                continue
            try:
                temperature = float(values[indexes[t_col]])
            except (TypeError, ValueError):
                temperature = math.nan

            if key != current_key:
                finish_profile()
                if key in completed_keys:
                    add_error("noncontiguous_profile", pipe_id, date, "The same ID-date appears in more than one block")
                current_key = key
                pipes.add(pipe_id)
                finite_t = math.isfinite(temperature)
                current = {
                    "rows": 1,
                    "first_x": x,
                    "last_x": x,
                    "last_t": temperature,
                    "first_t": temperature,
                    "second_t": temperature,
                    "post_first_min": math.inf,
                    "post_first_max": -math.inf,
                    "nonpositive_steps": 0,
                    "oversized_steps": 0,
                    "max_step": 0.0,
                    "temperature_increase_steps": 0,
                    "temperature_out_of_range": int(finite_t and not (args.temperature_min <= temperature <= args.temperature_max)),
                    "nonfinite_temperature": int(not finite_t),
                    "temperature_alias_mismatch": 0,
                    "temperature_floor_like_steps": int(finite_t and temperature <= args.temperature_min + args.floor_epsilon),
                    "temperature_min": temperature,
                    "temperature_max": temperature,
                }
            else:
                step = x - current["last_x"]
                current["rows"] += 1
                if current["rows"] == 2:
                    current["second_t"] = temperature
                current["post_first_min"] = min(current["post_first_min"], temperature)
                current["post_first_max"] = max(current["post_first_max"], temperature)
                if step <= TOL:
                    current["nonpositive_steps"] += 1
                if step > args.step_m + TOL:
                    current["oversized_steps"] += 1
                current["max_step"] = max(current["max_step"], step)
                if math.isfinite(temperature) and math.isfinite(current["last_t"]):
                    if temperature - current["last_t"] > TOL:
                        current["temperature_increase_steps"] += 1
                current["last_x"] = x
                current["last_t"] = temperature
                if not math.isfinite(temperature):
                    current["nonfinite_temperature"] += 1
                elif not (args.temperature_min <= temperature <= args.temperature_max):
                    current["temperature_out_of_range"] += 1
                elif temperature <= args.temperature_min + args.floor_epsilon:
                    current["temperature_floor_like_steps"] += 1
                if math.isfinite(temperature):
                    current["temperature_min"] = min(current["temperature_min"], temperature)
                    current["temperature_max"] = max(current["temperature_max"], temperature)

            if alias_index is not None:
                try:
                    alias = float(values[alias_index])
                except (TypeError, ValueError):
                    alias = math.nan
                if not (math.isfinite(alias) and math.isfinite(temperature) and math.isclose(alias, temperature, abs_tol=TOL, rel_tol=0.0)):
                    current["temperature_alias_mismatch"] += 1
            if floor_index is not None and str(values[floor_index]).strip().lower() in {"1", "true", "yes"}:
                temperature_floor_rows += 1
            if math.isfinite(temperature) and temperature <= args.temperature_min + args.floor_epsilon:
                temperature_floor_like_rows += 1
            if ceiling_index is not None and str(values[ceiling_index]).strip().lower() in {"1", "true", "yes"}:
                temperature_ceiling_rows += 1

    finish_profile()

    floor_like_share_total = temperature_floor_like_rows / rows_total if rows_total else 0.0
    if temperature_floor_rows:
        add_error(
            "temperature_floor_applied",
            "__ALL__",
            "__ALL__",
            f"floor_applied_rows={temperature_floor_rows}",
        )
    if args.fail_on_floor_like_share and floor_like_share_total > args.max_floor_share_total:
        add_error(
            "temperature_floor_plateau_total",
            "__ALL__",
            "__ALL__",
            f"floor_like_rows={temperature_floor_like_rows}; rows={rows_total}; share={floor_like_share_total:.6f}",
        )
    status = "PASS" if not error_counts else "FAIL"
    return {
        "status": status,
        "inputs": {"segments_csv": str(args.segments_csv), "master_json": str(args.master_json)},
        "rules": {
            "segment_step_m": args.step_m,
            "temperature_range_c": [args.temperature_min, args.temperature_max],
            "profile_rule": "0..L, strictly increasing coordinates, no step above configured size",
            "temperature_rule": "finite, bounded, nonincreasing along physical 0..L, no first-jump-then-flat profile",
            "temperature_floor_plateau_rule": {
                "floor_like_threshold_c": args.temperature_min + args.floor_epsilon,
                "max_floor_share_per_profile": args.max_floor_share_per_profile,
                "max_floor_share_total": args.max_floor_share_total,
                "fail_on_floor_like_share": bool(args.fail_on_floor_like_share),
                "note": "Near-ambient values are not a failure when ambient temperature is fixed at the minimum; actual floor clamp is checked via temperature_floor_applied.",
            },
        },
        "counts": {
            "rows": rows_total,
            "profiles_id_date": profiles_total,
            "pipes": len(pipes),
            "error_events": int(sum(error_counts.values())),
            "error_examples_saved": len(errors),
            "temperature_floor_rows": temperature_floor_rows,
            "temperature_floor_like_rows": temperature_floor_like_rows,
            "temperature_floor_like_share_total": floor_like_share_total,
            "temperature_ceiling_rows": temperature_ceiling_rows,
        },
        "errors_by_type": dict(sorted(error_counts.items())),
    }, errors


def main() -> int:
    args = parse_args()
    if args.step_m <= 0:
        raise ValueError("--step-m must be positive")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.errors_csv.parent.mkdir(parents=True, exist_ok=True)
    report, errors = validate(args)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.errors_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "date", "error", "details"])
        writer.writeheader()
        writer.writerows(errors)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
