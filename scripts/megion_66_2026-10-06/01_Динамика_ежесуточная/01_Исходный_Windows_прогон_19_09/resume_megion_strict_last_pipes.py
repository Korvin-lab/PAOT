"""Resume the interrupted strict Megion run without recalculating finished pipes."""
from __future__ import annotations

import json
import traceback
from pathlib import Path

import pandas as pd

import main_pipeline_final_csv as pipeline


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "RUN_2026-09-19_17-30-11"
STRICT_DIR = RUN_DIR / "strict"
REMAINING_IDS = ["1751068454", "1751067083"]
TEMP_SEGMENTS_CSV = STRICT_DIR / "resume_last_2_pipes__segments.tmp.csv"
TEMP_ZEROS_CSV = STRICT_DIR / "resume_last_2_pipes__zeros.tmp.csv"
PROGRESS_JSONL = STRICT_DIR / "resume_last_2_pipes__progress.jsonl"
BATCH_DATES = 30


def find_result_csv() -> Path:
    candidates = sorted(STRICT_DIR.glob("*.csv"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError("Strict segment CSV was not found")
    return candidates[0]


def find_zero_csv() -> Path | None:
    for path in STRICT_DIR.glob("*.csv"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            header = handle.readline().strip()
        if header.startswith("id,date,distance,metric,value,stage"):
            return path
    return None


def append_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    header = pd.read_csv(path, nrows=0).columns.tolist() if path.exists() and path.stat().st_size else list(frame.columns)
    frame.reindex(columns=header).to_csv(path, mode="a", index=False, header=not path.exists() or path.stat().st_size == 0)


def append_with_existing_header(frame: pd.DataFrame, path: Path, header: list[str]) -> None:
    if frame.empty:
        return
    frame.reindex(columns=header).to_csv(path, mode="a", index=False, header=not path.exists() or path.stat().st_size == 0)


def write_progress(payload: dict[str, object]) -> None:
    with PROGRESS_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    pipeline.bootstrap_runtime_imports(ROOT / "deps" / "pe_2_main.dll")
    result_csv = find_result_csv()
    zero_csv = find_zero_csv()
    seen_dates: set[tuple[str, str]] = set()
    if PROGRESS_JSONL.exists():
        for line in PROGRESS_JSONL.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen_dates.add((str(row["id"]), str(row["date"])))
    nodes_csv = STRICT_DIR / "csv_inputs" / "graph_nodes_129.csv"
    edges_csv = STRICT_DIR / "csv_inputs" / "graph_edges_129.csv"
    daily_csv = STRICT_DIR / "csv_inputs" / "pipe_daily_requested_129.csv"

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    daily_df = pd.read_csv(daily_csv, low_memory=False)
    nodes_df["id"] = nodes_df["id"].map(pipeline.normalize_id)
    edges_df["source"] = edges_df["source"].map(pipeline.normalize_id)
    edges_df["target"] = edges_df["target"].map(pipeline.normalize_id)
    daily_df["id"] = daily_df["id"].map(pipeline.normalize_id)
    daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce").dt.normalize()
    daily_df = daily_df[daily_df["date"].notna()].copy()

    nodes_idx = {row.id: row for _, row in nodes_df.iterrows()}
    children, parents = pipeline.build_adjacency(edges_df)
    all_ids = sorted(nodes_df["id"].astype(str).tolist())
    processing_order, unordered = pipeline.build_processing_order(all_ids, children, parents)
    expected_tail = processing_order[-2:]
    if expected_tail != REMAINING_IDS:
        raise RuntimeError(f"Unexpected remaining-pipe order: {expected_tail}")

    # The prepared daily input already contains valid T/P for both remaining pipes.
    # Their predecessor maps are therefore not consulted by run_one_pipe.
    end_maps: dict[str, dict[str, dict[str, float | None]]] = {}
    report: dict[str, object] = {
        "status": "started",
        "existing_segment_csv": str(result_csv),
        "temp_segments_csv": str(TEMP_SEGMENTS_CSV),
        "remaining_ids": REMAINING_IDS,
        "unordered_ids": unordered,
        "pipes": [],
    }
    result_header = pd.read_csv(result_csv, nrows=0).columns.tolist()
    zero_header = pd.read_csv(zero_csv, nrows=0).columns.tolist() if zero_csv is not None else None

    for pipe_id in REMAINING_IDS:
        pipe_rows = daily_df[daily_df["id"].astype(str) == pipe_id].copy()
        pipe_dates = [
            date_value for date_value in sorted(pipe_rows["date"].dropna().unique())
            if (pipe_id, str(pd.Timestamp(date_value).date())) not in seen_dates
        ]
        pipe_stats: dict[str, object] = {
            "id": pipe_id,
            "status": "ok",
            "rows_input": int(len(pipe_rows)),
            "rows_valid": 0,
            "rows_segments": 0,
            "dates_count": 0,
            "zero_events": 0,
            "dropped_bad_days": 0,
            "missing_tp_rows_before_runtime_fill": 0,
        }
        for offset in range(0, len(pipe_dates), BATCH_DATES):
            date_chunk = pipe_dates[offset:offset + BATCH_DATES]
            date_keys = [str(pd.Timestamp(date_value).date()) for date_value in date_chunk]
            batch_df = pipe_rows[pipe_rows["date"].isin(date_chunk)].copy()
            try:
                calc_df, zeros_df, stats, missing_df = pipeline.run_one_pipe(
                    pipe_id=pipe_id,
                    stage="all_129",
                    daily_df=batch_df,
                    nodes_idx=nodes_idx,
                    step_m=10,
                    calc_mode="strict",
                    end_maps=end_maps,
                    pred_ids_for_fill=parents.get(pipe_id, []),
                    angle_by_distance=None,
                )
            except Exception as exc:
                report["status"] = "error"
                report["failed_id"] = pipe_id
                report["failed_dates"] = date_keys
                report["error"] = repr(exc)
                report["traceback"] = traceback.format_exc()
                (STRICT_DIR / "STRICT_RESUME_LAST_2_PIPES_REPORT.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise
            if stats.get("status") not in {"ok", "all_days_dropped_by_nonpositive_p", "no_valid_rows_after_mask"}:
                raise RuntimeError(f"{pipe_id} {date_keys[0]}..{date_keys[-1]}: strict resume failed: {stats}")
            append_with_existing_header(calc_df, TEMP_SEGMENTS_CSV, result_header)
            if zero_header is not None:
                append_with_existing_header(zeros_df, TEMP_ZEROS_CSV, zero_header)
            pipe_stats["rows_valid"] = int(pipe_stats["rows_valid"]) + int(stats.get("rows_valid", 0))
            pipe_stats["rows_segments"] = int(pipe_stats["rows_segments"]) + int(len(calc_df))
            pipe_stats["dates_count"] = int(pipe_stats["dates_count"]) + int(calc_df["date"].nunique() if not calc_df.empty else 0)
            pipe_stats["zero_events"] = int(pipe_stats["zero_events"]) + int(len(zeros_df))
            pipe_stats["dropped_bad_days"] = int(pipe_stats["dropped_bad_days"]) + int(stats.get("dropped_bad_days", 0))
            pipe_stats["missing_tp_rows_before_runtime_fill"] = int(pipe_stats["missing_tp_rows_before_runtime_fill"]) + int(len(missing_df))
            segment_counts_by_date: dict[str, int] = {}
            if not calc_df.empty:
                grouped_counts = calc_df.groupby(calc_df["date"].astype(str)).size()
                segment_counts_by_date = {str(k)[:10]: int(v) for k, v in grouped_counts.items()}
            temp_bytes = TEMP_SEGMENTS_CSV.stat().st_size if TEMP_SEGMENTS_CSV.exists() else 0
            for date_key in date_keys:
                write_progress(
                    {
                        "id": pipe_id,
                        "date": date_key,
                        "rows_segments": int(segment_counts_by_date.get(date_key, 0)),
                        "temp_segments_bytes": temp_bytes,
                    }
                )
        report["pipes"].append(
            {
                "id": pipe_id,
                "stats": pipe_stats,
            }
        )

    with TEMP_SEGMENTS_CSV.open("r", encoding="utf-8", errors="replace") as src, result_csv.open("a", encoding="utf-8", newline="") as dst:
        _ = src.readline()
        for line in src:
            dst.write(line)
    if zero_csv is not None and TEMP_ZEROS_CSV.exists() and TEMP_ZEROS_CSV.stat().st_size:
        with TEMP_ZEROS_CSV.open("r", encoding="utf-8", errors="replace") as src, zero_csv.open("a", encoding="utf-8", newline="") as dst:
            _ = src.readline()
            for line in src:
                dst.write(line)

    report["status"] = "complete"
    report["segment_csv_bytes_after_resume"] = result_csv.stat().st_size
    report["temp_segment_csv_bytes"] = TEMP_SEGMENTS_CSV.stat().st_size if TEMP_SEGMENTS_CSV.exists() else 0
    (STRICT_DIR / "STRICT_RESUME_LAST_2_PIPES_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
