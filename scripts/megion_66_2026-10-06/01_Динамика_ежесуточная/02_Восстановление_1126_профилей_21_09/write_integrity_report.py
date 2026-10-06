#!/usr/bin/env python3
"""Write a concise, reproducible integrity report for the copied final CSV."""

import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[2] / (
    "MEGION_WINDOWS_FULL_PE2__LAST_RUN_PC_TO_SSD_2026-09-21/"
    "RUN_2026-09-19_17-30-11/final/final_dataset__WITH_H2S_GAS_PHASE.csv"
)
OUT = Path(__file__).resolve().parent / "audit" / "АУДИТ_ЦЕЛОСТНОСТИ_SSD_КОПИИ.json"
LAST_NEWLINE_END = 2_333_081_507


def main():
    size = SOURCE.stat().st_size
    with SOURCE.open("rb") as handle:
        handle.seek(LAST_NEWLINE_END)
        broken_start = handle.read(4096)
        handle.seek(-4096, 2)
        file_end = handle.read(4096)
    report = {
        "status": "FAIL",
        "source": str(SOURCE),
        "file_size_bytes": size,
        "completed_data_rows": 3_189_089,
        "completed_pipe_date_profiles": 5_172,
        "last_completed_byte_offset_exclusive": LAST_NEWLINE_END,
        "zero_filled_trailing_bytes": size - LAST_NEWLINE_END,
        "evidence": {
            "next_record_prefix": broken_start[:120].decode("ascii", errors="replace"),
            "next_4096_bytes_nul_count": broken_start.count(b"\0"),
            "last_4096_bytes_nul_count": file_end.count(b"\0"),
            "next_4096_bytes_lf_count": broken_start.count(b"\n"),
            "last_4096_bytes_lf_count": file_end.count(b"\n"),
        },
        "expected_from_windows_run": {
            "data_rows": 134_573_227,
            "calculated_active_pipe_dates": 70_334,
            "sha256_windows_source": "85B05BD428107EAF3B870FCF235A0D948E745F48FF2E506B7A9D22BE5AC7E036",
        },
        "sha256_ssd_copy": "0f11c67e2a837f669ae99f7fada94b37380780e95a4a61c6af2538c0c903c23c",
        "sha256_matches_windows_source": False,
        "conclusion": "The mounted SSD file is truncated and zero-padded. Do not use it for CSV post-fill or coverage analysis.",
        "required_next_step": "Obtain the intact Windows source CSV or recopy it, then compare full SHA-256 before any post-fill.",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
