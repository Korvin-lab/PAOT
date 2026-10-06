"""Run the remaining Megion stages only after the strict resume is complete."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "RUN_2026-09-19_17-30-11"
STRICT = RUN_DIR / "strict"
STAGE = RUN_DIR / "stage1"
FINAL = RUN_DIR / "final"
REPORT = STRICT / "STRICT_RESUME_LAST_2_PIPES_REPORT.json"
LOG = RUN_DIR / "logs" / "complete_after_resume.log"


def run(script: str, *args: Path | str) -> None:
    command = [sys.executable, str(ROOT / script), *map(str, args)]
    with LOG.open("a", encoding="utf-8") as log:
        log.write("RUN: " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    while not REPORT.exists():
        time.sleep(30)
    status = json.loads(REPORT.read_text(encoding="utf-8")).get("status")
    if status != "complete":
        raise RuntimeError(f"Strict resume report has unexpected status: {status}")

    segments = max(STRICT.glob("*.csv"), key=lambda path: path.stat().st_size)
    run(
        "validate_temperature_segments.py",
        "--segments-csv", segments,
        "--master-json", ROOT / "input" / "graph_megion_master.json",
        "--step-m", "10", "--temperature-min", "5", "--temperature-max", "90",
        "--floor-epsilon", "0.05", "--max-floor-share-per-profile", "0.05",
        "--max-floor-share-total", "0.005",
        "--report-json", STRICT / "TEMPERATURE_AND_SEGMENTS_VALIDATION.json",
        "--errors-csv", STRICT / "TEMPERATURE_AND_SEGMENTS_ERRORS.csv",
    )
    run(
        "build_final_dataset_stage1_co2.py",
        "--segments-csv", segments,
        "--chem-daily-csv", ROOT / "input" / "chem_daily_megion.csv",
        "--kvch-daily-csv", ROOT / "input" / "kvch_daily_megion.csv",
        "--visc-csv", ROOT / "input" / "visc_daily_megion.csv",
        "--requested-json", ROOT / "input" / "graph_megion_requested_daily_params.json",
        "--out-dir", STAGE,
    )
    run("create_ascii_aliases.py", "--var-out", STRICT, "--stage1-out", STAGE, "--chem-out", RUN_DIR / "unused_chem_alias")
    run(
        "finalize_prepared_dataset.py",
        "--source", STAGE / "stage1_required_columns.csv",
        "--destination", FINAL / "final_dataset.csv",
        "--chem-daily-csv", ROOT / "input" / "chem_daily_megion.csv",
    )
    run(
        "add_h2s_gas_phase.py",
        "--source", FINAL / "final_dataset.csv",
        "--destination", FINAL / "final_dataset__WITH_H2S_GAS_PHASE.csv",
    )
    run(
        "audit_final_quality_strict.py",
        "--final-csv", FINAL / "final_dataset__WITH_H2S_GAS_PHASE.csv",
        "--report-json", FINAL / "FINAL_QUALITY_STRICT_AUDIT.json",
        "--temperature-min", "5", "--floor-epsilon", "0.05", "--max-floor-share-total", "0.005",
    )
    with LOG.open("a", encoding="utf-8") as log:
        log.write("DONE\n")


if __name__ == "__main__":
    main()
