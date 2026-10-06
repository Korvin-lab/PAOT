from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def find_one(base: Path, predicate) -> Path | None:
    for p in base.glob("*"):
        if p.is_file() and predicate(p.name):
            return p
    return None


def copy_if_found(base: Path, pred, dst_name: str) -> bool:
    src = find_one(base, pred)
    if not src:
        return False
    dst = base / dst_name
    shutil.copy2(src, dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--var-out", type=Path, required=True)
    ap.add_argument("--stage1-out", type=Path, required=True)
    ap.add_argument("--chem-out", type=Path, required=True)
    args = ap.parse_args()

    var_out = args.var_out
    stage1_out = args.stage1_out
    chem_out = args.chem_out

    if not var_out.exists():
        print(f"ERROR: var_out not found: {var_out}")
        return 2

    # Strict aliases
    copy_if_found(
        var_out,
        lambda n: "сегментов" in n and "все_трубы" in n and n.lower().endswith(".csv"),
        "segments_all_pipes.csv",
    )
    copy_if_found(
        var_out,
        lambda n: "сегментов" in n and "статистика_по_трубам" in n and n.lower().endswith(".csv"),
        "segments_stats_by_pipe.csv",
    )
    copy_if_found(
        var_out,
        lambda n: "сегментов" in n and "уход_в_ноль" in n and n.lower().endswith(".csv"),
        "segments_zero.csv",
    )
    copy_if_found(
        var_out,
        lambda n: "углы__проверка" in n and n.lower().endswith(".csv"),
        "angles_check.csv",
    )
    copy_if_found(
        var_out,
        lambda n: "пересчет__сводка" in n and n.lower().endswith(".json"),
        "summary_ru.json",
    )

    # Stage1 aliases
    if stage1_out.exists():
        copy_if_found(
            stage1_out,
            lambda n: "__stage1__required_columns.csv" in n.lower(),
            "stage1_required_columns.csv",
        )
        copy_if_found(
            stage1_out,
            lambda n: "__stage1__full.csv" in n.lower(),
            "stage1_full.csv",
        )
        copy_if_found(
            stage1_out,
            lambda n: "__stage1__id_date_chem_co2.csv" in n.lower(),
            "stage1_id_date_chem_co2.csv",
        )
        copy_if_found(
            stage1_out,
            lambda n: "__stage1__report.json" in n.lower(),
            "stage1_report.json",
        )

    # Chem aliases
    if chem_out.exists():
        copy_if_found(
            chem_out,
            lambda n: "__stage1__required_columns__filled.csv" in n.lower(),
            "stage1_required_columns_filled.csv",
        )
        copy_if_found(
            chem_out,
            lambda n: "__stage1__id_date_chem_co2__filled.csv" in n.lower(),
            "stage1_id_date_chem_co2_filled.csv",
        )
        copy_if_found(
            chem_out,
            lambda n: "chem_fill_report_variant3.json" in n.lower(),
            "chem_fill_report_variant3.json",
        )

    # Guard: strict segments file is mandatory for full chain.
    if not (var_out / "segments_all_pipes.csv").exists():
        print("ERROR: strict output segments file not found for ASCII alias.")
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
