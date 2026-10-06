"""Compare the streaming aggregator with pandas on partial weeks and NaN/zero cases."""
import csv
import json
import subprocess
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "megion_kvch_ing_rebuild_2026-10-01/final_dataset__KVCH_ING50.csv"


def main():
    with SOURCE.open(encoding="utf-8-sig") as f:
        columns = next(csv.reader(f))
    numeric = [c for c in columns if c not in {"date", "id", "segment_id"}]
    keys = [("2026-10-01", "1750004707", 0), ("2026-10-01", "1750004707", 10),
            ("2026-09-29", "1750004707", 0), ("2026-10-05", "1750004707", 0),
            ("2026-10-05", "1750004707", 10), ("2026-09-29", "1750004708", 0),
            ("2026-09-30", "1750004708", 0), ("2026-10-01", "1750004708", 0)]
    rows = []
    for i, (day, pipe, segment) in enumerate(keys):
        record = {c: float(i + 1) for c in numeric}
        record.update(date=day, id=pipe, segment_id=segment)
        record["H2S in Water Phase"] = np.nan
        record["kvch"] = np.nan if i == 2 else 20 + i
        record["ing_factor"] = 0 if i % 2 == 0 else .5
        rows.append(record)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        source = folder / "source.csv"
        pd.DataFrame(rows)[columns].to_csv(source, index=False, encoding="utf-8-sig")
        blocks = []
        with source.open("rb") as f:
            f.readline()
            while True:
                offset = f.tell(); line = f.readline()
                if not line: break
                day, pipe = line.decode().split(",")[:2]
                d = date.fromisoformat(day); week = (d-timedelta(days=d.weekday())).isoformat()
                blocks.append([pipe, week, day, d.weekday(), offset, len(line), 1])
        plan = folder / "plan.csv"
        with plan.open("w", newline="") as f:
            writer = csv.writer(f);writer.writerow(["id", "week", "date", "day_index", "offset", "bytes", "rows"])
            writer.writerows(sorted(blocks))
        exe = folder / "weekly_mean"
        subprocess.run(["clang", "-O3", "-Wno-deprecated-declarations", "-DEXPECTED_INPUT=8",
                        str(HERE / "weekly_mean.c"), "-o", str(exe)], check=True)
        output, report = folder / "weekly.csv", folder / "report.json"
        subprocess.run([str(exe), str(source), str(plan), str(output), str(folder / "coverage.csv"), str(report)], check=True)
        actual = pd.read_csv(output, dtype={"id": str})
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame.date)
        frame["date"] = (frame.date-pd.to_timedelta(frame.date.dt.weekday, unit="D")).dt.strftime("%Y-%m-%d")
        expected = frame.groupby(["id", "date", "segment_id"], as_index=False)[numeric].mean()
        actual = actual.sort_values(["id", "date", "segment_id"]).reset_index(drop=True)
        expected = expected.sort_values(["id", "date", "segment_id"]).reset_index(drop=True)
        assert actual[["id", "date", "segment_id"]].equals(expected[["id", "date", "segment_id"]])
        np.testing.assert_allclose(actual[numeric].to_numpy(), expected[numeric].to_numpy(), rtol=1e-14, equal_nan=True)
        assert json.loads(report.read_text())["input_rows"] == 8
    print("PASS pandas comparison: out-of-order source, two weeks, two pipes, variable segments, missing values and zero dose")


if __name__ == "__main__":
    main()
