"""Independently scan the weekly CSV and compare selected whole weeks with pandas."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DAILY = HERE.parent / "megion_kvch_ing_rebuild_2026-10-01/final_dataset__KVCH_ING50.csv"
WEEKLY = HERE / "final_dataset__WEEKLY_7D.partial.csv"


class HashReader(io.RawIOBase):
    def __init__(self, path):
        self.handle = path.open("rb", buffering=0)
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.handle.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count

    def close(self):
        self.handle.close()
        super().close()


def main():
    weekly_path = WEEKLY if WEEKLY.exists() else HERE / "final_dataset__WEEKLY_7D.csv"
    report = json.loads((HERE / "weekly_report.json").read_text())
    plan = pd.read_csv(HERE / "weekly_plan.csv", dtype={"id": str, "date": str, "week": str})
    groups = pd.read_csv(HERE / "weekly_groups.csv", dtype={"id": str, "week": str})
    with DAILY.open("rb") as f:
        header = f.readline()
    columns = header.decode("utf-8-sig").strip().split(",")
    assert len(columns) == 50 and pd.read_csv(weekly_path, nrows=0).columns.tolist() == columns
    numerical = [c for c in columns if c not in {"id", "date", "segment_id"}]
    assert len(numerical) == 47
    source_bounds = {r["name"]: r for r in report["columns"]}
    assert set(source_bounds) == set(numerical)
    assert len(groups) == report["pipe_weeks"] == 11357
    assert groups.input_rows.sum() == report["input_rows"] == 138583872
    assert groups.segments.sum() == report["output_rows"]
    assert groups.dates.sum() == len(plan) == 71460
    assert not groups.duplicated(["id", "week"]).any()
    assert groups.min_days_per_segment.between(1, 7).all()
    assert groups.max_days_per_segment.between(1, 7).all()
    samples = set()
    for criterion in [groups.dates.eq(1), groups.dates.eq(7), groups.dates.eq(4),
                      groups.kvch_rows.gt(0), groups.ing_rows.gt(0)]:
        row = groups.loc[criterion].iloc[0]
        samples.add((row.id, row.week))
    row = groups.loc[groups.segments.idxmax()]; samples.add((row.id, row.week))
    samples.add((groups.iloc[-1].id, groups.iloc[-1].week))
    last_source_block = plan.loc[plan.offset.idxmax()]
    samples.add((last_source_block.id, last_source_block.week))
    sample_rows = defaultdict(list)
    counts, missing, zeros, sums = Counter(), Counter(), Counter(), Counter()
    minima, maxima = {}, {}
    week_counts = Counter()
    last = None; rows = 0
    reader = HashReader(weekly_path)
    stream = io.BufferedReader(reader, buffer_size=1024*1024)
    dtypes = {c: "float64" for c in columns if c not in {"id", "date"}}
    dtypes.update(id="string", date="string")
    for chunk_no, chunk in enumerate(pd.read_csv(stream, chunksize=100_000,
                                                dtype=dtypes, low_memory=False, float_precision="round_trip"), 1):
        ids = chunk.id.to_numpy(dtype=str)
        days = chunk.date.to_numpy(dtype=str)
        segments = chunk.segment_id.to_numpy(dtype=float)
        assert chunk.id.notna().all() and chunk.date.notna().all() and np.isfinite(segments).all()
        assert pd.to_datetime(chunk.date, format="%Y-%m-%d").dt.weekday.eq(0).all()
        first_key = (ids[0], days[0], segments[0])
        if last is not None:
            assert first_key > last, (first_key, last)
        same_id = ids[1:] == ids[:-1]
        same_week = same_id & (days[1:] == days[:-1])
        assert not ((ids[1:] < ids[:-1]) | (same_id & (days[1:] < days[:-1])) |
                    (same_week & (segments[1:] <= segments[:-1]))).any()
        last = (ids[-1], days[-1], segments[-1])
        for name in numerical:
            values = chunk[name].to_numpy(dtype=float)
            assert not np.isinf(values).any(), name
            finite = values[np.isfinite(values)]
            counts[name] += len(finite); missing[name] += len(values)-len(finite)
            zeros[name] += int(np.count_nonzero(finite == 0))
            bound = source_bounds[name]
            if finite.size:
                sums[name] += float(finite.sum())
                minima[name] = min(minima.get(name, float("inf")), float(finite.min()))
                maxima[name] = max(maxima.get(name, float("-inf")), float(finite.max()))
                tol = max(1, abs(bound["source_min"]), abs(bound["source_max"])) * 2e-12
                assert finite.min() >= bound["source_min"]-tol and finite.max() <= bound["source_max"]+tol, name
        week_counts.update(chunk.groupby(["id", "date"]).size().to_dict())
        for key in samples:
            selected = chunk.loc[chunk.id.eq(key[0]) & chunk.date.eq(key[1])]
            if not selected.empty:
                sample_rows[key].append(selected)
        rows += len(chunk)
        if chunk_no % 20 == 0:
            print(f"verified weekly rows={rows}", flush=True)
    assert rows == report["output_rows"]
    assert reader.digest.hexdigest() == report["sha256"]
    stream.close()
    expected_week_counts = {(r.id, r.week): int(r.segments) for r in groups.itertuples()}
    assert dict(week_counts) == expected_week_counts
    for name in numerical:
        assert counts[name] == source_bounds[name]["weekly_count"]
        assert missing[name] == source_bounds[name]["weekly_missing"]

    with DAILY.open("rb") as source:
        for pipe, week in sorted(samples):
            frames = []
            for block in plan.loc[plan.id.eq(pipe) & plan.week.eq(week)].itertuples():
                source.seek(block.offset)
                raw = source.read(block.bytes)
                frame = pd.read_csv(io.BytesIO(header+raw), dtype={"id": str}, float_precision="round_trip")
                assert len(frame) == block.rows
                frames.append(frame)
            daily = pd.concat(frames, ignore_index=True)
            expected = daily.groupby("segment_id", as_index=False)[numerical].mean().sort_values("segment_id")
            actual = pd.concat(sample_rows[(pipe, week)], ignore_index=True).sort_values("segment_id")
            np.testing.assert_allclose(actual.segment_id.to_numpy(), expected.segment_id.to_numpy(), rtol=0, atol=0)
            np.testing.assert_allclose(actual[numerical].to_numpy(), expected[numerical].to_numpy(), rtol=2e-12, atol=1e-12, equal_nan=True)
    summary = {"status": "PASS", "weekly_rows": rows, "columns": len(columns), "pipes": groups.id.nunique(),
               "pipe_weeks": len(groups), "pandas_whole_week_checks": [list(k) for k in sorted(samples)],
               "sha256": report["sha256"], "finite_counts": dict(counts), "missing": dict(missing), "zeros": dict(zeros)}
    summary["numeric_summary"] = {name: {"count": counts[name], "mean": sums[name]/counts[name] if counts[name] else None,
                                         "min": minima.get(name), "max": maxima.get(name)} for name in numerical}
    summary["pipes"] = int(summary["pipes"])
    (HERE / "independent_weekly_verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for pipe, subset in groups.groupby("id"):
        rows.append({"ID простого участка": pipe, "Рабочих дней в исходном CSV": int(subset.dates.sum()),
                     "Недель с данными": len(subset), "Полных недель с 7 днями": int(subset.dates.eq(7).sum()),
                     "Неполных недель": int(subset.dates.lt(7).sum()), "Сегментных строк после усреднения": int(subset.segments.sum()),
                     "Недельных сегментных строк с КВЧ": int(subset.kvch_rows.sum()),
                     "Недельных сегментных строк с ингибированием": int(subset.ing_rows.sum())})
    pd.DataFrame(rows).to_csv(HERE / "Сводка_по_трубам_7_дней.csv", index=False, encoding="utf-8-sig")
    print("PASS full weekly verification and independent pandas week means", flush=True)


if __name__ == "__main__":
    main()
