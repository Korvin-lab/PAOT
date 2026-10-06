"""Sort lightweight offsets into common Monday-Sunday calendar bins."""
import csv
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    with (HERE / "daily_blocks.csv").open() as f:
        blocks = list(csv.DictReader(f))
    assert len(blocks) == 71460 and sum(int(b["rows"]) for b in blocks) == 138583872
    assert len({(b["id"], b["date"]) for b in blocks}) == 71460
    groups = defaultdict(list)
    for block in blocks:
        day = date.fromisoformat(block["date"])
        start = day - timedelta(days=day.weekday())
        block["week"] = start.isoformat()
        block["day_index"] = day.weekday()
        groups[(block["id"], block["week"])].append(block)
    estimated_rows = sum(max(int(b["rows"]) for b in group) for group in groups.values())
    with (HERE / "weekly_plan.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "week", "date", "day_index", "offset", "bytes", "rows"])
        writer.writeheader()
        for key in sorted(groups):
            writer.writerows(sorted(groups[key], key=lambda b: b["date"]))
    result = {"source_rows": 138583872, "source_pipe_dates": len(blocks), "pipes": len({b['id'] for b in blocks}),
              "pipe_weeks": len(groups), "estimated_weekly_rows_if_same_segment_grid": estimated_rows,
              "days_per_week": dict(Counter(len(g) for g in groups.values())),
              "first_week": min(b["week"] for b in blocks), "last_week": max(b["week"] for b in blocks),
              "rule": "calendar weeks Monday-Sunday; mean of available finite daily values per pipe and segment"}
    (HERE / "plan_report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
