"""Build direction-linked daily mineralization without an untraceable global placeholder."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
PACKAGE = WORK / "MEGION_WINDOWS_FULL_PE2__PREPARING_2026-09-18"
SOURCE = HERE / "ukk_mineralization_mapped.json"


def main():
    rows = json.loads(SOURCE.read_text())
    assert len(rows) == 395
    raw = np.array([x["raw"] for x in rows], dtype=float)
    # The same mg/L-labeled column contains two non-overlapping scales.
    # Nearby samples for the same UKK differ by ~1000 before normalization.
    assert ((raw < 100) | (raw > 1000)).all()
    normalized = [
        (x["direction"], x["ukk"], x["date"],
         x["raw"] if x["raw"] < 100 else x["raw"] / 1000)
        for x in rows
    ]
    dedup = pd.DataFrame(normalized, columns=["direction", "ukk", "date", "Min_g_l"])
    before = len(dedup)
    dedup = dedup.drop_duplicates(["direction", "ukk", "date", "Min_g_l"])
    dedup["date"] = pd.to_datetime(dedup["date"], format="%Y-%m-%d")
    measurements = dedup.groupby(["direction", "date"], as_index=False)["Min_g_l"].median()
    assert measurements.direction.nunique() == 4 and len(measurements) == 142

    prepared = pd.read_csv(PACKAGE / "input/chem_daily_megion.csv",
                           usecols=["id простого участка", "Дата контроля", "Общая минерализация, г/л"],
                           dtype={"id простого участка": str, "Дата контроля": str})
    prepared = prepared.rename(columns={"id простого участка": "id", "Дата контроля": "date"})
    assert len(prepared) == 99212 and prepared.id.nunique() == 66
    assert prepared["Общая минерализация, г/л"].eq(22.6627).all()
    prepared["date"] = pd.to_datetime(prepared.date, format="%Y-%m-%d")
    assert not prepared.duplicated(["id", "date"]).any()
    all_dates = pd.DatetimeIndex(sorted(prepared.date.unique()))

    direction_by_pipe = {}
    for x in rows:
        for pid in x["pipes"]:
            if pid in direction_by_pipe and direction_by_pipe[pid] != x["direction"]:
                raise AssertionError("Conflicting direction for " + pid)
            direction_by_pipe[pid] = x["direction"]
    assert len(direction_by_pipe) == 12

    donor_curves = {}
    for direction, part in measurements.groupby("direction"):
        s = part.set_index("date")["Min_g_l"].sort_index()
        assert not s.index.duplicated().any() and s.gt(0).all()
        full = s.reindex(all_dates.union(s.index).sort_values())
        filled = full.interpolate(method="time").bfill().ffill().reindex(all_dates)
        assert filled.notna().all() and filled.gt(0).all()
        # Every observed point is kept exactly; ends equal nearest observed value.
        assert np.allclose(full.loc[s.index].to_numpy(), s.to_numpy(), rtol=0, atol=1e-10)
        assert np.isclose(filled.iloc[0], s.iloc[0]) if all_dates[0] <= s.index[0] else True
        donor_curves[direction] = filled

    donor_ids = sorted(direction_by_pipe)
    donor_matrix = np.column_stack([donor_curves[direction_by_pipe[pid]].to_numpy() for pid in donor_ids])
    median_by_day = pd.Series(np.median(donor_matrix, axis=1), index=all_dates)
    assert median_by_day.notna().all() and median_by_day.gt(0).all()
    direct = pd.Series(prepared.id.map(direction_by_pipe).to_numpy(), index=prepared.index)
    output = median_by_day.reindex(prepared.date).to_numpy().copy()
    for direction in donor_curves:
        mask = direct.eq(direction).to_numpy()
        output[mask] = donor_curves[direction].reindex(prepared.date[mask]).to_numpy()
    assert np.isfinite(output).all() and (output > 0).all()
    assert output.min() >= 10 and output.max() <= 60
    prepared["Min"] = output

    requested = json.loads((PACKAGE / "input/graph_megion_requested_daily_params.json").read_text())["by_id"]
    flows = ("Жидкости, м3/сут (дебит)", "Нефти, т/сут (дебит)", "Общего газа, тыс.м3/сут (дебит)")
    active = {(pid, pd.Timestamp(row["Дата"][:10]))
              for pid, payload in requested.items() for row in payload["daily"]
              if any(float(row.get(col) or 0) > 0 for col in flows)}
    assert len(active) == 71460
    is_active = np.fromiter(((pid, dt) in active for pid, dt in zip(prepared.id, prepared.date)),
                            dtype=bool, count=len(prepared))
    assert int(is_active.sum()) == len(active)
    daily = prepared[["id", "date", "Min"]].copy()
    daily["date"] = daily.date.dt.strftime("%Y-%m-%d")
    daily.to_csv(HERE / "daily_min_all_prepared_dates.csv", index=False)
    daily.loc[is_active].to_csv(HERE / "daily_min_active_patch.csv", index=False)

    changed = ~np.isclose(output, 22.6627, rtol=1e-12, atol=1e-10)
    summary = {
        "raw_ukk_rows_mapped": before,
        "mixed_scale_raw_lt_100": int((raw < 100).sum()),
        "mixed_scale_raw_gt_1000": int((raw > 1000).sum()),
        "unique_after_exact_dedup": len(dedup),
        "distinct_direction_dates": len(measurements),
        "donor_directions": len(donor_curves),
        "donor_pipes": len(donor_ids),
        "daily_median_pipes": prepared.id.nunique() - len(donor_ids),
        "daily_median_min_g_l": float(median_by_day.min()),
        "daily_median_max_g_l": float(median_by_day.max()),
        "all_prepared_pipe_dates": len(prepared),
        "active_pipe_dates": int(is_active.sum()),
        "changed_prepared_pipe_dates": int(changed.sum()),
        "changed_active_pipe_dates": int((changed & is_active).sum()),
        "new_min_g_l": float(output.min()),
        "new_median_g_l": float(np.median(output)),
        "new_max_g_l": float(output.max()),
        "new_unique_rounded_6": int(np.unique(np.round(output, 6)).size),
        "measurement_period": [str(measurements.date.min().date()), str(measurements.date.max().date())],
        "unit_inference": "Raw <100 treated as g/L; raw >1000 divided by 1000 from mg/L. Headers alone do not document this switch; inferred from a clean scale gap and same-UKK adjacent samples.",
    }
    (HERE / "daily_min_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
