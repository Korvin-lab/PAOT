from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("pipeline_under_test", ROOT / "main_pipeline_final_csv.py")
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)

PREFILL_SPEC = importlib.util.spec_from_file_location("prefill_under_test", ROOT / "preprocess_requested_fill_active.py")
prefill = importlib.util.module_from_spec(PREFILL_SPEC)
assert PREFILL_SPEC.loader is not None
PREFILL_SPEC.loader.exec_module(prefill)

ROUGH_SPEC = importlib.util.spec_from_file_location("rough_temperature_under_test", ROOT / "rough_temperature_model.py")
rough_temperature = importlib.util.module_from_spec(ROUGH_SPEC)
assert ROUGH_SPEC.loader is not None
ROUGH_SPEC.loader.exec_module(rough_temperature)


class FakeTemperatureModel:
    def __init__(self, **kwargs):
        self.t_bound = float(kwargs["t_bound"])
        self.direction = int(kwargs["direction"])
        self.length = float(kwargs["length"])

    def run(self):
        if self.direction < 0:
            return lambda distance: self.t_bound + 0.01 * (self.length - float(distance))
        return lambda distance: self.t_bound - 0.01 * float(distance)


def fake_pvt(**kwargs):
    return {
        "rp": 100.0,
        "rho_oil": 836.0,
        "rho_wat": 1000.0,
        "rho_gas": 1.2,
        "muo": 0.01,
        "muw": 0.001,
        "mug": 0.01,
        "q_gas_work_tsd_m3d_polytech": 5.0,
    }


def fake_pe2(**kwargs):
    # 0.01 bar/m, positive frictional loss.
    return 0.5, 0.0, 0.01, 0.0, 0.01, 1, b"mock"


def row(boundary: str, pressure_mpa: float = 1.0, temperature_boundary: str = "start") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "1",
                "date": pd.Timestamp("2020-01-01"),
                "q_oil": 100.0,
                "q_liq": 120.0,
                "q_gas": 5.0,
                "rho_oil": 836.0,
                "rho_wat": 1000.0,
                "rho_gas": 1.2,
                "watercut": 20.0,
                "D": 0.2,
                "S": 6.0,
                "t": 30.0,
                "p": pressure_mpa,
                "pressure_boundary_side": boundary,
                "flow_direction_coef": -1 if boundary == "end" else 1,
                "temperature_boundary_side": temperature_boundary,
                "source_t": "трубный техрежим: T конец" if temperature_boundary == "end" else "трубный техрежим: T начало",
            }
        ]
    )


def run() -> None:
    real_forward = rough_temperature.RoughTemperatureModel(
        length=250.0,
        tid=0.168,
        tir=5e-5,
        q_heat_in_by_degree=10000.0,
        surrounding_temperature=5.0,
        htc=15.0,
        t_bound=30.0,
        direction=1,
    ).run()
    xs = np.arange(0.0, 251.0, 10.0)
    forward_values = np.array([real_forward(x) for x in xs])
    assert np.isclose(forward_values[0], 30.0)
    assert np.all(np.diff(forward_values) < 0), "Forward temperature must change continuously toward ambient"
    assert len(np.unique(np.round(forward_values, 12))) == len(forward_values)

    real_reverse = rough_temperature.RoughTemperatureModel(
        length=250.0,
        tid=0.168,
        tir=5e-5,
        q_heat_in_by_degree=10000.0,
        surrounding_temperature=5.0,
        htc=15.0,
        t_bound=20.0,
        direction=-1,
    ).run()
    reverse_values = np.array([real_reverse(x) for x in xs])
    assert np.isclose(reverse_values[-1], 20.0)
    assert np.all(np.diff(reverse_values) < 0), "Reverse boundary calculation must remain physical in 0..L coordinates"

    pipeline.RoughTemperatureModel = FakeTemperatureModel
    pipeline.calculate_pvt_for_segment = fake_pvt
    pipeline.pe_2_correlation = fake_pe2
    pipeline.append_pe2_trace = lambda _: None
    pipeline.add_polytech_segment_flow_columns = lambda frame: frame
    pipeline.add_reynolds_columns = lambda frame: frame

    start = pipeline.expand_with_profile(row("start"), 25.0, step_m=10)
    end = pipeline.expand_with_profile(row("end"), 25.0, step_m=10)

    np.testing.assert_allclose(start["distance"], [0.0, 10.0, 20.0, 25.0])
    np.testing.assert_allclose(end["distance"], [0.0, 10.0, 20.0, 25.0])
    np.testing.assert_allclose(start["t"], end["t"])
    np.testing.assert_allclose(start["t"], [30.0, 29.9, 29.8, 29.75])
    np.testing.assert_allclose(start["p"], [1.0, 0.99, 0.98, 0.975])
    np.testing.assert_allclose(end["p"], [1.025, 1.015, 1.005, 1.0])
    assert end.loc[end["distance"].idxmax(), "p"] == 1.0
    assert start.loc[start["distance"].idxmin(), "p"] == 1.0

    end_temperature = pipeline.expand_with_profile(row("start", temperature_boundary="end"), 25.0, step_m=10)
    np.testing.assert_allclose(end_temperature["t"], [30.25, 30.15, 30.05, 30.0])
    assert end_temperature["temperature_boundary_side"].eq("end").all()

    guarded = pipeline.expand_with_profile(row("start", pressure_mpa=0.02), 25.0, step_m=10)
    assert bool(guarded["pressure_guard_failed"].all())
    assert guarded["p"].notna().sum() == 2, "PE2 must stop immediately at the first threshold crossing"
    filtered, dropped = pipeline.drop_dates_with_nonpositive_pressure(guarded, "1", "test")
    assert filtered.empty and len(dropped) == 1

    sides = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"]),
            "source_p": ["трубный техрежим: P факт конец", "интерполяция", "P расчет начало"],
        }
    )
    assigned = pipeline.assign_pressure_boundaries(sides)
    assert assigned["pressure_boundary_side"].tolist() == ["end", "end", "start"]
    assert assigned["flow_direction_coef"].tolist() == [-1.0, -1.0, 1.0]

    temperature_sides = pipeline.assign_temperature_boundaries(
        pd.DataFrame({"source_t": ["трубный техрежим: T конец", "ШТР скважин", "аналог месяца"]})
    )
    assert temperature_sides["temperature_boundary_side"].tolist() == ["end", "start", "start"]
    assert temperature_sides["temperature_direction_coef"].tolist() == [-1.0, 1.0, 1.0]

    inherited_temperature_sides = pipeline.assign_temperature_boundaries(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-10"]),
                "source_t": ["трубный техрежим: T конец", "интерполяция", "ШТР скважин"],
            }
        )
    )
    assert inherited_temperature_sides["temperature_boundary_side"].tolist() == ["end", "end", "start"]
    assert inherited_temperature_sides["temperature_boundary_rule"].tolist()[1].startswith("nearest_known_source_t:")

    feasible_end = row("start", temperature_boundary="end")
    feasible_end["temperature_boundary_side"] = "end"
    feasible_marked = pipeline.mark_reverse_temperature_feasibility(feasible_end, 25.0)
    assert bool(feasible_marked.at[0, "temperature_reverse_feasible"])

    infeasible_end = row("start", temperature_boundary="end")
    infeasible_end.loc[0, ["q_oil", "q_liq", "q_gas"]] = [0.01, 0.01, 0.001]
    infeasible_end["temperature_boundary_side"] = "end"
    infeasible_marked = pipeline.mark_reverse_temperature_feasibility(infeasible_end, 5000.0)
    assert not bool(infeasible_marked.at[0, "temperature_reverse_feasible"])
    assert np.isinf(infeasible_marked.at[0, "temperature_reverse_estimated_start_c"])

    assert pipeline.polytech_q_gas_work_m3s(10.0, 1.0, 20.0) is not None

    density_input = {
        "by_id": {
            "1": {
                "daily": [
                    {"Дата": "2020-01-01", "Жидкости, м3/сут (дебит)": 1.0, "Нефти, кг/м3": None},
                    {"Дата": "2020-01-02", "Жидкости, м3/сут (дебит)": 1.0, "Нефти, кг/м3": 836.0},
                    {"Дата": "2020-01-03", "Жидкости, м3/сут (дебит)": 1.0, "Нефти, кг/м3": None},
                ]
            }
        }
    }
    density_output, density_report, _, _ = prefill.process_requested(density_input)
    daily = density_output["by_id"]["1"]["daily"]
    assert daily[0]["Нефти, кг/м3"] == 836.0, "A leading gap must use the positive mean of the same pipe"
    assert daily[2]["Нефти, кг/м3"] == 836.0, "The previous positive density must be carried forward"
    assert density_report["filled_rho_oil_prev_or_pipe_mean"] == 2
    assert density_report["filled_rho_oil_pipe_mean"] == 1
    print("PASS: ascending temperature interpolation, start/end temperature boundaries, infeasible reverse-temperature rejection, reverse pressure, exact endpoint, pressure guard, boundary inheritance, Polytech heat gas, previous/pipe-mean oil-density fill")


if __name__ == "__main__":
    run()
