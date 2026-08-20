from pathlib import Path

import pandas as pd

from ai_models.sustainability.sustainability_predictor import SustainabilityPredictor


_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "sustainability_model.joblib"
_SUSTAINABILITY_PREDICTOR = None


def _get_sustainability_predictor() -> SustainabilityPredictor:
    global _SUSTAINABILITY_PREDICTOR

    if _SUSTAINABILITY_PREDICTOR is None:
        predictor = SustainabilityPredictor()
        predictor.load(str(_MODEL_PATH))
        _SUSTAINABILITY_PREDICTOR = predictor

    return _SUSTAINABILITY_PREDICTOR


def _clip01(value: float) -> float:
    return float(min(max(value, 0), 1))


def _telemetry_frame(twin_state, action=None) -> pd.DataFrame:
    if isinstance(twin_state, pd.DataFrame):
        telemetry = twin_state.copy()
    else:
        telemetry = pd.DataFrame(
            [
                {
                    "cpu_load": twin_state.get("cpu_load", twin_state.get("avg_cpu_load", 0)),
                    "memory_load": twin_state.get(
                        "memory_load",
                        twin_state.get("avg_memory_load", 0),
                    ),
                    "network_load": twin_state.get(
                        "network_load",
                        twin_state.get("avg_network_load", 0),
                    ),
                }
            ]
        )

    if action:
        predicted_effect = action.get("predicted_effect", {})
        for feature in SustainabilityPredictor.REQUIRED_FEATURES:
            if feature in predicted_effect:
                telemetry[feature] = (telemetry[feature] + predicted_effect[feature]).clip(0, 1)

    return telemetry[SustainabilityPredictor.REQUIRED_FEATURES]


def _prediction_totals(telemetry: pd.DataFrame) -> tuple[float, float]:
    predictions = _get_sustainability_predictor().predict(telemetry)
    energy_kwh = float(predictions["energy_kwh"].sum())
    cost_per_hour = float(predictions["cost_per_hour"].sum())

    return energy_kwh, cost_per_hour


def score_sustainability(action, twin_state) -> float:
    telemetry_after = _telemetry_frame(twin_state, action)
    predicted_energy_after, predicted_cost_after = _prediction_totals(telemetry_after)

    current_cost = twin_state["total_cost_per_hour"]
    current_energy = twin_state["total_energy_kwh"]
    predicted_cost_after_normalized = (
        predicted_cost_after / current_cost
        if current_cost
        else predicted_cost_after
    )
    predicted_energy_after_normalized = (
        predicted_energy_after / current_energy
        if current_energy
        else predicted_energy_after
    )

    utility = 1 - (
        predicted_cost_after_normalized +
        predicted_energy_after_normalized
    ) / 2

    return _clip01(utility)
