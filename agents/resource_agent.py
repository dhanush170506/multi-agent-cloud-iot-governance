from pathlib import Path

import pandas as pd

from ai_models.resource.resource_predictor import ResourcePredictor


MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "resource_model.joblib"
_resource_predictor: ResourcePredictor | None = None


def _get_resource_predictor() -> ResourcePredictor:
    global _resource_predictor
    if _resource_predictor is None:
        _resource_predictor = ResourcePredictor()
        _resource_predictor.load(str(MODEL_PATH))
    return _resource_predictor


def _metric(twin_state: dict, average_key: str, raw_key: str) -> float:
    if average_key in twin_state:
        return float(twin_state[average_key])
    value = twin_state[raw_key]
    if hasattr(value, "mean"):
        return float(value.mean())
    return float(value)


def _predicted_telemetry(action: dict, twin_state: dict) -> pd.DataFrame:
    predicted_effect = action.get("predicted_effect", {})
    cpu_load = _metric(twin_state, "avg_cpu_load", "cpu_load")
    memory_load = _metric(twin_state, "avg_memory_load", "memory_load")
    network_load = _metric(twin_state, "avg_network_load", "network_load")

    return pd.DataFrame(
        [
            {
                "cpu_load": min(max(cpu_load + predicted_effect.get("cpu_load", 0), 0), 1),
                "memory_load": min(
                    max(memory_load + predicted_effect.get("memory_load", 0), 0),
                    1,
                ),
                "network_load": min(
                    max(network_load + predicted_effect.get("network_load", 0), 0),
                    1,
                ),
            }
        ]
    )


def score_resource(action: dict, twin_state: dict) -> float:
    predictor = _get_resource_predictor()
    telemetry = _predicted_telemetry(action, twin_state)
    resource_pressure = float(predictor.predict(telemetry)[0])

    utility = 1 - resource_pressure
    return float(min(max(utility, 0), 1))
