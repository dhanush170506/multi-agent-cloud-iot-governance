from __future__ import annotations

from pathlib import Path

import ast
import pandas as pd

try:
    from .sustainability_predictor import SustainabilityPredictor
except ImportError:
    from sustainability_predictor import SustainabilityPredictor


def _mean_measurement(value: object) -> float:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if hasattr(value, "__iter__") and not isinstance(value, (bytes, dict)):
        values = pd.to_numeric(pd.Series(value), errors="coerce").dropna()
        return float(values.mean()) if not values.empty else float("nan")
    return float(value)


def load_real_workloads() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "data" / "real_sustainability" / "job_table.parquet"
    raw_data = pd.read_parquet(source_path)

    data = pd.DataFrame(
        {
            "cpu_load": raw_data["num_cores_alloc"],
            "memory_load": raw_data["mem_alloc"],
            "network_load": raw_data["run_time"],
            "power_consumption": raw_data["power_consumption"].map(_mean_measurement),
        }
    )
    return data.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)


def main() -> None:
    telemetry = load_real_workloads()

    predictor = SustainabilityPredictor(electricity_price_per_kwh=0.12)
    predictor.train(telemetry)

    print(f"Energy validation MAE: {predictor.energy_validation_mae}")

    model_path = Path(__file__).resolve().parents[2] / "data" / "sustainability_model.joblib"
    predictor.save(str(model_path))

    loaded_predictor = SustainabilityPredictor()
    loaded_predictor.load(str(model_path))

    samples = pd.DataFrame(
        {
            "cpu_load": [32, 64, 128],
            "memory_load": [1024, 4096, 8192],
            "network_load": [3600, 18000, 86400],
        }
    )
    predictions = loaded_predictor.predict(samples)
    print(f"Sample energy predictions: {predictions['energy_kwh']}")
    print(f"Sample estimated cost predictions: {predictions['cost_per_hour']}")


if __name__ == "__main__":
    main()
