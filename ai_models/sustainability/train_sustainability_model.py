from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .sustainability_predictor import SustainabilityPredictor
except ImportError:
    from sustainability_predictor import SustainabilityPredictor


def generate_synthetic_telemetry(sample_count: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "cpu_load": rng.random(sample_count),
            "memory_load": rng.random(sample_count),
            "network_load": rng.random(sample_count),
        }
    )


def main() -> None:
    telemetry = generate_synthetic_telemetry(1000)

    predictor = SustainabilityPredictor()
    predictor.train(telemetry)

    print(f"Energy validation MAE: {predictor.energy_validation_mae}")
    print(f"Cost validation MAE: {predictor.cost_validation_mae}")

    model_path = Path("data/sustainability_model.joblib")
    predictor.save(str(model_path))

    loaded_predictor = SustainabilityPredictor()
    loaded_predictor.load(str(model_path))

    samples = pd.DataFrame(
        {
            "cpu_load": [0.2, 0.5, 0.8],
            "memory_load": [0.3, 0.6, 0.4],
            "network_load": [0.1, 0.7, 0.9],
        }
    )
    predictions = loaded_predictor.predict(samples)
    print(f"Energy predictions: {predictions['energy_kwh']}")
    print(f"Cost predictions: {predictions['cost_per_hour']}")


if __name__ == "__main__":
    main()
