from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ai_models.resource.resource_predictor import ResourcePredictor


def generate_synthetic_telemetry(samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "cpu_load": rng.uniform(0, 100, samples),
            "memory_load": rng.uniform(0, 100, samples),
            "network_load": rng.uniform(0, 100, samples),
        }
    )


def main() -> None:
    data = generate_synthetic_telemetry(samples=1000)

    predictor = ResourcePredictor(random_state=42)
    predictor.train(data)

    print(f"Validation MAE: {predictor.validation_mae:.4f}")

    model_path = "data/resource_model.joblib"
    predictor.save(model_path)

    loaded_predictor = ResourcePredictor()
    loaded_predictor.load(model_path)
    loaded_predictor.predict(data.head(5))


if __name__ == "__main__":
    main()
