from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ai_models.resource.resource_predictor import ResourcePredictor


def load_real_telemetry() -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "data" / "real_resource" / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
    raw_data = pd.read_csv(source_path)

    required_columns = [
        "cpu_util_percent",
        "mem_util_percent",
        "net_in",
        "net_out",
    ]
    missing_columns = [column for column in required_columns if column not in raw_data]
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")

    network_traffic = raw_data["net_in"] + raw_data["net_out"]
    network_scale = max(float(network_traffic.max()), 1.0)

    telemetry = pd.DataFrame(
        {
            "cpu_load": raw_data["cpu_util_percent"] / 100.0,
            "memory_load": raw_data["mem_util_percent"] / 100.0,
            "network_load": network_traffic / network_scale,
        }
    )
    return telemetry.dropna().reset_index(drop=True)


def main() -> None:
    data = load_real_telemetry()

    predictor = ResourcePredictor(random_state=42)
    predictor.train(data)

    print(f"Validation MAE: {predictor.validation_mae:.4f}")

    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "data" / "resource_model.joblib"
    predictor.save(str(model_path))

    loaded_predictor = ResourcePredictor()
    loaded_predictor.load(str(model_path))
    loaded_predictor.predict(data.head(5))


if __name__ == "__main__":
    main()
