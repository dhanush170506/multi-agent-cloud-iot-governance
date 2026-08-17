from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class SustainabilityPredictor:
    """Predict energy consumption and operational cost from normalized telemetry."""

    REQUIRED_FEATURES = ["cpu_load", "memory_load", "network_load"]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.energy_model: RandomForestRegressor | None = None
        self.cost_model: RandomForestRegressor | None = None
        self.energy_validation_mae: float | None = None
        self.cost_validation_mae: float | None = None

    def train(self, data: pd.DataFrame) -> None:
        self._validate_features(data)

        features = data[self.REQUIRED_FEATURES]
        energy_kwh = (
            0.5 * data["cpu_load"]
            + 0.3 * data["memory_load"]
            + 0.2 * data["network_load"]
        )
        cost_per_hour = (
            0.6 * energy_kwh
            + 0.2 * data["cpu_load"]
            + 0.2 * data["network_load"]
        )

        x_train, x_val, y_energy_train, y_energy_val, y_cost_train, y_cost_val = train_test_split(
            features,
            energy_kwh,
            cost_per_hour,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.energy_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
        )
        self.cost_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
        )

        self.energy_model.fit(x_train, y_energy_train)
        self.cost_model.fit(x_train, y_cost_train)

        energy_predictions = self.energy_model.predict(x_val)
        cost_predictions = self.cost_model.predict(x_val)

        self.energy_validation_mae = float(mean_absolute_error(y_energy_val, energy_predictions))
        self.cost_validation_mae = float(mean_absolute_error(y_cost_val, cost_predictions))

    def predict(self, data: pd.DataFrame) -> dict:
        self._validate_features(data)
        if self.energy_model is None or self.cost_model is None:
            raise ValueError("Models are not trained or loaded.")

        features = data[self.REQUIRED_FEATURES]
        return {
            "energy_kwh": np.asarray(self.energy_model.predict(features)),
            "cost_per_hour": np.asarray(self.cost_model.predict(features)),
        }

    def save(self, path: str) -> None:
        if self.energy_model is None or self.cost_model is None:
            raise ValueError("Models are not trained.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "energy_model": self.energy_model,
                "cost_model": self.cost_model,
                "energy_validation_mae": self.energy_validation_mae,
                "cost_validation_mae": self.cost_validation_mae,
            },
            model_path,
        )

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.energy_model = payload["energy_model"]
        self.cost_model = payload["cost_model"]
        self.energy_validation_mae = payload.get("energy_validation_mae")
        self.cost_validation_mae = payload.get("cost_validation_mae")

    def _validate_features(self, data: pd.DataFrame) -> None:
        missing_features = [
            feature for feature in self.REQUIRED_FEATURES if feature not in data.columns
        ]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


