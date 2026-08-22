from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SustainabilityPredictor:
    """Predict energy consumption and operational cost from normalized telemetry."""

    REQUIRED_FEATURES = ["cpu_load", "memory_load", "network_load"]

    def __init__(self, random_state: int = 42, electricity_price_per_kwh: float = 0.12) -> None:
        self.random_state = random_state
        self.electricity_price_per_kwh = electricity_price_per_kwh
        self.energy_model: RandomForestRegressor | None = None
        self.feature_scaler = StandardScaler()
        self.energy_validation_mae: float | None = None
        self.cost_validation_mae: float | None = None

    def train(self, data: pd.DataFrame) -> None:
        self._validate_features(data)

        features = data[self.REQUIRED_FEATURES]
        if "power_consumption" not in data.columns:
            raise ValueError("Missing required target: power_consumption")

        energy_kwh = data["power_consumption"]

        x_train, x_val, y_energy_train, y_energy_val = train_test_split(
            features,
            energy_kwh,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.energy_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state,
        )
        x_train_scaled = self.feature_scaler.fit_transform(x_train)
        x_val_scaled = self.feature_scaler.transform(x_val)

        self.energy_model.fit(x_train_scaled, y_energy_train)

        energy_predictions = self.energy_model.predict(x_val_scaled)

        self.energy_validation_mae = float(mean_absolute_error(y_energy_val, energy_predictions))
        self.cost_validation_mae = None

    def predict(self, data: pd.DataFrame) -> dict:
        self._validate_features(data)
        if self.energy_model is None:
            raise ValueError("Models are not trained or loaded.")

        features = data[self.REQUIRED_FEATURES]
        features_scaled = self.feature_scaler.transform(features)
        energy_predictions = np.asarray(self.energy_model.predict(features_scaled))
        return {
            "energy_kwh": energy_predictions,
            "cost_per_hour": energy_predictions * self.electricity_price_per_kwh,
        }

    def save(self, path: str) -> None:
        if self.energy_model is None:
            raise ValueError("Models are not trained.")

        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "energy_model": self.energy_model,
                "feature_scaler": self.feature_scaler,
                "electricity_price_per_kwh": self.electricity_price_per_kwh,
                "energy_validation_mae": self.energy_validation_mae,
                "cost_validation_mae": self.cost_validation_mae,
            },
            model_path,
        )

    def load(self, path: str) -> None:
        payload = joblib.load(path)
        self.energy_model = payload["energy_model"]
        self.feature_scaler = payload.get("feature_scaler", StandardScaler())
        self.electricity_price_per_kwh = payload.get(
            "electricity_price_per_kwh", self.electricity_price_per_kwh
        )
        self.energy_validation_mae = payload.get("energy_validation_mae")
        self.cost_validation_mae = payload.get("cost_validation_mae")

    def _validate_features(self, data: pd.DataFrame) -> None:
        missing_features = [
            feature for feature in self.REQUIRED_FEATURES if feature not in data.columns
        ]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


