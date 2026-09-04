from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


class ResourcePredictor:
    """Predict resource pressure from telemetry load metrics."""

    FEATURES = ["cpu_load", "memory_load", "network_load"]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
        )
        self.validation_mae: float | None = None

    def train(self, data: pd.DataFrame) -> None:
        self._validate_features(data)

        x = data[self.FEATURES]
        y = x.mean(axis=1)

        x_train, x_validation, y_train, y_validation = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=self.random_state,
        )

        self.model.fit(x_train, y_train)
        predictions = self.model.predict(x_validation)
        self.validation_mae = float(mean_absolute_error(y_validation, predictions))

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        self._validate_features(data)
        return self.model.predict(data[self.FEATURES])

    def save(self, path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)

    def load(self, path: str) -> None:
        self.model = joblib.load(path)

    def _validate_features(self, data: pd.DataFrame) -> None:
        missing = [feature for feature in self.FEATURES if feature not in data.columns]
        if missing:
            raise ValueError(f"Missing required telemetry columns: {missing}")
