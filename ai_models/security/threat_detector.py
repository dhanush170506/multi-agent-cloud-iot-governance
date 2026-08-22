from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class ThreatDetector:
    """Detect abnormal cloud-IoT telemetry and compute threat severity."""

    FEATURES = [
        "rst_count",
        "ICMP",
        "Min",
        "AVG",
        "IAT",
        "Number",
        "Variance",
        "Weight",
    ]

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=0.05,
            random_state=random_state,
        )

    def train(self, data: pd.DataFrame) -> None:
        self._validate_features(data)
        self.model.fit(data[self.FEATURES])

    def predict_anomaly(self, data: pd.DataFrame) -> np.ndarray:
        self._validate_features(data)
        raw_scores = -self.model.decision_function(data[self.FEATURES])
        min_score = float(raw_scores.min())
        max_score = float(raw_scores.max())

        if max_score == min_score:
            return np.zeros(len(raw_scores), dtype=float)

        normalized_scores = (raw_scores - min_score) / (max_score - min_score)
        return np.clip(normalized_scores, 0, 1)

    def compute_tss(self, data: pd.DataFrame) -> np.ndarray:
        anomaly_score = self.predict_anomaly(data)
        if "trust_score" not in data:
            return anomaly_score

        trust_degradation = 1 - data["trust_score"].to_numpy()
        tss = 0.7 * anomaly_score + 0.3 * trust_degradation
        return np.clip(tss, 0, 1)

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
