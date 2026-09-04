from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ai_models.security.threat_detector import ThreatDetector


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


def load_real_telemetry() -> tuple[pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parents[2]
    source_path = project_root / "data" / "real_security" / "df_Binary_FL_CICIoT2023.csv"
    data = pd.read_csv(source_path).drop(columns=["Unnamed: 0"])

    missing = [column for column in FEATURES + ["sub_label"] if column not in data]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=FEATURES + ["sub_label"])
    benign_data = data[data["sub_label"] == 0][FEATURES]
    attack_data = data[data["sub_label"] != 0][FEATURES]

    training_data = benign_data.sample(n=min(len(benign_data), 50000), random_state=42)
    evaluation_data = pd.concat(
        [benign_data.sample(n=min(len(benign_data), 10000), random_state=42),
         attack_data.sample(n=min(len(attack_data), 10000), random_state=42)],
        ignore_index=True,
    )
    return training_data.reset_index(drop=True), evaluation_data.reset_index(drop=True)


def main() -> None:
    normal_data, evaluation_data = load_real_telemetry()

    detector = ThreatDetector(random_state=42)
    detector.train(normal_data)

    normal_count = len(normal_data)
    evaluation_normal_count = min(len(normal_data), 10000)
    attack_count = len(evaluation_data) - evaluation_normal_count
    anomaly_scores = detector.predict_anomaly(evaluation_data)
    detected_anomalies = int((anomaly_scores > 0.5).sum())
    detected_attack_anomalies = int((anomaly_scores[evaluation_normal_count:] > 0.5).sum())
    attack_detection_rate = detected_attack_anomalies / attack_count if attack_count else 0.0

    print(f"Benign training samples: {normal_count}")
    print(f"Benign evaluation samples: {evaluation_normal_count}")
    print(f"Attack evaluation samples: {attack_count}")
    print(f"Detected evaluation anomalies: {detected_anomalies}")
    print(f"Attack anomaly detection rate: {attack_detection_rate:.4f}")

    model_path = Path(__file__).resolve().parents[2] / "data" / "threat_detector.joblib"
    detector.save(str(model_path))

    loaded_detector = ThreatDetector()
    loaded_detector.load(str(model_path))
    loaded_detector.predict_anomaly(evaluation_data.head(5))


if __name__ == "__main__":
    main()
