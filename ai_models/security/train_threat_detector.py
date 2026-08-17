from __future__ import annotations

from pathlib import Path
import sys

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ai_models.security.threat_detector import ThreatDetector


def generate_normal_telemetry(samples: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "cpu_load": rng.uniform(0.2, 0.7, samples),
            "network_load": rng.uniform(0.1, 0.6, samples),
            "trust_score": rng.uniform(0.7, 1.0, samples),
            "connection_rate": rng.uniform(0.1, 0.5, samples),
        }
    )


def generate_attack_telemetry(samples: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "cpu_load": rng.uniform(0.55, 0.95, samples),
            "network_load": rng.uniform(0.75, 1.0, samples),
            "trust_score": rng.uniform(0.05, 0.45, samples),
            "connection_rate": rng.uniform(0.7, 1.0, samples),
        }
    )


def main() -> None:
    normal_data = generate_normal_telemetry(samples=1000)
    attack_data = generate_attack_telemetry(samples=200)

    detector = ThreatDetector(random_state=42)
    detector.train(normal_data)

    combined_data = pd.concat([normal_data, attack_data], ignore_index=True)
    anomaly_scores = detector.predict_anomaly(combined_data)
    tss_scores = detector.compute_tss(combined_data)

    normal_count = len(normal_data)
    attack_count = len(attack_data)
    detected_anomalies = int((anomaly_scores > 0.5).sum())
    detected_attack_anomalies = int((anomaly_scores[normal_count:] > 0.5).sum())
    attack_detection_rate = detected_attack_anomalies / attack_count
    average_normal_tss = float(tss_scores[:normal_count].mean())
    average_attack_tss = float(tss_scores[normal_count:].mean())

    print(f"Normal samples: {normal_count}")
    print(f"Attack samples: {attack_count}")
    print(f"Detected anomalies: {detected_anomalies}")
    print(f"Attack anomaly detection rate: {attack_detection_rate:.4f}")
    print(f"Average TSS normal: {average_normal_tss:.4f}")
    print(f"Average TSS attack: {average_attack_tss:.4f}")

    model_path = "data/threat_detector.joblib"
    detector.save(model_path)

    loaded_detector = ThreatDetector()
    loaded_detector.load(model_path)
    loaded_detector.predict_anomaly(combined_data.head(5))
    loaded_detector.compute_tss(combined_data.head(5))


if __name__ == "__main__":
    main()
