from pathlib import Path

import pandas as pd

from ai_models.security.threat_detector import ThreatDetector


_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "threat_detector.joblib"
_THREAT_DETECTOR = None


def _get_threat_detector() -> ThreatDetector:
    global _THREAT_DETECTOR

    if _THREAT_DETECTOR is None:
        detector = ThreatDetector()
        detector.load(str(_MODEL_PATH))
        _THREAT_DETECTOR = detector

    return _THREAT_DETECTOR


def _clip01(value: float) -> float:
    return float(min(max(value, 0), 1))


def _telemetry_frame(twin_state) -> pd.DataFrame:
    if isinstance(twin_state, pd.DataFrame):
        telemetry = twin_state.copy()
        if "connection_rate" not in telemetry.columns:
            telemetry["connection_rate"] = telemetry["network_load"]
        return telemetry[ThreatDetector.FEATURES]

    network_load = twin_state.get("network_load", twin_state.get("avg_network_load", 0))
    telemetry = {
        "cpu_load": twin_state.get("cpu_load", twin_state.get("avg_cpu_load", 0)),
        "network_load": network_load,
        "trust_score": twin_state.get("trust_score", twin_state.get("avg_trust_score", 1)),
        "connection_rate": twin_state.get(
            "connection_rate",
            twin_state.get("avg_connection_rate", network_load),
        ),
    }
    return pd.DataFrame([telemetry], columns=ThreatDetector.FEATURES)


def score_security(action: dict, twin_state: dict) -> float:
    predicted_trust_after = (
        twin_state["avg_trust_score"] +
        action["predicted_effect"]["trust_score"]
    )
    predicted_network_after = (
        twin_state["avg_network_load"] +
        action["predicted_effect"]["network_load"]
    )

    utility = predicted_trust_after - 0.3 * predicted_network_after
    threat_severity = compute_threat_severity(twin_state)

    return _clip01(utility * (1 - threat_severity))


def compute_threat_severity(twin_state, baseline_trust=0.8) -> float:
    telemetry = _telemetry_frame(twin_state)
    severity = _get_threat_detector().compute_tss(telemetry).mean()

    return _clip01(severity)
