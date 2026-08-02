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
    return float(min(max(utility, 0), 1))


def compute_threat_severity(twin_state, baseline_trust=0.8) -> float:
    severity = max(
        0,
        baseline_trust - twin_state["avg_trust_score"]
    ) / baseline_trust

    return float(min(max(severity, 0), 1))
