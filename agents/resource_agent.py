def score_resource(action: dict, twin_state: dict) -> float:
    predicted_cpu_after = (
        twin_state["avg_cpu_load"] +
        action["predicted_effect"]["cpu_load"]
    )
    predicted_network_after = (
        twin_state["avg_network_load"] +
        action["predicted_effect"]["network_load"]
    )

    utility = 1 - (predicted_cpu_after + predicted_network_after) / 2
    return float(min(max(utility, 0), 1))
