def score_sustainability(action, twin_state) -> float:
    predicted_cost_after = (
        twin_state["total_cost_per_hour"] +
        action["predicted_effect"]["cost_per_hour"]
    )
    predicted_energy_after = (
        twin_state["total_energy_kwh"] +
        action["predicted_effect"]["energy_kwh"]
    )

    current_cost = twin_state["total_cost_per_hour"]
    current_energy = twin_state["total_energy_kwh"]
    predicted_cost_after_normalized = (
        predicted_cost_after / current_cost
        if current_cost
        else predicted_cost_after
    )
    predicted_energy_after_normalized = (
        predicted_energy_after / current_energy
        if current_energy
        else predicted_energy_after
    )

    utility = 1 - (
        predicted_cost_after_normalized +
        predicted_energy_after_normalized
    ) / 2

    return float(min(max(utility, 0), 1))
