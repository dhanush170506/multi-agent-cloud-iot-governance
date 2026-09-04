import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat, schedule_random_threats
from digital_twin.twin import DigitalTwin
from negotiation.baselines import (
    greedy_cost_baseline,
    greedy_latency_baseline,
    static_weighted_baseline,
)
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna


SEEDS = [42, 123, 456]
NUM_TICKS = 100


def _score_actions(actions, twin_state, resource_state):
    U_R_list = [
        score_resource(action, resource_state)
        for action in actions
    ]
    U_S_list = [
        score_security(action, twin_state)
        for action in actions
    ]
    U_C_list = [
        score_sustainability(action, twin_state)
        for action in actions
    ]
    return U_R_list, U_S_list, U_C_list


def _run_strategy(strategy, threat_schedule, seed):
    random.seed(seed)
    np.random.seed(seed)

    twin = DigitalTwin(100)
    prev_wS = 1 / 3
    rows = []

    for tick in range(NUM_TICKS):
        twin.tick()

        if tick in threat_schedule:
            for device in twin.devices:
                inject_threat(twin, device.id, 0.5)
            inject_threat(twin, device.id, 0.5)

        twin_state = generate_telemetry(twin)
        resource_state = twin.get_state()
        actions = generate_candidate_actions(twin)
        U_R_list, U_S_list, U_C_list = _score_actions(actions, twin_state, resource_state)
        tss = compute_threat_severity(twin_state)

        if strategy == "TAPNA":
            chosen_action, weights = run_tapna(
                actions,
                U_R_list,
                U_S_list,
                U_C_list,
                tss,
                prev_wS,
            )
            prev_wS = weights["wS"]
        elif strategy == "Greedy Cost":
            chosen_action = greedy_cost_baseline(actions, U_C_list)
            weights = {"wR": 0.0, "wS": 0.0, "wC": 1.0}
        elif strategy == "Greedy Latency":
            chosen_action = greedy_latency_baseline(actions, U_R_list)
            weights = {"wR": 1.0, "wS": 0.0, "wC": 0.0}
        else:
            chosen_action = static_weighted_baseline(
                actions,
                U_R_list,
                U_S_list,
                U_C_list,
            )
            weights = {"wR": 1 / 3, "wS": 1 / 3, "wC": 1 / 3}

        chosen_index = actions.index(chosen_action)
        cigi = compute_cigi(
            U_R_list[chosen_index],
            U_S_list[chosen_index],
            U_C_list[chosen_index],
            weights,
        )

        rows.append(
            {
                "strategy": strategy,
                "tick": tick,
                "TSS": tss,
                "wR": weights["wR"],
                "wS": weights["wS"],
                "wC": weights["wC"],
                "CIGI": cigi,
                "response_lag": 0,
                "chosen_action": chosen_action["id"],
                "chosen_action_type": chosen_action["type"],
                "target_device_id": chosen_action["target_device_id"],
            }
        )

    return rows


def main():
    threat_schedule = {50}
    rows = []

    for run_number, seed in enumerate(SEEDS, start=1):
        random.seed(seed)
        np.random.seed(seed)

        for strategy in [
            "TAPNA",
            "Greedy Cost",
            "Greedy Latency",
            "Static Weighted",
        ]:
            rows.extend(_run_strategy(strategy, threat_schedule, seed))

    output_path = Path("data/experiments") / "experiment_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Saved experiment results to {output_path}")


if __name__ == "__main__":
    main()
