import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat, schedule_random_threats
from digital_twin.twin import DigitalTwin
from agents.resource_agent import score_resource
from agents.security_agent import compute_threat_severity, score_security
from agents.sustainability_agent import score_sustainability
from negotiation.cigi import compute_cigi
from negotiation.tapna import run_tapna


def main():
    twin = DigitalTwin(100)
    threat_schedule = set(schedule_random_threats(num_ticks=100))
    prev_wS = 1 / 3
    rows = []

    for tick in range(100):
        twin.tick()

        if tick in threat_schedule:
            device = random.choice(twin.devices)
            severity = random.uniform(0.2, 0.6)
            inject_threat(twin, device.id, severity)

        telemetry = generate_telemetry(twin)
        candidate_actions = generate_candidate_actions(twin)
        twin_state = telemetry

        U_R_list = [
            score_resource(action, twin_state)
            for action in candidate_actions
        ]
        U_S_list = [
            score_security(action, twin_state)
            for action in candidate_actions
        ]
        U_C_list = [
            score_sustainability(action, twin_state)
            for action in candidate_actions
        ]
        tss = compute_threat_severity(twin_state)
        chosen_action, weights = run_tapna(
            candidate_actions,
            U_R_list,
            U_S_list,
            U_C_list,
            tss,
            prev_wS,
        )
        chosen_index = candidate_actions.index(chosen_action)
        cigi = compute_cigi(
            U_R_list[chosen_index],
            U_S_list[chosen_index],
            U_C_list[chosen_index],
            weights,
        )
        prev_wS = weights["wS"]

        rows.append(
            {
                "tick": tick,
                "TSS": tss,
                "wR": weights["wR"],
                "wS": weights["wS"],
                "wC": weights["wC"],
                "CIGI": cigi,
                "chosen_action": chosen_action["id"],
            }
        )

    pd.DataFrame(rows).to_csv("data/tapna_log.csv", index=False)


if __name__ == "__main__":
    main()
