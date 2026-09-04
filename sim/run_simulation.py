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


# Severity high enough to push every device across the attack condition:
#   avg_trust_score < 0.7  OR  avg_network_load > 0.7
# With 100 devices each shifted by 0.5 the averages move by 0.5, which
# guarantees crossing both sides of the OR.
_ATTACK_SEVERITY = 0.5


def main():
    twin = DigitalTwin(100)
    prev_wS = 1 / 3
    rows = []

    for tick in range(100):
        twin.tick()

        # Controlled attack at tick 50: inject every device so the
        # averaged telemetry clearly crosses the attack threshold.
        if tick == 50:
            for device in twin.devices:
                inject_threat(twin, device.id, _ATTACK_SEVERITY)

        telemetry = generate_telemetry(twin)
        candidate_actions = generate_candidate_actions(twin)
        twin_state = telemetry
        resource_state = twin.get_state()

        U_R_list = [
            score_resource(action, resource_state)
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
                "chosen_action_type": chosen_action["type"],
                "target_device_id": chosen_action["target_device_id"],
                "avg_trust_score": telemetry["avg_trust_score"],
                "avg_network_load": telemetry["avg_network_load"],
            }
        )

    pd.DataFrame(rows).to_csv("data/tapna_log.csv", index=False)


if __name__ == "__main__":
    main()
