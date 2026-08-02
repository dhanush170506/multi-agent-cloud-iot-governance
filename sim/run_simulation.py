import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from digital_twin.action_generator import generate_candidate_actions
from digital_twin.telemetry_generator import generate_telemetry
from digital_twin.threat_injector import inject_threat, schedule_random_threats
from digital_twin.twin import DigitalTwin


def main():
    twin = DigitalTwin(100)
    threat_schedule = set(schedule_random_threats(num_ticks=100))
    rows = []

    for tick in range(100):
        twin.tick()

        if tick in threat_schedule:
            device = random.choice(twin.devices)
            severity = random.uniform(0.2, 0.6)
            inject_threat(twin, device.id, severity)

        telemetry = generate_telemetry(twin)
        candidate_actions = generate_candidate_actions(twin)

        rows.append(
            {
                "tick": tick,
                "avg_cpu_load": telemetry["avg_cpu_load"],
                "avg_network_load": telemetry["avg_network_load"],
                "avg_trust_score": telemetry["avg_trust_score"],
                "total_cost_per_hour": telemetry["total_cost_per_hour"],
                "total_energy_kwh": telemetry["total_energy_kwh"],
                "number_of_actions_generated": len(candidate_actions),
            }
        )

    pd.DataFrame(rows).to_csv("data/twin_log.csv", index=False)


if __name__ == "__main__":
    main()
