import random
from uuid import uuid4


def generate_candidate_actions(twin) -> list[dict]:
    devices = twin.devices
    total_actions = random.randint(5, 10)
    lowest_trust_device = min(devices, key=lambda device: device.trust_score)

    actions = [
        {
            "id": str(uuid4()),
            "type": "isolate",
            "target_device_id": lowest_trust_device.id,
            "predicted_effect": {
                "cpu_load": random.uniform(-0.05, 0.02),
                "network_load": random.uniform(-0.30, -0.10),
                "trust_score": random.uniform(0.02, 0.08),
                "cost_per_hour": random.uniform(0.00, 0.03),
                "energy_kwh": random.uniform(-0.05, 0.02),
            },
        },
        {
            "id": str(uuid4()),
            "type": "no_op",
            "target_device_id": random.choice(devices).id,
            "predicted_effect": {
                "cpu_load": 0.0,
                "network_load": 0.0,
                "trust_score": 0.0,
                "cost_per_hour": 0.0,
                "energy_kwh": 0.0,
            },
        },
    ]

    action_types = ["migrate", "scale_down", "isolate"]
    effect_ranges = {
        "migrate": {
            "cpu_load": (-0.10, 0.04),
            "network_load": (-0.05, 0.10),
            "trust_score": (-0.02, 0.04),
            "cost_per_hour": (-0.02, 0.04),
            "energy_kwh": (-0.08, 0.05),
        },
        "scale_down": {
            "cpu_load": (-0.12, -0.02),
            "network_load": (-0.04, 0.02),
            "trust_score": (-0.03, 0.01),
            "cost_per_hour": (-0.05, -0.01),
            "energy_kwh": (-0.15, -0.03),
        },
        "isolate": {
            "cpu_load": (-0.05, 0.02),
            "network_load": (-0.30, -0.10),
            "trust_score": (0.02, 0.08),
            "cost_per_hour": (0.00, 0.03),
            "energy_kwh": (-0.05, 0.02),
        },
    }

    while len(actions) < total_actions:
        action_type = random.choice(action_types)
        ranges = effect_ranges[action_type]
        actions.append(
            {
                "id": str(uuid4()),
                "type": action_type,
                "target_device_id": random.choice(devices).id,
                "predicted_effect": {
                    key: random.uniform(*value)
                    for key, value in ranges.items()
                },
            }
        )

    return actions
