import random


def inject_threat(twin, device_id, severity):
    for device in twin.devices:
        if device.id == device_id:
            device.trust_score = min(max(device.trust_score - severity, 0), 1)
            device.network_load = min(max(device.network_load + severity, 0), 1)
            break


def schedule_random_threats(num_ticks, probability=0.05):
    return [
        tick
        for tick in range(num_ticks)
        if random.random() < probability
    ]
