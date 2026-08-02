import numpy as np
import pandas as pd
from dataclasses import dataclass
from faker import Faker


@dataclass
class Device:
    id: str
    cpu_load: float
    memory_load: float
    network_load: float
    trust_score: float
    cost_per_hour: float
    energy_kwh: float


class DigitalTwin:
    def __init__(self, num_devices=100):
        fake = Faker()
        self.devices = [
            Device(
                id=fake.uuid4(),
                cpu_load=np.random.uniform(0.1, 0.8),
                memory_load=np.random.uniform(0.2, 0.9),
                network_load=np.random.uniform(0.05, 0.7),
                trust_score=np.random.uniform(0.7, 1.0),
                cost_per_hour=np.random.uniform(0.05, 0.30),
                energy_kwh=np.random.uniform(0.10, 1.50),
            )
            for _ in range(num_devices)
        ]

    def tick(self):
        for device in self.devices:
            device.cpu_load = np.clip(device.cpu_load + np.random.normal(0, 0.02), 0, 1)
            device.memory_load = np.clip(device.memory_load + np.random.normal(0, 0.02), 0, 1)
            device.network_load = np.clip(device.network_load + np.random.normal(0, 0.02), 0, 1)
            device.trust_score = np.clip(device.trust_score + np.random.normal(0, 0.02), 0, 1)

    def get_state(self):
        return pd.DataFrame(
            [
                {
                    "id": device.id,
                    "cpu_load": device.cpu_load,
                    "memory_load": device.memory_load,
                    "network_load": device.network_load,
                    "trust_score": device.trust_score,
                    "cost_per_hour": device.cost_per_hour,
                    "energy_kwh": device.energy_kwh,
                }
                for device in self.devices
            ]
        )
