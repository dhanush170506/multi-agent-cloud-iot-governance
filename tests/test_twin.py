from copy import deepcopy

from digital_twin.action_generator import generate_candidate_actions
from digital_twin.threat_injector import inject_threat
from digital_twin.twin import DigitalTwin


def test_digital_twin_values_remain_within_valid_bounds_after_100_ticks():
    twin = DigitalTwin(50)

    for _ in range(100):
        twin.tick()

    for device in twin.devices:
        assert 0 <= device.cpu_load <= 1
        assert 0 <= device.memory_load <= 1
        assert 0 <= device.network_load <= 1
        assert 0 <= device.trust_score <= 1


def test_inject_threat_modifies_only_targeted_device():
    twin = DigitalTwin(5)
    target = twin.devices[0]
    original_devices = deepcopy(twin.devices)

    inject_threat(twin, target.id, 0.3)

    assert twin.devices[0].trust_score != original_devices[0].trust_score
    assert twin.devices[0].network_load != original_devices[0].network_load
    assert twin.devices[0].cpu_load == original_devices[0].cpu_load
    assert twin.devices[0].memory_load == original_devices[0].memory_load
    assert twin.devices[0].cost_per_hour == original_devices[0].cost_per_hour
    assert twin.devices[0].energy_kwh == original_devices[0].energy_kwh

    for device, original_device in zip(twin.devices[1:], original_devices[1:]):
        assert device == original_device


def test_generate_candidate_actions_returns_non_empty_list():
    twin = DigitalTwin(10)

    actions = generate_candidate_actions(twin)

    assert isinstance(actions, list)
    assert actions
