import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "robomaster_velocity_test",
    Path(__file__).parents[1] / "robot_deploy/scripts/robomaster_velocity_test.py",
)
test_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_script)


@pytest.mark.parametrize("mode", ["success", "no_ack", "exception"])
def test_velocity_pulse_handles_push_without_ack_and_stops_on_exception(monkeypatch, mode):
    commands = []

    class Chassis:
        def drive_speed(self, **command):
            commands.append(command)
            if mode == "exception" and command["x"] != 0:
                raise RuntimeError("transport failure")
            return mode != "no_ack"

    monkeypatch.setattr(test_script.time, "sleep", lambda seconds: None)
    if mode == "exception":
        with pytest.raises(RuntimeError, match="transport failure"):
            test_script.velocity_pulse(Chassis(), 0.3, 1.0)
    else:
        test_script.velocity_pulse(Chassis(), 0.3, 1.0)
    assert commands[0] == {"x": 0.3, "y": 0, "z": 0, "timeout": 1.0}
    assert commands[1:] == [{"x": 0, "y": 0, "z": 0}] * 3
