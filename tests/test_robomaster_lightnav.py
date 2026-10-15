import importlib.util
import math
import sys
import threading
from pathlib import Path

import pytest

pytest.importorskip("casadi")
pytest.importorskip("cv2")
pytest.importorskip("aiohttp")

sys.path.insert(0, str(Path(__file__).parents[1] / "robot_deploy/scripts"))
spec = importlib.util.spec_from_file_location(
    "robomaster_lightnav", Path(__file__).parents[1] / "robot_deploy/scripts/robomaster_lightnav.py",
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


def test_command_limits_and_clockwise_sdk_yaw():
    v, clockwise = pipeline.sdk_command(0.9, 1.0, 0.15, 0.3)
    assert v == 0.15
    assert clockwise == pytest.approx(-math.degrees(0.3))
    assert pipeline.sdk_command(-0.2, -0.4, 0.15, 0.3)[0] == 0
    with pytest.raises(ValueError):
        pipeline.sdk_command(float("nan"), 0, 0.15, 0.3)


def test_stop_latches_and_dry_run_never_sends_positive_velocity():
    class Chassis:
        def __init__(self):
            self.commands = []

        def drive_speed(self, **kwargs):
            self.commands.append(kwargs)

    ep = pipeline.EPConnection.__new__(pipeline.EPConnection)
    ep.lock = threading.RLock()
    ep.chassis = Chassis()
    ep.motion = True
    ep.latched_stop = False
    ep.max_v, ep.max_w = 0.15, 0.3
    ep.send(0.5, 1)
    assert ep.chassis.commands[-1]["x"] == 0.15
    ep.stop()
    ep.send(0.5, 1)
    assert ep.chassis.commands[-1] == {"x": 0, "y": 0, "z": 0, "timeout": 0.3}
    ep.latched_stop = False
    ep.motion = False
    ep.send(0.5, 1)
    assert ep.chassis.commands[-1]["x"] == ep.chassis.commands[-1]["z"] == 0


def test_wheel_odometry_signs_heading_anchor_and_freshness(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: clock[0])
    odom = pipeline.EPOdometry()
    odom.on_attitude((90, 0, 0))
    odom.on_velocity((0, 0, 0, 1, 1, 0))
    clock[0] = 1.1
    odom.on_attitude((90, 0, 0))
    odom.on_velocity((0, 0, 0, 1, 1, 0))
    fresh, history, _ = odom.snapshot()
    assert fresh
    assert history[-1].x == pytest.approx(0.1)
    assert history[-1].y == pytest.approx(-0.1)
    assert history[-1].yaw == 0
    clock[0] = 1.2
    odom.on_attitude((80, 0, 0))
    odom.on_velocity((0, 0, 0, 0, 0, 0))
    assert odom.snapshot()[1][-1].yaw == pytest.approx(math.radians(10))
    clock[0] = 2.0
    assert odom.snapshot()[0] is False


@pytest.mark.parametrize("mode,completes", [("objnav", True), ("track", False)])
def test_model_stop_zeros_motion_but_tracking_can_resume(mode, completes):
    class FakeEP:
        def __init__(self):
            self.commands = []

        def send(self, v, w):
            self.commands.append((v, w))

    ep = FakeEP()
    assert pipeline.handle_policy_stop(mode, True, ep) is completes
    assert ep.commands == [(0, 0)]
    assert pipeline.handle_policy_stop(mode, False, ep) is False
    assert ep.commands == [(0, 0)]
