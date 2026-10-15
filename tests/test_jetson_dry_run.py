"""Bench validation does not require a camera, ROS, or model weights."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("aiohttp")
pytest.importorskip("cv2")

sys.path.insert(0, str(Path(__file__).parents[1] / "robot_deploy/scripts"))

spec = importlib.util.spec_from_file_location(
    "jetson_dry_run", Path(__file__).parents[1] / "robot_deploy/scripts/jetson_dry_run.py",
)
dry_run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dry_run)


def test_camera_validation_rejects_empty_depth_and_constant_rgb():
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    rgb[0, 0] = 255
    depth = np.zeros((360, 640), dtype=np.uint16)
    with pytest.raises(RuntimeError, match="no valid pixels"):
        dry_run.inspect_frames(rgb, depth, 0.001)
    depth[:] = 1000
    rgb[:] = 0
    with pytest.raises(RuntimeError, match="constant"):
        dry_run.inspect_frames(rgb, depth, 0.001)


def test_policy_stop_skips_solver_and_returns_zero_preview():
    class UnexpectedSolver:
        def solve(self, *args, **kwargs):
            pytest.fail("stop must not invoke MPC")

    result = dry_run.velocity_preview(UnexpectedSolver(), [[1.0, 0.0, 0.0]], True)
    assert result["linear_x_m_s"] == result["angular_z_rad_s"] == 0.0
    assert result["motion_published"] is False


def test_invalid_prediction_is_rejected_before_solver():
    with pytest.raises(ValueError, match="finite"):
        dry_run.velocity_preview(None, [[np.nan, 0.0, 0.0]], False)


def test_pointing_scales_inference_pixels_to_preview_and_omits_absent_points():
    result = SimpleNamespace(
        apos_state="point", apos_px=(240, 135),
        opos_state="point", opos_px=(120, 54),
    )
    record = dry_run.pointing_record(result, 480, 270)
    assert record["pointing"]["frame_size"] == [480, 270]
    assert record["pointing_markers"] == [
        {"channel": "apos", "x": 0.5, "y": 0.5},
        {"channel": "opos", "x": 0.25, "y": 0.2},
    ]
    # The 640x360 canvas places the inference centre at (320, 180).
    marker = record["pointing_markers"][0]
    assert (marker["x"] * 640, marker["y"] * 360) == (320, 180)
    result.apos_state = "none"
    result.opos_px = None
    assert dry_run.pointing_record(result, 480, 270)["pointing_markers"] == []
    result.apos_state = "point"
    result.apos_px = (480, 135)
    assert dry_run.pointing_record(result, 480, 270)["pointing_markers"] == []


def test_preview_serves_camera_and_rejects_control_requests():
    import cv2

    viewer = dry_run.PreviewServer("127.0.0.1", 0)
    url = f"http://127.0.0.1:{viewer.server.server_port}"
    try:
        viewer.publish(np.zeros((360, 640, 3), dtype=np.uint8), {"seq": 1})
        with urlopen(url + "/camera.jpg") as response:
            image = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (360, 640, 3)
        with urlopen(url + "/state.json") as response:
            assert response.read() == b'{"seq": 1}'
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url, data=b"move", method="POST"))
        assert error.value.code == 501
        viewer.publish_status({"status": "RECONNECTING", "error": "invalid prediction"})
        with urlopen(url + "/state.json") as response:
            assert "seq" not in json.load(response)
        with pytest.raises(HTTPError) as error:
            urlopen(url + "/camera.jpg")
        assert error.value.code == 404
        viewer.publish(np.zeros((360, 640, 3), dtype=np.uint8), {"seq": 2})
        with urlopen(url + "/state.json") as response:
            assert json.load(response)["seq"] == 2
    finally:
        viewer.close()


def test_motion_preview_stop_and_live_camera_during_recovery():
    stopped = []
    viewer = dry_run.PreviewServer("127.0.0.1", 0, on_stop=lambda: stopped.append(True))
    url = f"http://127.0.0.1:{viewer.server.server_port}"
    try:
        viewer.publish_camera(np.zeros((360, 640, 3), dtype=np.uint8))
        viewer.publish_status({"status": "RECONNECTING"}, clear_camera=False)
        with urlopen(url + "/camera.jpg") as response:
            assert response.status == 200
        with urlopen(Request(url + "/stop", data=b"", method="POST")) as response:
            assert response.status == 200
        assert stopped == [True]
    finally:
        viewer.close()


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
def test_runner_survives_missing_rvq_response(monkeypatch, tmp_path, continuous, persistent):
    rgb = np.zeros((360, 640, 3), dtype=np.uint8)
    rgb[0, 0] = 255
    depth = np.ones((360, 640), dtype=np.uint16)
    cleanup = []
    clock = [0.0]

    class Camera:
        count = 0

        def start(self, config):
            sensor = SimpleNamespace(get_depth_scale=lambda: 0.001)
            device = SimpleNamespace(
                first_depth_sensor=lambda: sensor, get_info=lambda key: "test-camera",
            )
            return SimpleNamespace(get_device=lambda: device)

        def wait_for_frames(self, timeout):
            self.count += 1
            if self.count > 35:
                raise KeyboardInterrupt
            color = SimpleNamespace(get_data=lambda: rgb, get_timestamp=lambda: self.count)
            return SimpleNamespace(
                get_color_frame=lambda: color,
                get_depth_frame=lambda: SimpleNamespace(get_data=lambda: depth),
            )

        def stop(self):
            cleanup.append("camera")

    class Client:
        sequence = 0

        def __init__(self, **kwargs):
            pass

        def start(self, instruction):
            pass

        def snapshot(self):
            self.sequence += 1
            if persistent:
                clock[0] += 1.0
            return SimpleNamespace(
                error="Missing rvq act levels [0]"
                if self.sequence in (2, 3) or (persistent and self.sequence >= 2) else "",
            )

        def offer_frame(self, **kwargs):
            pass

        def take_results(self):
            if self.sequence in (2, 3) or (persistent and self.sequence >= 2):
                return []
            return [SimpleNamespace(
                sequence=self.sequence, waypoints=[[1.0, 0.0, 0.0]], stop=True,
                latency_ms=123.0, frame=SimpleNamespace(stamp_sec=1, stamp_nanosec=0),
            )]

        def close(self):
            cleanup.append("client")

    sdk = SimpleNamespace(
        pipeline=Camera,
        config=lambda: SimpleNamespace(enable_stream=lambda *args: None),
        stream=SimpleNamespace(color=1, depth=2), format=SimpleNamespace(rgb8=1, z16=2),
        camera_info=SimpleNamespace(name=1, serial_number=2),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", sdk)
    monkeypatch.setattr(dry_run, "VlnClient", Client)
    monkeypatch.setattr(dry_run, "make_controller", lambda: None)
    argv = ["dry-run", "--samples", "2", "--output", str(tmp_path)]
    if continuous:
        argv.append("--continuous")
    if persistent:
        monkeypatch.setattr(dry_run.time, "monotonic", lambda: clock[0])
        argv += ["--timeout", "3"]
    monkeypatch.setattr(sys, "argv", argv)
    if persistent:
        with pytest.raises(TimeoutError):
            dry_run.main()
    else:
        dry_run.main()
    results = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in results][:2] == ([1] if persistent else [1, 4])
    assert all(r["velocity_preview"]["motion_published"] is False for r in results)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"] is (not persistent)
    assert summary["recoverable_errors"] == 1
    assert "Missing rvq" in (tmp_path / "errors.jsonl").read_text()
    assert set(cleanup) == {"camera", "client"}
