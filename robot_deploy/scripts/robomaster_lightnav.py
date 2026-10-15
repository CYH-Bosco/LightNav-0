"""RealSense -> GPU LightNav -> measured wheel odometry/MPC -> EP velocity."""

import argparse
import json
import logging
import math
import signal
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from jetson_dry_run import inspect_frames
from jetson_preview import PreviewServer, pointing_record
from robomaster_velocity_test import stop_chassis
from vln_client.vln_client import OUTPUT_HEIGHT, OUTPUT_WIDTH, VlnClient
from vln_mpc.geometry import (
    TimedPose,
    pose_at_stamp,
    project_body_to_odom,
    project_odom_to_local,
    wrap_angle,
)
from vln_mpc.mpc import MPCController, build_pose_aligned_reference


def sdk_command(v, w, max_v, max_w):
    if not all(math.isfinite(x) for x in (v, w, max_v, max_w)):
        raise ValueError("velocity and limits must be finite")
    if max_v <= 0 or max_w <= 0:
        raise ValueError("velocity limits must be positive")
    return min(max(v, 0.0), max_v), -math.degrees(min(max(w, -max_w), max_w))


class EPOdometry:
    """Wheel velocity integration with IMU yaw, stamped on the Jetson clock.

    EP body +Y is right and positive yaw is clockwise. LightNav uses left/CCW.
    Anchor the IMU heading at startup to avoid mixing absolute IMU heading with
    the EP's separately initialized position coordinate frame.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.yaw = None
        self.initial_yaw = None
        self.attitude_time = 0.0
        self.velocity_time = 0.0
        self.previous_time = None
        self.x = self.y = 0.0
        self.previous_yaw = 0.0
        self.previous = (0.0, 0.0)
        self.lateral = 0.0
        self.history = deque(maxlen=400)

    def on_attitude(self, values):
        yaw = -math.radians(float(values[0]))
        if not math.isfinite(yaw):
            return
        with self.lock:
            if self.initial_yaw is None:
                self.initial_yaw = yaw
            self.yaw = wrap_angle(yaw - self.initial_yaw)
            self.attitude_time = time.monotonic()

    def on_velocity(self, values):
        vx, vy = float(values[3]), -float(values[4])
        if not (math.isfinite(vx) and math.isfinite(vy)):
            return
        now = time.monotonic()
        with self.lock:
            self.velocity_time = now
            if self.yaw is None or now - self.attitude_time > 0.3:
                return
            dt = 0.0 if self.previous_time is None else now - self.previous_time
            w = 0.0
            if 0 < dt < 0.3:
                dyaw = wrap_angle(self.yaw - self.previous_yaw)
                mid_yaw = self.previous_yaw + dyaw / 2
                self.x += (vx * math.cos(mid_yaw) - vy * math.sin(mid_yaw)) * dt
                self.y += (vx * math.sin(mid_yaw) + vy * math.cos(mid_yaw)) * dt
                w = dyaw / dt
            self.previous = (vx, w)
            self.lateral = vy
            self.previous_time = now
            self.previous_yaw = self.yaw
            self.history.append(TimedPose(time.time_ns(), self.x, self.y, self.yaw))

    def snapshot(self):
        with self.lock:
            fresh = (
                bool(self.history)
                and time.monotonic() - min(self.attitude_time, self.velocity_time) < 0.3
            )
            return fresh, list(self.history), self.previous


class EPConnection:
    def __init__(self, motion, max_v, max_w):
        from robomaster import robot

        self.lock = threading.RLock()
        self.latched_stop = False
        self.motion = motion
        self.max_v, self.max_w = max_v, max_w
        self.ep = robot.Robot()
        self.chassis = None
        self.subscriptions = []
        self.odom = EPOdometry()

    def start(self):
        if not self.ep.initialize(conn_type="rndis"):
            raise RuntimeError("EP USB/RNDIS initialization failed")
        self.chassis = self.ep.chassis
        self.send(0.0, 0.0)
        for name, callback in (("attitude", self.odom.on_attitude), ("velocity", self.odom.on_velocity)):
            if not getattr(self.chassis, "sub_" + name)(freq=20, callback=callback):
                raise RuntimeError(f"EP {name} subscription failed")
            self.subscriptions.append(name)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            fresh, history, previous = self.odom.snapshot()
            if fresh and len(history) >= 3:
                if abs(previous[0]) > 0.03 or abs(self.odom.lateral) > 0.03 or abs(previous[1]) > 0.05:
                    raise RuntimeError("EP is already moving")
                return
            time.sleep(0.05)
        raise TimeoutError("no fresh EP odometry")

    def send(self, v, w):
        with self.lock:
            if not self.motion or self.latched_stop:
                v = w = 0.0
            linear, clockwise_deg = sdk_command(v, w, self.max_v, self.max_w)
            if self.chassis is not None:
                # Speed PUSH packets have no ACK; SDK return False is expected.
                self.chassis.drive_speed(x=linear, y=0, z=clockwise_deg, timeout=0.3)
            return linear, -math.radians(clockwise_deg)

    def stop(self):
        with self.lock:
            self.latched_stop = True
            self.send(0.0, 0.0)

    def close(self):
        self.stop()
        if self.chassis is not None:
            stop_chassis(self.chassis)
            for name in self.subscriptions:
                try:
                    getattr(self.chassis, "unsub_" + name)()
                except Exception:
                    logging.exception("EP telemetry cleanup failed")
        self.ep.close()


def handle_policy_stop(mode, stop, ep):
    """Stop physical motion immediately; only ObjNav completes the session."""
    if stop is not True:
        return False
    ep.send(0.0, 0.0)
    return mode == "objnav"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8050")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--mode", choices=("objnav", "track"), default="objnav",
                        help="objnav ends on model stop; track pauses and keeps inference running")
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--max-vel", type=float, default=0.15, help="commanded forward speed cap, m/s")
    parser.add_argument("--max-yaw", type=float, default=0.3, help="commanded yaw speed cap, rad/s")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--duration", type=float, default=0, help="stop after seconds; 0 runs until stopped")
    parser.add_argument("--output", type=Path, default=Path("logs/robomaster-lightnav"))
    args = parser.parse_args()
    if not args.instruction.strip():
        parser.error("instruction must not be empty")
    if not (0 < args.max_vel <= 0.3 and 0 < args.max_yaw <= 0.5):
        parser.error("max-vel must be in (0, 0.3] and max-yaw in (0, 0.5]")
    if not math.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative")
    import pyrealsense2 as rs

    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    ep = EPConnection(args.motion, args.max_vel, args.max_yaw)
    stopped = threading.Event()

    def request_stop(*unused):
        ep.stop()
        stopped.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    pipeline = rs.pipeline()
    client = viewer = None
    camera_started = False
    results_seen = 0
    report = {"motion_enabled": args.motion, "max_vel": args.max_vel, "max_yaw": args.max_yaw}
    try:
        ep.start()
        report["robot_ip"] = ep.ep.ip
        report["firmware"] = ep.ep.get_version()
        viewer = PreviewServer(args.host, args.port, on_stop=request_stop)
        print(f"Browser: http://{args.host}:{args.port}; motion={args.motion}", flush=True)
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
        profile = pipeline.start(config)
        camera_started = True
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        controller = MPCController(
            horizon=5, dt_s=0.1, w_max=args.max_yaw, a_max_v=0.3, a_max_w=0.5,
            q_weights=(10.0, 10.0, 1.0), r_weights=(0.1, 0.1),
        )
        client = VlnClient(logger=logging.getLogger("vln_client"), server_url=args.server)
        client.start(args.instruction)
        trajectory = None
        last_record = None
        policy_paused = False
        last_result_at = 0.0
        last_capture_ns = 0
        next_control = next_camera = 0.0
        started_at = time.monotonic()
        frame_count = 0
        with (args.output / "events.jsonl").open("w") as output:
            while not stopped.is_set():
                now = time.monotonic()
                if args.duration and now - started_at >= args.duration:
                    break
                frames = pipeline.wait_for_frames(1000)
                rgb = np.asanyarray(frames.get_color_frame().get_data())
                depth = np.asanyarray(frames.get_depth_frame().get_data())
                frame_count += 1
                if frame_count <= 30:
                    continue
                inspect_frames(rgb, depth, depth_scale)
                stamp_ns = time.time_ns()
                now = time.monotonic()
                if now >= next_camera:
                    viewer.publish_camera(rgb)
                    next_camera = now + 0.1
                snapshot = client.snapshot()
                fresh, history, previous = ep.odom.snapshot()
                if snapshot.error or not fresh:
                    trajectory = None
                    ep.send(0.0, 0.0)
                    client.take_results()
                    viewer.publish_status({
                        "status": "RECONNECTING" if snapshot.error else "WAITING FOR ODOM",
                        "error": snapshot.error, "motion_enabled": args.motion,
                    }, clear_camera=False)
                    continue
                client.offer_frame(
                    stamp_sec=stamp_ns // 1_000_000_000, stamp_nanosec=stamp_ns % 1_000_000_000,
                    width=640, height=360, step=1920, rgb=rgb.tobytes(),
                )
                for result in client.take_results():
                    capture_ns = result.frame.stamp_sec * 1_000_000_000 + result.frame.stamp_nanosec
                    match = pose_at_stamp(history, capture_ns, 0.15)
                    if match is None:
                        trajectory = None
                        ep.send(0.0, 0.0)
                        continue
                    results_seen += 1
                    last_capture_ns = capture_ns
                    last_result_at = now
                    last_record = {
                        **pointing_record(result, OUTPUT_WIDTH, OUTPUT_HEIGHT),
                        "seq": result.sequence, "latency_ms": result.latency_ms,
                        "stop": result.stop, "instruction": args.instruction,
                        "mode": args.mode, "visible": result.visible,
                        "motion_enabled": args.motion, "live_camera": True,
                    }
                    trajectory = project_body_to_odom(result.waypoints, match[0])
                    policy_paused = result.stop is True
                    if handle_policy_stop(args.mode, result.stop, ep):
                        request_stop()
                        break
                    if policy_paused:
                        trajectory = None
                if stopped.is_set():
                    break
                if now < next_control:
                    continue
                next_control = now + 0.1
                if policy_paused:
                    ep.send(0.0, 0.0)
                    viewer.publish_status({
                        "status": "TRACKING PAUSED: MODEL STOP; INFERENCE CONTINUES",
                        "motion_enabled": args.motion, "mode": args.mode,
                    }, clear_camera=False)
                    continue
                if trajectory is None or now - last_result_at > 1.0 or time.time_ns() - last_capture_ns > 1e9:
                    trajectory = None
                    ep.send(0.0, 0.0)
                    viewer.publish_status({"status": "WAITING FOR VALID INFERENCE", "motion_enabled": args.motion}, clear_camera=False)
                    continue
                origin = history[-1]
                pose = np.asarray([origin.x, origin.y, origin.yaw])
                reference = build_pose_aligned_reference(
                    np.asarray(trajectory), pose, horizon=5, weights=(10.0, 10.0, 1.0),
                )
                local_reference = np.asarray(project_odom_to_local(reference, origin))
                try:
                    (v, w), _ = controller.solve(np.zeros(3), local_reference, previous, args.max_vel)
                    if not all(math.isfinite(x) for x in (v, w)):
                        raise RuntimeError("nonfinite MPC output")
                except Exception as exc:
                    trajectory = None
                    ep.send(0.0, 0.0)
                    viewer.publish_status({"status": "MPC ERROR", "error": str(exc), "motion_enabled": args.motion}, clear_camera=False)
                    continue
                # Recheck telemetry/inference after the solve before sending.
                fresh, _, _ = ep.odom.snapshot()
                if not fresh or client.snapshot().error or time.time_ns() - last_capture_ns > 1e9:
                    ep.send(0.0, 0.0)
                    trajectory = None
                    continue
                sent_v, sent_w = ep.send(v, w)
                record = {
                    **last_record, "waypoints": project_odom_to_local(trajectory, origin),
                    "velocity_preview": {
                        "linear_x_m_s": sent_v, "angular_z_rad_s": sent_w,
                        "motion_published": args.motion and not ep.latched_stop,
                    },
                    "odom": [origin.x, origin.y, origin.yaw],
                    "measured_forward_m_s": previous[0],
                    "planned_velocity": [float(v), float(w)],
                    "max_vel": args.max_vel, "max_yaw": args.max_yaw,
                }
                viewer.publish_record(record)
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
        report["results"] = results_seen
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        ep.stop()
        if client is not None:
            client.close()
        if camera_started:
            pipeline.stop()
        if viewer is not None:
            viewer.close()
        try:
            ep.close()
        finally:
            (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
