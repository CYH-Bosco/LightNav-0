"""Verify RoboMaster EP and optionally send a brief direct chassis velocity pulse."""

import argparse
import json
import signal
import threading
import time
from pathlib import Path


def stop_chassis(chassis):
    completed = []
    for _ in range(3):
        try:
            # The installed SDK sends speed-mode PUSH packets without ACKs.
            # Its False return is expected; stopping is verified by telemetry.
            chassis.drive_speed(x=0, y=0, z=0)
            completed.append(True)
        except Exception:
            completed.append(False)
        time.sleep(0.05)
    return completed


def velocity_pulse(chassis, speed, duration):
    deadline = time.monotonic() + duration
    try:
        chassis.drive_speed(x=speed, y=0, z=0, timeout=duration)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
    finally:
        stopped = stop_chassis(chassis)
        if not any(stopped):
            raise RuntimeError("all zero velocity calls raised exceptions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--move", action="store_true", help="send the forward velocity pulse")
    parser.add_argument("--speed", type=float, default=0.3)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("logs/robomaster-velocity-test.json"))
    args = parser.parse_args()
    if not 0 < args.speed <= 0.3 or not 0 < args.duration <= 2.0:
        parser.error("speed must be in (0, 0.3] m/s and duration in (0, 2] seconds")
    from robomaster import robot

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    ep = robot.Robot()
    chassis = None
    velocity_subscribed = False
    lock = threading.Lock()
    velocities = []
    report = {"passed": False, "motion_requested": args.move, "connection": "rndis"}

    def on_velocity(values):
        with lock:
            velocities.append({"time": time.monotonic(), "values": list(values)})

    try:
        if not ep.initialize(conn_type="rndis"):
            raise RuntimeError("RoboMaster initialization failed")
        report["robot_ip"] = ep.ip
        report["version"] = ep.get_version()
        if not report["version"]:
            raise RuntimeError("robot did not return firmware version")
        chassis = ep.chassis
        if not any(stop_chassis(chassis)):
            raise RuntimeError("all initial zero velocity calls raised exceptions")
        velocity_subscribed = bool(chassis.sub_velocity(freq=10, callback=on_velocity))
        if not velocity_subscribed:
            raise RuntimeError("velocity telemetry subscription failed")
        time.sleep(1.0)
        with lock:
            before = list(velocities)
        if len(before) < 2:
            raise RuntimeError("no usable velocity telemetry")
        if any(abs(float(v)) > 0.03 for row in before[-2:] for v in row["values"][3:]):
            raise RuntimeError("robot is already moving; test aborted")
        report["connected"] = True
        print(json.dumps({"connected": True, "version": report["version"], "ip": ep.ip}), flush=True)
        if args.move:
            report["command"] = {"x_m_s": args.speed, "y_m_s": 0, "yaw_deg_s": 0, "duration_s": args.duration}
            print(json.dumps(report["command"]), flush=True)
            velocity_pulse(chassis, args.speed, args.duration)
        stopped_at = time.monotonic()
        time.sleep(0.7)
        with lock:
            report["velocity_samples"] = list(velocities)
        after = report["velocity_samples"][-2:]
        if len(after) < 2 or any(row["time"] < stopped_at for row in after):
            raise RuntimeError("no fresh post-stop velocity telemetry")
        report["peak_forward_m_s"] = max(float(row["values"][3]) for row in report["velocity_samples"])
        if args.move and report["peak_forward_m_s"] <= 0.03:
            raise RuntimeError("telemetry did not show forward motion")
        report["stopped"] = all(abs(float(v)) <= 0.03 for row in after for v in row["values"][3:])
        if not report["stopped"]:
            raise RuntimeError("telemetry does not confirm stopped chassis")
        report["passed"] = True
    except BaseException as exc:
        report["error"] = str(exc) or type(exc).__name__
        raise
    finally:
        if chassis is not None:
            report["final_stop_calls_completed"] = any(stop_chassis(chassis))
            if velocity_subscribed:
                chassis.unsub_velocity()
        try:
            ep.close()
        except Exception as exc:
            report["close_error"] = str(exc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
