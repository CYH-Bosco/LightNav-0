"""RealSense -> existing VLN client -> existing MPC -> local JSON only.

No ROS publisher, robot SDK, or robot connection is created. The MPC uses an
explicit stationary pose assumption; this is a bench check, not live control.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from jetson_preview import PreviewServer, pointing_record

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORKSPACE / "src/vln_client"), str(WORKSPACE / "src/vln_mpc")]

from vln_client.vln_client import OUTPUT_HEIGHT, OUTPUT_WIDTH, VlnClient  # noqa: E402
from vln_mpc.mpc import MPCController, build_pose_aligned_reference  # noqa: E402


def inspect_frames(rgb: np.ndarray, depth: np.ndarray, scale: float) -> dict:
    if rgb.shape != (360, 640, 3) or rgb.dtype != np.uint8:
        raise RuntimeError("camera must supply 640x360 uint8 RGB")
    if depth.shape != (360, 640) or depth.dtype != np.uint16:
        raise RuntimeError("camera must supply 640x360 uint16 depth")
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("invalid depth scale")
    valid = depth[depth > 0].astype(np.float64) * scale
    if valid.size == 0:
        raise RuntimeError("depth frame has no valid pixels")
    if float(rgb.std()) == 0:
        raise RuntimeError("RGB frame is constant")
    return {
        "rgb_shape": list(rgb.shape),
        "rgb_std": float(rgb.std()),
        "valid_depth_fraction": float(valid.size / depth.size),
        "depth_m_p05_p50_p95": np.percentile(valid, [5, 50, 95]).tolist(),
    }


def make_controller() -> MPCController:
    return MPCController(
        horizon=5, dt_s=0.1, w_max=0.4, a_max_v=0.5, a_max_w=1.0,
        q_weights=(10.0, 10.0, 1.0), r_weights=(0.1, 0.1),
    )


def velocity_preview(controller: MPCController, waypoints: list, stop: bool | None) -> dict:
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError("waypoints must be nonempty Nx3")
    if not np.all(np.isfinite(points)):
        raise ValueError("waypoints must be finite")
    if stop is True:
        v, w = 0.0, 0.0
        reason = "policy_stop"
    else:
        pose = np.zeros(3)
        reference = build_pose_aligned_reference(
            points, pose, horizon=5, weights=(10.0, 10.0, 1.0),
        )
        (v, w), _ = controller.solve(pose, reference, (0.0, 0.0), v_max=0.2)
        reason = "mpc_solved"
    if not (math.isfinite(v) and math.isfinite(w)):
        raise RuntimeError("MPC returned non-finite velocity")
    return {
        "linear_x_m_s": v, "angular_z_rad_s": w, "reason": reason,
        "motion_published": False, "pose_source": "stationary_bench_assumption",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="ws://127.0.0.1:8050")
    parser.add_argument("--instruction", default="Move forward along the clear path.")
    parser.add_argument("--camera-only", action="store_true")
    parser.add_argument("--view", action="store_true", help="show camera and trajectory in a browser")
    parser.add_argument("--continuous", action="store_true", help="run until Ctrl+C")
    parser.add_argument("--view-host", default="127.0.0.1")
    parser.add_argument("--view-port", type=int, default=8088)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=WORKSPACE.parents[0] / "logs/jetson-dry-run")
    args = parser.parse_args()
    if args.samples <= 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("samples and timeout must be positive and finite")
    if args.view and args.camera_only:
        parser.error("--view requires inference; omit --camera-only")
    logging.basicConfig(level=logging.INFO)
    import pyrealsense2 as rs

    args.output.mkdir(parents=True, exist_ok=True)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 360, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    client = None
    started = False
    summary = {
        "motion_published": False, "server": args.server, "passed": False,
        "recoverable_errors": 0,
    }
    frames_seen = 0
    results_seen = 0
    first_stamp = None
    last_stamp = None
    viewer = None
    try:
        if args.view:
            viewer = PreviewServer(args.view_host, args.view_port)
            print(f"Trajectory preview: http://{args.view_host}:{args.view_port}", flush=True)
        profile = pipeline.start(config)
        started = True
        device = profile.get_device()
        scale = device.first_depth_sensor().get_depth_scale()
        summary["camera"] = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "depth_scale_m": scale,
        }
        controller = None
        if not args.camera_only:
            controller = make_controller()
            client = VlnClient(logger=logging.getLogger("vln_client"), server_url=args.server)
            client.start(args.instruction)
        deadline = time.monotonic() + args.timeout
        last_result_at = time.monotonic()
        last_error = ""
        with (
            (args.output / "samples.jsonl").open("w") as output,
            (args.output / "errors.jsonl").open("w") as errors,
        ):
            while args.continuous or time.monotonic() < deadline:
                if args.continuous and time.monotonic() - last_result_at > args.timeout:
                    raise TimeoutError("no new samples before timeout")
                frames = pipeline.wait_for_frames(5000)
                color = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color or not depth_frame:
                    continue
                # Let auto-exposure settle before validating pixels.
                frames_seen += 1
                if frames_seen <= 30:
                    continue
                rgb = np.asanyarray(color.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                stamp = color.get_timestamp()
                if last_stamp is not None and stamp <= last_stamp:
                    raise RuntimeError("color camera timestamps are not increasing")
                if first_stamp is None:
                    first_stamp = stamp
                last_stamp = stamp
                metrics = inspect_frames(rgb, depth, scale)
                summary["latest_camera_metrics"] = metrics
                if frames_seen == 31:
                    if not cv2.imwrite(str(args.output / "rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                        raise RuntimeError("could not save RGB preview")
                    np.save(args.output / "depth.npy", depth)
                    print(json.dumps({"camera": summary["camera"], **metrics}), flush=True)
                if args.camera_only:
                    results_seen += 1
                    last_result_at = time.monotonic()
                    output.write(json.dumps(metrics) + "\n")
                else:
                    snapshot = client.snapshot()
                    if snapshot.error:
                        # VlnClient reconnects automatically. Never feed a failed
                        # response to MPC or keep displaying its previous output.
                        if snapshot.error != last_error:
                            last_error = snapshot.error
                            summary["recoverable_errors"] += 1
                            event = {
                                "status": "RECONNECTING", "error": snapshot.error,
                                "stamp_ns": time.time_ns(), "motion_published": False,
                            }
                            errors.write(json.dumps(event) + "\n")
                            errors.flush()
                            print(json.dumps(event), flush=True)
                            if viewer is not None:
                                viewer.publish_status(event)
                        client.take_results()
                        continue
                    last_error = ""
                    stamp_ns = time.time_ns()
                    client.offer_frame(
                        stamp_sec=stamp_ns // 1_000_000_000,
                        stamp_nanosec=stamp_ns % 1_000_000_000,
                        width=640, height=360, step=1920, rgb=rgb.tobytes(),
                    )
                    for result in client.take_results():
                        preview = velocity_preview(controller, result.waypoints, result.stop)
                        record = {
                            **pointing_record(result, OUTPUT_WIDTH, OUTPUT_HEIGHT),
                            "seq": result.sequence, "latency_ms": result.latency_ms,
                            "waypoints": result.waypoints, "stop": result.stop,
                            "capture_stamp_ns": result.frame.stamp_sec * 1_000_000_000 + result.frame.stamp_nanosec,
                            "velocity_preview": preview,
                        }
                        output.write(json.dumps(record, allow_nan=False) + "\n")
                        output.flush()
                        if viewer is not None:
                            frame_rgb = np.frombuffer(result.frame.rgb, dtype=np.uint8).reshape(
                                result.frame.height, result.frame.step,
                            )[:, :result.frame.width * 3].reshape(
                                result.frame.height, result.frame.width, 3,
                            )
                            viewer.publish(frame_rgb, record)
                        print(json.dumps(record, allow_nan=False), flush=True)
                        results_seen += 1
                        last_result_at = time.monotonic()
                if results_seen >= args.samples and not args.continuous:
                    summary["passed"] = True
                    break
        if not summary["passed"]:
            raise TimeoutError(f"only {results_seen}/{args.samples} samples before timeout")
    except KeyboardInterrupt:
        summary["passed"] = results_seen > 0
        summary["stopped_by_user"] = True
    except Exception as exc:
        summary["error"] = str(exc)
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if client is not None:
            client.close()
        if started:
            pipeline.stop()
        summary["frames_seen"] = frames_seen
        summary["samples"] = results_seen
        if first_stamp is not None and last_stamp > first_stamp:
            summary["camera_hz"] = (frames_seen - 31) * 1000.0 / (last_stamp - first_stamp)
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
