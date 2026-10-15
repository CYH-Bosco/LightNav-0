# Jetson camera and velocity bench check

Replace `gpu-host.example` and `jetson-host.example` with your own hostnames.
Client scripts default to `ws://127.0.0.1:8050`; pass `--server` for a remote GPU.

## Keep the Jetson reachable while idle

On 2026-10-08, desktop AC idle suspend was already disabled, battery idle suspend
was enabled, and Wi-Fi power saving was already off. Tailscale and NetworkManager
were enabled and running. Available logs did not establish the cause of the
reported idle disconnects. The desktop settings for the desktop user have now been set
to `nothing` with timeout `0` for both AC and battery idle sleep. Display blanking
can still occur without putting the system to sleep.

Complete the system-wide settings in a Jetson terminal:

```bash
nmcli -f NAME,UUID,DEVICE connection show --active
read -r -p 'Wi-Fi profile UUID: ' wifi_uuid
read -r -p 'Wi-Fi interface: ' wifi_iface
sudo bash ~/LightNav-0/robot_deploy/scripts/keep_jetson_online.sh "$wifi_uuid" "$wifi_iface"
```

This masks system sleep/suspend/hibernate targets, sets the current active
Wi-Fi profile to retry reconnection indefinitely and explicitly disable power
saving, and enables Tailscale at boot. It does not reboot, restart Wi-Fi, or
change robot control. Administrator authentication is required; the agent's
session cannot authenticate sudo. The script requires the Wi-Fi profile UUID and interface as arguments.
NetworkManager documents [reconnection retries](https://networkmanager.dev/docs/api/latest/settings-connection.html)
and [Wi-Fi power saving](https://networkmanager.dev/docs/api/latest/settings-802-11-wireless.html).

To restore the inspected system/profile defaults:

```bash
sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
sudo nmcli connection modify "$wifi_uuid" connection.autoconnect-retries -1 802-11-wireless.powersave 0
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 3600
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'suspend'
gsettings reset org.gnome.settings-daemon.plugins.power sleep-inactive-battery-timeout
```

These settings address sleep and reconnection; power loss, a Wi-Fi driver hang,
or loss of campus connectivity can still interrupt remote access. Tailscale
currently reports a missing Jetson kernel `connmark` match; peer connectivity
works despite that warning, and firewall settings have not been changed.

## Full LightNav pipeline with EP motion and live browser view

Keep the GPU server running, connect the EP to the Jetson USB-A host port,
and keep the RealSense connected. Stop any other camera/EP control process.
Run on the Jetson:

```bash
cd ~/LightNav-0
read -r -p 'Navigation instruction: ' lightnav_instruction
.venv-jetson/bin/python robot_deploy/scripts/robomaster_lightnav.py \
  --server ws://gpu-host.example:8050 \
  --instruction "$lightnav_instruction" \
  --motion --max-vel 0.3 --max-yaw 0.3 --host 0.0.0.0
```

Open `http://127.0.0.1:8088` on the Jetson, or
`http://jetson-host.example:8088` from a computer on its Tailscale network.
The RGB image overlays a mint **Affordance** crosshair (`apos`) and a magenta
**Goal object** crosshair (`opos`) when the model returns visible pointing pixels.
Coordinates are scaled from the 480×270 inference image to the displayed image.
The caption reports each channel's state; absent or out-of-frame points have no
marker. Points come from the latest inference, so they can trail the live camera
during motion. Recovery or stale-inference status clears the markers. The bench
viewer overlays them on the exact camera frame used for inference.
The page shows live ego RGB at approximately 10 FPS, the trajectory relative
to the current robot pose, inference latency, velocity commands, measured
forward velocity, and the configured command limit. Ctrl+C, browser **STOP
ROBOT**, or a policy stop terminates the run and sends zero velocity.

`--max-vel` caps commands, not measured physical speed: the earlier EP test
commanded 0.3 m/s and telemetry peaked at 0.46 m/s. `--max-yaw` is in rad/s;
the direct SDK call converts CCW-positive yaw to clockwise-positive degrees/s.
No locomotion controller or robot adapter launch is used.

Wheel velocity and IMU yaw are subscribed at 20 Hz. Body-frame wheel velocity
is integrated into odometry with the initial IMU heading anchored to zero.
Camera and telemetry use Jetson receive timestamps. The existing geometry
utilities align predictions to odometry at image reception, and the existing
MPC runs at approximately 10 Hz with measured velocity feedback. Wheel slip,
IMU error, and USB latency affect this odometry; physical turning and trajectory
accuracy need validation on the robot.

Commands are zeroed on missing/stale telemetry (0.3 s), inference errors,
predictions older than 1 s, solver errors, or an unmatched camera timestamp.
Each SDK command has a 0.3 s client-side stop timer. Browser STOP latches the
connection into zero-command mode; restarting is required to enable motion
again. The software timer requires the Jetson process to remain operational.

To inspect the complete pipeline with the EP stationary, **omit `--motion`**.
The same odometry, camera, inference, controller, and browser still run; only
zero commands are sent. Add `--duration 15` for a bounded check.
The integration was verified with motion disabled; autonomous motion has not
been started by the agent.
The bounded check returned 73 GPU predictions and 110 MPC control records using
real EP telemetry, with zero velocity sent throughout. A separate browser check
returned HTTP 200 for the page and live RGB, displayed ten trajectory waypoints,
and exited cleanly after the STOP endpoint was called. Evidence is in
`logs/robomaster-lightnav-smoke/` and `logs/robomaster-lightnav-browser/`.

## Continuous tracking rather than completing on model stop

The direct EP pipeline defaults to `--mode objnav`: one `stop=true` response
ends the run. Use `--mode track` to send zero velocity and discard the previous
path during a model stop, while continuing to send camera frames. A later valid
non-stop prediction resumes control. Ctrl+C, browser STOP, duration limits, and
telemetry/inference watchdogs still apply. No forward command is invented when
the model predicts stop. Repeated stop predictions therefore keep the robot
stationary; this does not enforce a desired target distance.

```bash
cd ~/LightNav-0
.venv-jetson/bin/python robot_deploy/scripts/robomaster_lightnav.py \
  --server ws://gpu-host.example:8050 \
  --instruction 'Track the man in the gray shirt and white shoes.' \
  --mode track --motion --max-vel 0.3 --max-yaw 0.3 --host 0.0.0.0
```

Robot `--mode` controls session completion; the GPU server's `--task` selects
the model prompt and checkpoint task settings. For continuous following, stop
the existing VLN server on the GPU host and start its tracking task there:

```bash
cd ~/LightNav-0
CUDA_VISIBLE_DEVICES=0 .venv/bin/lightnav-serve \
  --task tracking --model_path ~/LightNav-0/checkpoints/LightNav-0 \
  --backend vllm_local --gpu_memory_utilization 0.85 --host 0.0.0.0 --port 8050
```

Wait for `READY ... task=tracking` before starting the Jetson command. Merely
writing "Track" in the instruction does not select the server's tracking task.
The tracking mode change was checked with CPU tests; no autonomous motion was
started for validation.

## RoboMaster EP USB connection and direct velocity test

Connect the powered EP controller to a Jetson USB-A host port using a data
cable. On this machine the working connection appeared as DJI `2ca3:001f`,
interface `usb2` with Jetson IP `<jetson-usb-ip>/24` and EP IP `<ep-usb-ip>`.
The initial USB-C connection did not enumerate the EP.

The existing SDK source in `~/RoboMaster-SDK/src/robomaster` was copied
into `.venv-jetson/lib/python3.10/site-packages/robomaster`. `netaddr`,
`netifaces`, and `pybind11` were installed in that environment, and the SDK's
native media binding was compiled for Python 3.10 against the installed FFmpeg
and Opus libraries. No files in the separate robot test checkout were modified.
This SDK addition is separate from `setup_jetson.sh`, which sets up the bench
camera/client dependencies only.

Verify firmware, telemetry, and stationary chassis (zero commands only):

```bash
cd ~/LightNav-0
.venv-jetson/bin/python robot_deploy/scripts/robomaster_velocity_test.py
```

Run a one-second forward velocity test:

```bash
.venv-jetson/bin/python robot_deploy/scripts/robomaster_velocity_test.py \
  --move --speed 0.3 --duration 1.0
```

This uses the reference code's `Robot.initialize(conn_type="rndis")` and
`chassis.drive_speed(x=0.3, y=0, z=0)`, with a client-side stop timer and repeated
zero commands in cleanup. The installed SDK sends speed commands as PUSH packets
without acknowledgements; movement and stopping are checked using fresh chassis
velocity telemetry. This is a direct SDK velocity test, independent of the
LightNav inference pipeline and its read-only preview.

Verified on 2026-10-08: USB ping passed, firmware `01.01.1150` returned,
stationary telemetry passed, and the one-second `0.3 m/s` forward command
produced movement followed by a verified stop. Body velocity telemetry peaked
at **0.46 m/s**, exceeding the commanded speed; the controller does not enforce
a strict physical 0.3 m/s cap in this test. Three regression tests cover
no-ACK returns and stop calls after exceptions. Evidence:
`logs/robomaster-connection.json` and `logs/robomaster-velocity-test.json`.

This Jetson uses Ubuntu 22.04 / JetPack 6 and an Intel RealSense D435i.
The GPU server listens at `gpu-host.example:8050`, with the prepared
`lightnav-host` Conda environment in `~/LightNav-0/.venv`.

The bench client reuses `vln_client` and `vln_mpc` directly, without ROS.
It captures RGB and depth, sends RGB to the GPU, receives waypoints, and solves
the existing MPC locally. The output is a JSON velocity preview, with
`linear_x_m_s` and `angular_z_rad_s`. No robot connection, ROS publisher,
locomotion adapter, or motion command is created.

## Jetson environment

```bash
cd ~/LightNav-0
bash robot_deploy/scripts/setup_jetson.sh
source .venv-jetson/bin/activate
```

Use system Python 3.10 because the installed RealSense binding at
`/usr/local/lib/python3.10/dist-packages/pyrealsense2.so` is built for it.
Setup links only that existing SDK binding into the isolated client environment.
The top-level LightNav model dependencies are not needed on this client.
ROS is not installed by this setup. Run SDK imports and camera access outside
the Codex sandbox; the sandbox blocks USB access.

## Camera verification

```bash
.venv-jetson/bin/python robot_deploy/scripts/jetson_dry_run.py \
  --camera-only --samples 60 --output logs/jetson-camera
```

Both streams are 640×360 at 30 FPS: color `RGB8`, depth `Z16`.
The test checks changing camera timestamps, nonconstant RGB, and nonzero depth.
`summary.json` contains depth validity, distance percentiles, and frame rate;
`rgb.jpg` and `depth.npy` preserve one sample. Depth is verified locally;
LightNav's inference input is RGB only. Depth validation is not obstacle avoidance.

## GPU server

Run on the GPU host:

```bash
cd ~/LightNav-0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lightnav-host
CUDA_VISIBLE_DEVICES=0 lightnav-serve \
  --task vln --model_path ~/LightNav-0/checkpoints/LightNav-0 \
  --backend vllm_local --gpu_memory_utilization 0.85 \
  --host 0.0.0.0 --port 8050
```

Wait for `READY` after warm-up before testing from the Jetson.

## Full bench pipeline

For a continuous on-screen view, run this on the Jetson:

```bash
cd ~/LightNav-0
.venv-jetson/bin/python robot_deploy/scripts/jetson_dry_run.py \
  --server ws://gpu-host.example:8050 \
  --instruction 'Move forward along the clear path.' \
  --view --continuous --view-host 0.0.0.0
```

Open `http://127.0.0.1:8088` in a Jetson browser, or
`http://jetson-host.example:8088` from a computer connected to the Jetson's Tailscale
network. The display shows the camera frame used for each inference alongside
the exact body-frame trajectory in a top-down view: forward up, left left,
with yaw marks at each waypoint. The view needs no camera mounting calibration.
Latency and computed forward/yaw velocities update with each result.
Press Ctrl+C in the terminal to stop; no velocities are published.
`--timeout` in continuous mode is the maximum wait for a new response.

```bash
.venv-jetson/bin/python robot_deploy/scripts/jetson_dry_run.py \
  --server ws://gpu-host.example:8050 \
  --instruction 'Move forward along the clear path.' \
  --samples 5 --timeout 60 --output logs/jetson-pipeline
```

`samples.jsonl` records each response's sequence, capture timestamp, latency,
waypoints, policy stop, and computed velocity. A passing run requires the
requested number of real responses and successful MPC solves (or a zero preview
when the policy requests stop). Transient inference or connection errors are
recorded in `errors.jsonl` while the client reconnects automatically. During
recovery the view clears the old trajectory and velocity. Only valid responses
count toward the requested samples; persistent failures or missing responses hit
the timeout and fail the test. No synthetic inference response substitutes for
the GPU host.

The bench test assumes the robot stays stationary at pose `[0, 0, 0]` and
measured velocity `[0, 0]`. Each response is solved independently, with a
five-step horizon, `dt=0.1 s`, `v_max=0.2 m/s`, and `w_max=0.4 rad/s`.
It verifies camera/network/model/controller computation, not real odometry,
capture-time pose alignment, or sustained 10 Hz control. Live MPC needs real
odometry on the same clock as camera capture. The existing ROS controller emits
`geometry_msgs/TwistStamped`; a robot accepting `Twist` needs the corresponding
message type at the publication boundary, not a locomotion adapter.

Inspect output locally; this script has no option to publish velocities.

## Verification on 2026-10-08

- Client environment installed at `~/LightNav-0/.venv-jetson`;
  dependency consistency check passed.
- D435i: 60 RGB/depth samples passed at 29.97 FPS,
  with 65.3% valid depth pixels in the final sample and median depth 0.446 m.
  Evidence: `logs/jetson-camera-recheck/summary.json`, `rgb.jpg`, `depth.npy`.
- One intervening capture attempt failed during transient USB interface
  contention; the subsequent full camera check passed. Run one camera consumer
  at a time.
- 29 tests passed across the bench validator, client, MPC, and geometry.
- New Python files pass Ruff; repository-wide `make check` reports four existing
  import-order errors in unrelated test files.
- The GPU host SSH port is reachable from the Jetson. SSH authentication from the
  Jetson was unavailable; the server was started through the GPU-host chat.
- Full live pipeline passed against `ws://gpu-host.example:8050`: five real camera
  requests returned ten waypoints each, with 115–158 ms client latency
  (128 ms mean). All five MPC solves succeeded. Forward velocity previews
  ranged from 0.00037 to 0.050 m/s and yaw previews from -0.10 to +0.10 rad/s.
  No motion was published. Evidence: `logs/jetson-pipeline/summary.json` and
  `logs/jetson-pipeline/samples.jsonl`. The Jetson test process exited and released
  the camera; the GPU server remains under the GPU-host chat's control.
- Recovery verification: a 120-response live run passed, including one real
  response timeout followed by automatic reconnection and further valid responses.
  Evidence: `logs/jetson-recovery-smoke/summary.json`, `errors.jsonl`, and
  `samples.jsonl`. No motion was published. Regression tests cover the reported
  missing-RVQ-level error, stale preview clearing, and persistent-error timeouts.
