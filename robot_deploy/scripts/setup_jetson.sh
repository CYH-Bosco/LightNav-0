#!/usr/bin/env bash
set -eo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/../.." && pwd)"
cd "${repo_dir}"
# The preinstalled native RealSense binding is built for system Python 3.10.
/usr/bin/python3 -m venv --without-pip .venv-jetson
test -f /usr/local/lib/python3.10/dist-packages/pyrealsense2.so
ln -sfn /usr/local/lib/python3.10/dist-packages/pyrealsense2.so \
    .venv-jetson/lib/python3.10/site-packages/pyrealsense2.so
/usr/bin/python3 -m pip --python .venv-jetson/bin/python install \
    -r robot_deploy/scripts/jetson-requirements.txt
.venv-jetson/bin/python -c 'import aiohttp, casadi, cv2, numpy, pyrealsense2; print("Jetson client dependencies ready")'
