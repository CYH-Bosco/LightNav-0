#!/usr/bin/env bash
# Persistent remote-access settings for this Jetson. No reboot or link restart.
set -euo pipefail

if (( EUID != 0 )); then
    echo 'Run with sudo from your Jetson terminal; administrator access is required.' >&2
    exit 1
fi

if (( $# != 2 )); then
    echo "Usage: $0 <wifi-profile-uuid> <wifi-interface>" >&2
    exit 2
fi
wifi_uuid="$1"
wifi_iface="$2"
# Validate the chosen profile before changing anything.
nmcli -g connection.id connection show "$wifi_uuid"

# Disable sleep even before login or when the desktop power daemon is absent.
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
nmcli connection modify "$wifi_uuid" \
    connection.autoconnect yes \
    connection.autoconnect-retries 0 \
    802-11-wireless.powersave 2
iw dev "$wifi_iface" set power_save off
systemctl enable --now tailscaled.service

echo 'Verification:'
systemctl is-enabled sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target || true
nmcli -f connection.autoconnect,connection.autoconnect-retries,802-11-wireless.powersave connection show "$wifi_uuid"
iw dev "$wifi_iface" get power_save
systemctl is-active tailscaled.service
tailscale ip -4
