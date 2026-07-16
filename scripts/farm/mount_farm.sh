#!/bin/bash
# Mount the farm's state share in WSL. Credentials must be cached on the
# WINDOWS side first:  net use \\192.168.20.59\dkstates /user:<user> *
set -e
HOST=${1:-192.168.20.59}
MOUNT=${2:-/mnt/dkfarm}
sudo mkdir -p "$MOUNT"
mountpoint -q "$MOUNT" || sudo mount -t drvfs "\\\\$HOST\\dkstates" "$MOUNT"
touch "$MOUNT/write_test" && rm "$MOUNT/write_test"
echo "farm share mounted and writable at $MOUNT"
