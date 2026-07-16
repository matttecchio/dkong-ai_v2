#!/bin/bash
# Sync curriculum .sta files to the farm's state share.
MOUNT=${1:-/mnt/dkfarm}
[ -d "$MOUNT" ] || { echo "mount $MOUNT missing"; exit 1; }
rsync -a --include='*.sta' --exclude='*' artifacts/backward_dense14/ "$MOUNT/"
rsync -a artifacts/states/dkong/ "$MOUNT/" 2>/dev/null
echo "states synced to $MOUNT"
