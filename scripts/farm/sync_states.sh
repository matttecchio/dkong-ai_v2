#!/bin/bash
# Sync curriculum .sta files to the farm's state share.
MOUNT=${1:-/mnt/dkfarm}
[ -d "$MOUNT" ] || { echo "mount $MOUNT missing"; exit 1; }
rsync -rt --no-perms --no-owner --no-group --inplace --include='*.sta' --exclude='*' artifacts/backward_dense14/ "$MOUNT/"
rsync -rt --no-perms --no-owner --no-group --inplace artifacts/states/dkong/ "$MOUNT/" 2>/dev/null
echo "states synced to $MOUNT"
