#!/bin/bash
# Sync curriculum .sta files to the farm's state share.
MOUNT=${1:-/mnt/dkfarm}
[ -d "$MOUNT" ] || { echo "mount $MOUNT missing"; exit 1; }
rsync -r --checksum --no-perms --no-owner --no-group --inplace --quiet --include='*.sta' --exclude='*' artifacts/backward_dense14/ "$MOUNT/dkong/"
rsync -r --checksum --no-perms --no-owner --no-group --inplace --quiet artifacts/states/dkong/ "$MOUNT/dkong/" 2>/dev/null
echo "states synced to $MOUNT"
