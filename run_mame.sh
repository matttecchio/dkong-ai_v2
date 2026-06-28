#!/usr/bin/env bash
# Launch MAME + dkong with the bridge, WINDOWED, for interactive debugging.
# Pass the rom dir as $1 (defaults to ./roms). Useful for watching the game and
# for running the cheatfind plugin to confirm RAM addresses.
set -euo pipefail
ROMDIR="${1:-./roms}"
exec mame dkong \
  -rompath "$ROMDIR" \
  -autoboot_script scripts/bridge.lua \
  -autoboot_delay 0 \
  -skip_gameinfo \
  -plugin cheatfind \
  -window
