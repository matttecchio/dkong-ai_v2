#!/usr/bin/env bash
# Watch a recorded game .inp WINDOWED at normal speed.
#   ./scripts/playback.sh artifacts/recordings/dkong_pXXXX_YYYY.inp
# (No bridge script: playback drives all inputs from the .inp itself.)
set -euo pipefail
INP="${1:?usage: playback.sh <file.inp>}"
DIR="$(cd "$(dirname "$INP")" && pwd)"
exec mame dkong \
  -rompath ./roms \
  -input_directory "$DIR" \
  -playback "$(basename "$INP")" \
  -skip_gameinfo -exit_after_playback -window
