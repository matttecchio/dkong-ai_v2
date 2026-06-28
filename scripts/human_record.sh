#!/usr/bin/env bash
# Play Donkey Kong yourself (windowed) and RECORD it to a .inp we can learn from.
#   ./scripts/human_record.sh demo1        # records artifacts/recordings/human_demo1.inp
#
# Default MAME keys: arrow keys = joystick, Left-Ctrl = jump (Button 1),
#   5 = insert coin, 1 = 1-player start, Esc/Tab = menu/quit.
# Play a few full games (even imperfect). Press Esc to finish so the .inp flushes.
set -euo pipefail
NAME="${1:-demo}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/artifacts/recordings"
mkdir -p "$DIR"
echo "Recording to $DIR/human_${NAME}.inp  — play, then press Esc to finish."
exec mame dkong \
  -rompath ./roms \
  -input_directory "$DIR" \
  -record "human_${NAME}" \
  -skip_gameinfo -window
