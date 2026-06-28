"""Dump the DK barrel-board tilemap from live MAME and identify broken ladders.

Launches one MAME instance with DK_RAMDUMP="0x7400:0x7800", plays the intro,
waits for the barrel board (screen_id==1), then reads the tilemap and prints:
  - A 32×32 ASCII heat-map of non-zero tile codes
  - Columns that contain ladder tiles and whether they span a full girder gap

Run with: python -m dkong_ai.find_broken_ladders --rom-dir ./roms
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np

# Patch DK_RAMDUMP into the env BEFORE importing mame_env, so _launch_mame
# picks it up.  We want the 0x7400-0x7800 range appended to every obs.
os.environ["DK_RAMDUMP"] = "0x7400:0x7800"

from dkong_ai.mame_env import DonkeyKongEnv   # noqa: E402 (after env var set)
from dkong_ai import memory_map               # noqa: E402

TILEMAP_LEN = 0x7800 - 0x7400   # 1024 bytes
WATCH_LEN   = len(memory_map.WATCH_ORDER)    # 31 bytes (no EXTRACT flag)

# DK tilemap in ROM is 32 cols × 32 rows (stride=32, column-major based on
# score-tile analysis). tile(col, row) = ram[col*32 + row]. After the 90°
# cabinet rotation: screen_x ≈ col*8, screen_y ≈ row*8.
TILEMAP_COLS = 32
TILEMAP_ROWS = 32


def extract_tilemap(ram: bytes) -> np.ndarray:
    """Return 32×32 uint8 array of tile codes from the appended RAM dump."""
    blob = ram[WATCH_LEN : WATCH_LEN + TILEMAP_LEN]
    arr  = np.frombuffer(blob, dtype=np.uint8).reshape(TILEMAP_COLS, TILEMAP_ROWS)
    return arr                     # arr[col, row]; screen_x=col*8, screen_y=row*8


def find_ladder_tile_code(tm: np.ndarray) -> set[int]:
    """Guess ladder tile codes by looking for vertical stripes at known complete-
    ladder x positions (x≈53,131,143,67,147 in game coords = screen_x / 8 rounded)."""
    known_x_game = [53, 67, 131, 143, 147]
    candidates: set[int] = set()
    for gx in known_x_game:
        col = round(gx / 8)
        if 0 <= col < TILEMAP_COLS:
            for code in tm[col, :]:
                if code not in (0x00, 0xFF, 0x10):   # skip blank / solid / bg
                    candidates.add(int(code))
    return candidates


def print_tilemap(tm: np.ndarray, ladder_codes: set[int]) -> None:
    print("\nTilemap (32 cols = screen x, 32 rows = screen y; L=ladder, .=blank, #=other):")
    print("     " + "".join(f"{c:2d}" for c in range(TILEMAP_COLS)))
    for row in range(TILEMAP_ROWS):
        row_str = f"y{row*8:3d} "
        for col in range(TILEMAP_COLS):
            code = tm[col, row]
            if code == 0:
                row_str += " ."
            elif code in ladder_codes:
                row_str += " L"
            else:
                row_str += " #"
        print(row_str)


def classify_ladders(tm: np.ndarray, ladder_codes: set[int]) -> None:
    """For each column that has ladder tiles, check if it spans a full girder gap."""
    # Known girder y-positions in screen coords (approximate from game mario_y data):
    # mario_y 240→screen_y 224, mario_y 196→screen_y 180, etc.
    # Screen_y = mario_y * (224/240) ≈ mario_y * 0.933
    girder_sy = [round(y * 224/240) for y in [240, 196, 162, 128, 96, 58]]
    gap_rows  = [(round(a/8), round(b/8)) for a, b in zip(girder_sy, girder_sy[1:])]

    print("\nLadder column analysis (game_x = col*8):")
    print(f"{'game_x':>8}  {'col':>4}  {'y_range':>14}  type")
    for col in range(TILEMAP_COLS):
        col_tiles = {row for row in range(TILEMAP_ROWS) if tm[col, row] in ladder_codes}
        if not col_tiles:
            continue
        min_row, max_row = min(col_tiles), max(col_tiles)
        game_x = col * 8
        y_lo, y_hi = min_row * 8, max_row * 8

        # Check if this column spans across any girder gap.
        complete = False
        for (ga, gb) in gap_rows:
            if min_row <= ga and max_row >= gb:
                complete = True
                break
        label = "COMPLETE" if complete else "BROKEN (stub)"
        print(f"{game_x:>8}  {col:>4}  y={y_lo:>3}-{y_hi:>3}      {label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--port", type=int, default=5200)
    args = ap.parse_args()

    env = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port,
                        record=False, headless=True)
    try:
        print("Launching MAME and playing intro…")
        obs = env.reset()   # triggers full intro (~19s)
        # Advance a few frames until we're on the barrel board.
        for step in range(200):
            obs, _, done, _, info = env.step(0)  # noop
            s = info["state"]
            if s.get("screen_id") == 1 and s.get("mario_y", 0) > 50:
                break
            if done:
                obs = env.reset()

        # Read one more obs to get a clean frame with the barrel board loaded.
        # The RAM blob is in env._rxbuf after step; re-read via a noop exchange.
        env._sock.sendall(bytes([0x00]))
        ram, pix = env._read_obs()

        print(f"Barrel board frame captured. RAM blob length: {len(ram)}")
        if len(ram) < WATCH_LEN + TILEMAP_LEN:
            print("ERROR: tilemap not in RAM blob — was DK_RAMDUMP set correctly?")
            sys.exit(1)

        tm = extract_tilemap(ram)
        ladder_codes = find_ladder_tile_code(tm)
        print(f"Detected ladder tile codes: {sorted(ladder_codes)}")

        print_tilemap(tm, ladder_codes)
        classify_ladders(tm, ladder_codes)

    finally:
        env.close()


if __name__ == "__main__":
    main()
