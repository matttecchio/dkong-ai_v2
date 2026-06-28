"""Discovery tool — run FIRST once the dkong ROM is available.

Launches MAME with the bridge, prints the handshake (screen geometry + every
input port/field MAME found), and grabs a couple of frames. Use the printed
field tags to fill scripts/bridge.lua CONTROLLED_FIELDS, and confirm RAM
addresses with the cheatfind plugin.

Usage:
    python -m dkong_ai.probe --rom-dir /path/to/roms
"""
import argparse

from .mame_env import DonkeyKongEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-dir", required=True)
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--headless", action="store_true", default=True)
    args = ap.parse_args()

    env = DonkeyKongEnv(rom_dir=args.rom_dir, port=args.port,
                        headless=args.headless)
    env._launch_mame()
    env._connect()
    env._read_handshake()
    g = env._geom
    print(f"\n== HANDSHAKE ==\nscreen: {g['w']}x{g['h']}  bpp={g['bpp']}  "
          f"frameskip={g['frameskip']}")
    print(f"\n== {len(g['fields'])} INPUT FIELDS (port|field) ==")
    for f in sorted(g["fields"]):
        print("  ", f)

    print("\n== first frames ==")
    env._sock.sendall(bytes([0]))
    for i in range(3):
        ram, pix = env._read_obs()
        print(f"  frame {i}: ram={ram.hex()}  pixels={len(pix)} bytes")
        env._sock.sendall(bytes([0]))
    env.close()
    print("\nDone. Wire CONTROLLED_FIELDS in scripts/bridge.lua from the fields above.")


if __name__ == "__main__":
    main()
