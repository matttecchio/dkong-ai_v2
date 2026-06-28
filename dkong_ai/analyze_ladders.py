"""Build a height->target-x 'corridor' from the expert trajectory
(/tmp/dk_ladder.log): for each height band, the median x the expert occupied.
This encodes the zig-zag route up the barrel board (which side the next ladder
is on) without needing fragile discrete-ladder detection on the slanted girders.

Saves artifacts/expert_corridor.json for the env's route-guidance reward.
"""
import json
import os
import statistics
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dk_ladder.log"
BAND = 12   # height-band size in pixels
BASE_Y = 240

by_band = {}
for line in open(LOG):
    fr, x, y = (int(v) for v in line.split())
    if y <= 0:
        continue
    h = BASE_Y - y
    if h < 0:
        continue
    by_band.setdefault(h // BAND, []).append(x)

corridor = []
for b in sorted(by_band):
    xs = by_band[b]
    corridor.append({"h_lo": b * BAND, "h_hi": (b + 1) * BAND,
                     "x_med": int(statistics.median(xs)),
                     "x_lo": min(xs), "x_hi": max(xs), "n": len(xs)})

print("height band   target_x(median)   x-range        samples")
for c in corridor:
    print(f"  {c['h_lo']:3d}-{c['h_hi']:3d}      x={c['x_med']:3d}            "
          f"[{c['x_lo']:3d},{c['x_hi']:3d}]      {c['n']}")

out = os.path.join(os.path.dirname(__file__), "..", "artifacts", "expert_corridor.json")
out = os.path.abspath(out)
with open(out, "w") as f:
    json.dump(corridor, f, indent=2)
print(f"\nsaved corridor -> {out}")
