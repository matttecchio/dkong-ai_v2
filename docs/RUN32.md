# Run 32 shelf — obs candidates (deploy on run-31 wall or post-first-clear)

Accumulating list; each item is DATA-READY unless noted. Bundle into one
obs bump (new feature dim = new run number; letters = restarts).

- **Per-barrel WILD flag** (user request 2026-07-18): `barrel{i}_crazy`
  (+0x01) is ALREADY in the 62-entry watch — the env sees it every step;
  feature wiring only (+6 dims). Pair with the existing `difficulty`
  feature (in obs since run 28) so "wild barrel in play at difficulty D"
  is directly learnable. Wild barrels bounce vertically and defy
  girder-based dodging; pro play treats them as a separate threat class.
- **Projected-occupancy channel** (user doctrine: "track all barrels and
  imagine where they may go in the next ~6s"): paint predicted barrel
  positions into image channel 1 (or a channel 2). Prototype pending —
  the biggest remaining perception gap. Wild flag above feeds this
  (wild = vertical projection, not girder-following).
- **Blue-barrel flag as a feature** (+6 dims, also already watched,
  +0x02): blue barrels preferentially take ladders and become fireballs
  at the oil can — pro players track them individually. Dashboard shows
  them (blue tint) since 2026-07-18; policy can't see the flag yet.
- Frameskip-2 experiment: POST-FIRST-CLEAR only (doubles sample cost;
  sharpens jump timing). Not an obs change; separate decision.

NOTE: bustart success-record threshold (h>=68) could drop to ~55 to make
g3-touch bottom-ups filmable/harvestable — code change, not obs; can ship
in any run-31 letter.
