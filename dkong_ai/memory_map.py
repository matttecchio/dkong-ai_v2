"""Donkey Kong (`dkong`) RAM map — AUTHORITATIVE (provided by user 2026-06-23).

Confirmed consistent with empirical discovery (lives/screen_id/level matched;
mario_x/mario_y verified by movement correlation). Score lives in the on-screen
tile RAM at 0x77xx (stride 0x20 because the DK monitor is rotated); each digit
byte is a tile code where the low nibble is the digit ('0'=0x10 .. '9'=0x19).
"""

ADDR = {
    "game_start":   0x622C,   # 1 once a real game/level is underway
    "eol_counter":  0x6388,   # end-of-level counter
    "bonus":        0x62B1,
    "lives":        0x6228,
    "level":        0x6229,
    "screen_id":    0x6227,   # 1=barrels 2=pie 3=elevator 4=rivet
    "is_dead":      0x6200,   # INVERTED: 1=alive, 0=dead/inactive (verified
                              # 2026-07-04: probes; corpses read 0). Death is
                              # detected via lives drop in mame_env._reward —
                              # never gate on this name's face value.
    "is_jumping":   0x6216,
    "jump_dir":     0x6211,
    "has_hammer":   0x6217,
    "bonus_item":   0x6343,
    "mario_x":      0x6203,   # +right (empirically confirmed)
    "mario_y":      0x6205,   # smaller = higher (empirically confirmed)
    # Score digits (tile RAM, stride 0x20). DK scores are multiples of 100, so
    # the tens digit (0x7701) is always 0 AND lands in the volatile timer region
    # -> excluded. Hundreds..hundred-thousands read clean '0' tiles (0x10).
    "score_100":    0x7721,
    "score_1k":     0x7741,
    "score_10k":    0x7761,
    "score_100k":   0x7781,
    # Barrel object array: 6 slots at 0x6700, stride 0x20.
    # Per slot: +0x00=status (0=inactive,1=rolling,2=deploying), +0x01=crazy
    # (wild barrel bouncing vertically down the board), +0x02=blue (rolls into
    # the oil drum -> fireball), +0x03=x, +0x05=y.
    # Confirmed from Don Hodges 2008 Z80 disassembly; same coord system as mario_y.
    "barrel0_st": 0x6700, "barrel0_x": 0x6703, "barrel0_y": 0x6705,
    "barrel1_st": 0x6720, "barrel1_x": 0x6723, "barrel1_y": 0x6725,
    "barrel2_st": 0x6740, "barrel2_x": 0x6743, "barrel2_y": 0x6745,
    "barrel3_st": 0x6760, "barrel3_x": 0x6763, "barrel3_y": 0x6765,
    "barrel4_st": 0x6780, "barrel4_x": 0x6783, "barrel4_y": 0x6785,
    "barrel5_st": 0x67A0, "barrel5_x": 0x67A3, "barrel5_y": 0x67A5,
    "barrel0_crazy": 0x6701, "barrel0_blue": 0x6702,
    "barrel1_crazy": 0x6721, "barrel1_blue": 0x6722,
    "barrel2_crazy": 0x6741, "barrel2_blue": 0x6742,
    "barrel3_crazy": 0x6761, "barrel3_blue": 0x6762,
    "barrel4_crazy": 0x6781, "barrel4_blue": 0x6782,
    "barrel5_crazy": 0x67A1, "barrel5_blue": 0x67A2,
    # Fireballs (flame enemies that chase Mario): 5 slots at 0x6400, stride 0x20.
    # +0x00=status (0=inactive,1=active), +0x03=x, +0x05=y.
    # Barrel stage typically has 1 active; track all 5 for completeness.
    "fireball0_st": 0x6400, "fireball0_x": 0x6403, "fireball0_y": 0x6405,
    "fireball1_st": 0x6420, "fireball1_x": 0x6423, "fireball1_y": 0x6425,
    "fireball2_st": 0x6440, "fireball2_x": 0x6443, "fireball2_y": 0x6445,
    "fireball3_st": 0x6460, "fireball3_x": 0x6463, "fireball3_y": 0x6465,
    "fireball4_st": 0x6480, "fireball4_x": 0x6483, "fireball4_y": 0x6485,
    # Hammer sprite at #6A1C-#6A1F. Pattern: +0=X, +3=Y. has_hammer=0x6217.
    "hammer_x": 0x6A1C, "hammer_y": 0x6A1F, "has_hammer": 0x6217,
    "is_jumping": 0x6216,
    # Internal difficulty (1-5): starts at the level number, ramps on a timer
    # within each board; indexes the game's barrel-steering/fireball aggression
    # tables. TRACKED ONLY (episode CSV/TB) — deliberately NOT in the RAM
    # observation vector yet; fold into the next obs-space change.
    "difficulty": 0x6380,
}

# Bytes the bridge ships each step, in this order (must match bridge WATCH_ADDRS).
WATCH_ORDER = [
    "lives", "screen_id", "mario_y", "mario_x", "is_dead", "game_start",
    "score_100", "score_1k", "score_10k", "score_100k",
    "barrel0_st", "barrel0_x", "barrel0_y",
    "barrel1_st", "barrel1_x", "barrel1_y",
    "barrel2_st", "barrel2_x", "barrel2_y",
    "barrel3_st", "barrel3_x", "barrel3_y",
    "barrel4_st", "barrel4_x", "barrel4_y",
    "barrel5_st", "barrel5_x", "barrel5_y",
    "fireball0_st", "fireball0_x", "fireball0_y",
    "fireball1_st", "fireball1_x", "fireball1_y",
    "fireball2_st", "fireball2_x", "fireball2_y",
    "fireball3_st", "fireball3_x", "fireball3_y",
    "fireball4_st", "fireball4_x", "fireball4_y",
    "hammer_x", "hammer_y", "has_hammer",
    "is_jumping",
    # Appended (order-stable): barrel type flags. crazy = wild barrel bouncing
    # vertically down the board; blue = heads for the oil drum (spawns fireball).
    "barrel0_crazy", "barrel0_blue",
    "barrel1_crazy", "barrel1_blue",
    "barrel2_crazy", "barrel2_blue",
    "barrel3_crazy", "barrel3_blue",
    "barrel4_crazy", "barrel4_blue",
    "barrel5_crazy", "barrel5_blue",
    "difficulty",
]

WATCH_ADDRS = [ADDR[name] for name in WATCH_ORDER]

_SCORE_PLACE = {"score_100": 100, "score_1k": 1000,
                "score_10k": 10000, "score_100k": 100000}


def decode_score(state: dict):
    """Decode the 4 score digit tiles -> integer, or None if any tile isn't a
    valid digit (mid-update / not displayed) so the caller skips the delta.

    A digit's value is the tile's low nibble. The high nibble is 0x0 in live
    play ('0'=0x00) or 0x1 in some HUD states ('0'=0x10); anything else is
    garbage."""
    total = 0
    for name, place in _SCORE_PLACE.items():
        raw = state[name]
        hi, lo = raw >> 4, raw & 0x0F
        if hi not in (0, 1) or lo > 9:
            return None
        total += lo * place
    return total
