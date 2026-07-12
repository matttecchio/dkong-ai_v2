"""Mechanical enforcement of the WATCH_ORDER / WATCH_ADDRS invariant.

Both memory_map.WATCH_ORDER (Python) and the WATCH_ADDRS table in scripts/bridge.lua
(Lua) must list exactly the same 59 RAM addresses in the same order. A mismatch silently
corrupts every RAM observation: wrong feature in wrong slot with no runtime error.

Run with: python -m pytest tests/test_bridge_sync.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
BRIDGE_LUA = ROOT / "scripts" / "bridge.lua"
EXPECTED_COUNT = 62


def _parse_lua_addrs():
    text = BRIDGE_LUA.read_text()
    m = re.search(r"local WATCH_ADDRS\s*=\s*\{(.*?)\}", text, re.DOTALL)
    assert m, "Could not find 'local WATCH_ADDRS = {...}' block in bridge.lua"
    block = m.group(1)
    addrs = []
    for line in block.splitlines():
        bare = re.sub(r"--.*", "", line)        # strip Lua comments
        addrs.extend(int(x, 16) for x in re.findall(r"0x[0-9A-Fa-f]+", bare))
    return addrs


def test_watch_order_length():
    from dkong_ai.memory_map import WATCH_ORDER
    assert len(WATCH_ORDER) == EXPECTED_COUNT, (
        f"memory_map.WATCH_ORDER has {len(WATCH_ORDER)} entries, expected {EXPECTED_COUNT}"
    )


def test_watch_addrs_length():
    addrs = _parse_lua_addrs()
    assert len(addrs) == EXPECTED_COUNT, (
        f"bridge.lua WATCH_ADDRS has {len(addrs)} entries, expected {EXPECTED_COUNT}"
    )


def test_watch_order_matches_bridge():
    from dkong_ai.memory_map import WATCH_ORDER, ADDR
    lua_addrs = _parse_lua_addrs()
    py_addrs = [ADDR[name] for name in WATCH_ORDER]
    mismatches = [
        f"  [{i}] py={name}=0x{py_addrs[i]:04x}  lua=0x{lua_addrs[i]:04x}"
        for i, name in enumerate(WATCH_ORDER)
        if i < len(lua_addrs) and lua_addrs[i] != py_addrs[i]
    ]
    assert not mismatches, "WATCH_ORDER / WATCH_ADDRS mismatch:\n" + "\n".join(mismatches)


def test_appended_entries_positions():
    """Entries are append-only (order-stable): is_jumping stayed at index 46
    where run-24's fix put it, and the 12 barrel type flags follow it."""
    from dkong_ai.memory_map import WATCH_ORDER, ADDR
    assert WATCH_ORDER[46] == "is_jumping", (
        f"Expected 'is_jumping' at WATCH_ORDER[46], got '{WATCH_ORDER[46]}'"
    )
    expected_tail = [f"barrel{i}_{kind}" for i in range(6)
                     for kind in ("crazy", "blue")] + ["difficulty",
                     "bonus_timer", "mario_facing"]
    assert WATCH_ORDER[47:] == expected_tail, (
        f"Expected barrel type flags + difficulty after is_jumping, "
        f"got {WATCH_ORDER[47:]}"
    )
    lua_addrs = _parse_lua_addrs()
    assert lua_addrs[46] == ADDR["is_jumping"]
    assert lua_addrs[-3:] == [ADDR["difficulty"], ADDR["bonus_timer"],
                              ADDR["mario_facing"]]
