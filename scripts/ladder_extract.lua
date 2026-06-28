-- Log Mario's per-frame x/y (+ screen_id) during .inp playback, so we can find
-- the ladder x-positions from where the expert climbs. Read-only; writes to
-- DK_LADDER_LOG. Stops logging once we leave the first barrel board.
local LOG = os.getenv("DK_LADDER_LOG") or "/tmp/dk_ladder.log"
local mac = manager.machine
local f = io.open(LOG, "w")
local cpu, space, n = nil, nil, 0

emu.register_frame_done(function()
  if not cpu then
    cpu = mac.devices[":maincpu"]
    if not cpu then return end
    space = cpu.spaces["program"]
  end
  n = n + 1
  local screen = space:readv_u8(0x6227)
  local x = space:readv_u8(0x6203)
  local y = space:readv_u8(0x6205)
  -- only the first barrel board (screen 1); once it advances past 1, stop.
  if screen == 1 then
    f:write(string.format("%d %d %d\n", n, x, y))
  end
  if n % 240 == 0 then f:flush() end
end)
