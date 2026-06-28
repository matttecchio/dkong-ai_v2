-- Read-only logger for verifying .inp playback fidelity. Logs mario x/y, the
-- stage (screen_id), and lives over time. No input injection (playback drives
-- inputs from the .inp). Writes to DK_PLAYBACK_LOG.
local LOG = os.getenv("DK_PLAYBACK_LOG") or "/tmp/dk_playback.log"
local mac = manager.machine
local f = io.open(LOG, "w")
local cpu, space, n = nil, nil, 0
local last_screen = -1

emu.register_frame_done(function()
  if not cpu then
    cpu = mac.devices[":maincpu"]
    if not cpu then return end
    space = cpu.spaces["program"]
  end
  n = n + 1
  local screen = space:readv_u8(0x6227)
  if screen ~= last_screen then
    f:write(string.format("frame %d STAGE CHANGE screen_id=%d\n", n, screen))
    f:flush()
    last_screen = screen
  end
  if n % 120 == 0 then
    f:write(string.format("frame %d mario=(%d,%d) screen=%d lives=%d\n",
      n, space:readv_u8(0x6203), space:readv_u8(0x6205), screen, space:readv_u8(0x6228)))
    f:flush()
  end
end)
