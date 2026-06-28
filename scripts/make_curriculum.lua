-- Replay the expert .inp and SAVE a MAME save-state each time the expert first
-- crosses a height threshold on the FIRST barrel board (screen_id==1). Produces
-- curric_0..N (low->high) for the env's self-curriculum: training starts some
-- episodes from these upper-board states so the agent practices the top + finish.
-- Run with -state_directory artifacts/states so training can load them.
local LOG = os.getenv("DK_CURRIC_LOG") or "/tmp/dk_curric.log"
local mac = manager.machine
local f = io.open(LOG, "w")
local cpu, space = nil, nil
-- mario_y thresholds (smaller = higher). curric_(i-1) saved when first reached.
-- Four entries in the wall zone (y=205..190, heights 35-50) put curriculum
-- states right at the 2nd-girder left-traverse the agent has never solved.
local THRESH = {205, 200, 195, 190, 188, 175, 145, 115, 85, 58}
local saved = {}

emu.register_frame_done(function()
  if not cpu then
    cpu = mac.devices[":maincpu"]; if not cpu then return end
    space = cpu.spaces["program"]
  end
  if space:readv_u8(0x6227) ~= 1 then return end   -- first barrel board only
  local y = space:readv_u8(0x6205)
  if y == 0 then return end
  for i, th in ipairs(THRESH) do
    if not saved[i] and y <= th then
      saved[i] = true
      local name = "curric_" .. (i - 1)
      local ok, err = pcall(function() mac:save(name) end)
      f:write(string.format("saved %s at y=%d (height %d) -> %s %s\n",
              name, y, 240 - y, tostring(ok), tostring(err)))
      f:flush()
    end
  end
end)
