-- Mine world-class route states from .inp playback: on the FIRST barrel board
-- (screen_id==1 AND level==1), save a MAME state every SNAP_EVERY frames while
-- Mario is alive, logging frame/x/y/height/difficulty per snapshot so the
-- exporter can curate (pro play milks points slowly — late states have burned
-- bonus timer; frame index is the timer proxy). Run with
-- -state_directory artifacts/states; states land as wc_NNN in .../dkong/.
local LOG = os.getenv("DK_WC_LOG") or "artifacts/wc_mine.log"
local SNAP_EVERY = tonumber(os.getenv("DK_WC_EVERY") or "45")
local mac = manager.machine
local f = io.open(LOG, "w")
local cpu, space = nil, nil
local n, since, frames = 0, 0, 0
local done = false

emu.register_frame_done(function()
  if done then return end
  if not cpu then
    cpu = mac.devices[":maincpu"]; if not cpu then return end
    space = cpu.spaces["program"]
  end
  frames = frames + 1
  local screen = space:readv_u8(0x6227)
  local level  = space:readv_u8(0x6229)
  if level > 1 then done = true; f:write("done: left level 1\n"); f:flush(); return end
  if screen ~= 1 then return end
  local y = space:readv_u8(0x6205)
  local alive = space:readv_u8(0x6200) == 1
  if y == 0 or not alive then since = 0; return end   -- dead/void: don't snap
  since = since + 1
  if since >= SNAP_EVERY then
    since = 0
    local x = space:readv_u8(0x6203)
    local d = space:readv_u8(0x6380)
    local name = string.format("wc_%03d", n)
    local ok = pcall(function() mac:save(name) end)
    f:write(string.format("%s frame=%d x=%d y=%d h=%d diff=%d ok=%s\n",
            name, frames, x, y, 240 - y, d, tostring(ok)))
    f:flush()
    n = n + 1
  end
end)
