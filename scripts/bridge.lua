--[[
  bridge.lua — MAME <-> Python lock-step RL bridge for Donkey Kong.

  Load with:  mame dkong -autoboot_script scripts/bridge.lua -autoboot_delay 0

  MAME is the SOCKET SERVER (listens); Python connects as the client and speaks
  FIRST so MAME never enters its lock-step loop without a confirmed peer.

  Handshake:
    Python -> MAME:  one byte  'H'  (hello)
    MAME  -> Python: TEXT, two newline-terminated lines:
        "HELLO W=<w> H=<h> BPP=4 FRAMESKIP=<n> NFIELDS=<k>\n"
        "FIELDS=<port|field>;;<port|field>;;...\n"
      and logs the same to the console + DK_BRIDGE_LOG.

  Per step (lock-step):
    MAME -> Python:  [4-byte BE len][payload]; payload = [2-byte ram_len][ram][pixels]
                     pixels = W*H 32-bit ARGB; ram = WATCH_ADDRS bytes in order
    Python -> MAME:  [1 byte] bitmask over CONTROLLED_FIELDS (bit i -> field i)
  MAME stalls in the frame callback until the action byte arrives -> clean MDP.

  CONTROLLED_FIELDS / WATCH_ADDRS are filled from the discovery run.
--]]

-- MAME re-executes -autoboot_script on every soft_reset. Register our callbacks
-- and socket only once; on re-exec just fall through (the originals persist and
-- keep serving). Without this, duplicate frame handlers corrupt the protocol.
if _G.__dk_bridge_init then
  emu.print_info("[bridge] re-exec (soft_reset); keeping existing bridge")
  return
end
_G.__dk_bridge_init = true

local PORT      = tonumber(os.getenv("DK_BRIDGE_PORT")) or 5000
local FRAMESKIP = tonumber(os.getenv("DK_FRAMESKIP"))   or 4
local LOGPATH   = os.getenv("DK_BRIDGE_LOG") or "/tmp/dk_bridge.log"
-- Extraction mode (behavioral cloning): during .inp PLAYBACK, don't apply
-- actions (playback drives inputs); instead append the CURRENT play-input
-- bitmask to each obs so Python can record (pixels -> expert action) pairs.
local EXTRACT   = os.getenv("DK_EXTRACT") ~= nil

-- Play controls; action bitmask bit i -> entry i (bit 0 = Left ... bit 4 = Jump).
local PLAY_FIELDS = {
  { port = ":IN0", field = "P1 Left"     },  -- bit 0
  { port = ":IN0", field = "P1 Right"    },  -- bit 1
  { port = ":IN0", field = "P1 Up"       },  -- bit 2
  { port = ":IN0", field = "P1 Down"     },  -- bit 3
  { port = ":IN0", field = "P1 Button 1" },  -- bit 4 (jump)
}
-- System inputs the env pulses to start a game.
local COIN  = { port = ":IN2", field = "Coin 1" }
local START = { port = ":IN2", field = "1 Player Start" }
local ACT_COIN, ACT_START, ACT_RESET, ACT_QUIT = 0xF1, 0xF2, 0xFE, 0xFD
local ACT_SAVE, ACT_LOAD = 0xFC, 0xFB

-- Reward addresses; ORDER MUST MATCH memory_map.WATCH_ORDER.
local WATCH_ADDRS = {
  0x6228,  -- lives
  0x6227,  -- screen_id (1=barrels..4=rivet)
  0x6205,  -- mario_y (smaller = higher)
  0x6203,  -- mario_x
  0x6200,  -- is_dead
  0x622C,  -- game_start
  0x7721,  -- score_100  (tile: digit = low nibble)
  0x7741,  -- score_1k
  0x7761,  -- score_10k
  0x7781,  -- score_100k
  -- Barrel object array: 6 slots at 0x6700, stride 0x20.
  -- +0=status (0=inactive,1=rolling,2=deploying), +3=x, +5=y.
  0x6700, 0x6703, 0x6705,  -- barrel 0
  0x6720, 0x6723, 0x6725,  -- barrel 1
  0x6740, 0x6743, 0x6745,  -- barrel 2
  0x6760, 0x6763, 0x6765,  -- barrel 3
  0x6780, 0x6783, 0x6785,  -- barrel 4
  0x67A0, 0x67A3, 0x67A5,  -- barrel 5
  -- Fireball/flame enemies: 5 slots at 0x6400, stride 0x20.
  -- +0=status (0=inactive,1=active), +3=x, +5=y.
  0x6400, 0x6403, 0x6405,  -- fireball 0
  0x6420, 0x6423, 0x6425,  -- fireball 1
  0x6440, 0x6443, 0x6445,  -- fireball 2
  0x6460, 0x6463, 0x6465,  -- fireball 3
  0x6480, 0x6483, 0x6485,  -- fireball 4
  -- Hammer sprite (#6A1C-#6A1F): +0=X, +3=Y. has_hammer at 0x6217.
  0x6A1C, 0x6A1F, 0x6217,  -- hammer x, hammer y, has_hammer
  0x6216,                   -- is_jumping (non-zero while Mario is in a jump arc)
  -- Appended: barrel type flags, +1=crazy (wild/bouncing), +2=blue (oil-drum).
  0x6701, 0x6702,  -- barrel 0 crazy, blue
  0x6721, 0x6722,  -- barrel 1
  0x6741, 0x6742,  -- barrel 2
  0x6761, 0x6762,  -- barrel 3
  0x6781, 0x6782,  -- barrel 4
  0x67A1, 0x67A2,  -- barrel 5
  0x6380,          -- internal difficulty 1-5 (aggression table index)
  0x62B1,          -- bonus timer (Stage B; verified 2026-07-13)
  0x6207,          -- mario facing: bit7 1=right (Stage B; verified)
}

-- Training-wheels barrel-freeze: Python sends 0xF8 to disable barrels+fireball
-- for one episode (agent learns the traverse route without danger), 0xF7 to re-enable.
-- The flag persists until explicitly cleared, so Python must send one or the other
-- at the start of every episode.
local freeze_barrels = false
local FREEZE_STATUS_ADDRS = {0x6700,0x6720,0x6740,0x6760,0x6780,0x67A0,
                              0x6400,0x6420,0x6440,0x6460,0x6480}

-- Optional RAM-dump mode for address discovery: DK_RAMDUMP="0x6000:0x7000"
-- appends that [start,end) byte range to every observation.
local DUMP_LO, DUMP_HI
do
  local spec = os.getenv("DK_RAMDUMP")
  if spec then
    local a, b = spec:match("(0x%x+):(0x%x+)")
    if a then DUMP_LO, DUMP_HI = tonumber(a), tonumber(b) end
  end
end

----------------------------------------------------------------------------
local mac    = manager.machine
local logf   = io.open(LOGPATH, "w")
local function log(msg)
  if logf then logf:write(msg .. "\n"); logf:flush() end
  emu.print_info(msg)
end

local socket = emu.file("", 7)   -- READ|WRITE|CREATE -> listening server
local STATE  = "init"            -- init -> listening -> ready
local frame_n = 0
local cpu, space, screen

local function u32_be(n) return string.char((n>>24)&0xFF,(n>>16)&0xFF,(n>>8)&0xFF,n&0xFF) end
local function u16_be(n) return string.char((n>>8)&0xFF, n&0xFF) end

local function reopen_socket()
  pcall(function() socket:close() end)
  socket = emu.file("", 7)
  local BIND = os.getenv("DK_BRIDGE_BIND") or "127.0.0.1"
  socket:open("socket." .. BIND .. ":" .. PORT)
  STATE = "listening"
  log("[bridge] client lost; re-listening on " .. BIND .. ":" .. PORT)
end

local function read_exact(want)
  -- Returns nil if the client goes silent for ~6s (dead/disconnected):
  -- the old infinite spin froze the whole emulator inside frame_done
  -- (the farm-zombie bug, 2026-07-16). Caller must handle nil.
  local buf = ""
  local deadline = os.time() + 6
  while #buf < want do
    local chunk = socket:read(want - #buf)
    if chunk and #chunk > 0 then buf = buf .. chunk
    elseif os.time() > deadline then return nil end
  end
  return buf
end

local function resolve_devices()
  cpu   = mac.devices[":maincpu"]
  space = cpu.spaces["program"]
  for _, scr in pairs(mac.screens) do screen = scr; break end
end

local function send_handshake()
  local w = screen and screen.width  or 0
  local h = screen and screen.height or 0
  local lines = {}
  log(string.format("[bridge] screen %dx%d", w, h))
  for ptag, port in pairs(mac.ioport.ports) do
    for fname, _ in pairs(port.fields) do
      log(string.format("[bridge] PORT %s FIELD %s", ptag, fname))
      lines[#lines+1] = ptag .. "|" .. fname
    end
  end
  socket:write(string.format("HELLO W=%d H=%d BPP=4 FRAMESKIP=%d NFIELDS=%d\n",
                             w, h, FRAMESKIP, #PLAY_FIELDS))
  socket:write("FIELDS=" .. table.concat(lines, ";;") .. "\n")
  log("[bridge] handshake sent; " .. #lines .. " input fields")
end

local function set_field(cf, val)
  local port = mac.ioport.ports[cf.port]
  if port then
    local field = port.fields[cf.field]
    if field then field:set_value(val) end
  end
end

local function apply_action(b)
  -- Always release system inputs unless this action pulses them.
  set_field(COIN,  b == ACT_COIN  and 1 or 0)
  set_field(START, b == ACT_START and 1 or 0)
  -- Play controls: only when this is a play action (bitmask < 0x20).
  local mask = (b < 0x20) and b or 0
  for i, cf in ipairs(PLAY_FIELDS) do
    set_field(cf, (mask >> (i-1)) & 1)
  end
end

-- Read which play inputs are currently active (in extraction mode this is what
-- the .inp playback is driving). Returns a bitmask matching PLAY_FIELDS order.
local function read_play_mask()
  local mask = 0
  for i, cf in ipairs(PLAY_FIELDS) do
    local port = mac.ioport.ports[cf.port]
    if port then
      local field = port.fields[cf.field]
      if field then
        local pressed = (port:read() & field.mask) ~= (field.defvalue & field.mask)
        if pressed then mask = mask | (1 << (i - 1)) end
      end
    end
  end
  return mask
end

local function build_obs()
  local ram = {}
  for _, addr in ipairs(WATCH_ADDRS) do
    ram[#ram+1] = string.char(space:readv_u8(addr) & 0xFF)
  end
  if EXTRACT then
    -- append the current play-input bitmask as one extra ram byte
    ram[#ram+1] = string.char(read_play_mask() & 0xFF)
  end
  if DUMP_LO then
    for addr = DUMP_LO, DUMP_HI - 1 do
      ram[#ram+1] = string.char(space:readv_u8(addr) & 0xFF)
    end
  end
  local ram_blob = table.concat(ram)
  local ok, pix = pcall(function() return screen:pixels() end)
  if not ok or not pix then pix = "" end
  local payload = u16_be(#ram_blob) .. ram_blob .. pix
  return u32_be(#payload) .. payload
end

-- Open listening socket once devices exist.
local open_tries = 0
emu.register_periodic(function()
  if STATE == "init" then
    if not mac.devices[":maincpu"] then return end
    resolve_devices()
    -- DK_BRIDGE_BIND=0.0.0.0 for remote-farm mode (scripts/farm/);
    -- the default stays loopback: single machine is THE default.
    local BIND = os.getenv("DK_BRIDGE_BIND") or "127.0.0.1"
    local err = socket:open("socket." .. BIND .. ":" .. PORT)
    if err ~= nil then
      open_tries = open_tries + 1
      if open_tries <= 3 then
        log("[bridge] FAILED to open socket (try " .. open_tries .. "): " .. tostring(err))
      elseif open_tries == 4 then
        log("[bridge] giving up on port " .. PORT .. "; exiting")
        mac:exit()
      end
      return
    end
    log("[bridge] listening on " .. (os.getenv("DK_BRIDGE_BIND") or "127.0.0.1") .. ":" .. PORT)
    STATE = "listening"
  elseif STATE == "listening" then
    -- Wait (non-blocking) for the client's hello byte before doing anything.
    local hello = socket:read(1)
    if hello and #hello > 0 then
      send_handshake()
      STATE = "ready"
    end
  end
end)

-- Lock-step driver: only active once a client is confirmed (STATE == ready).
emu.register_frame_done(function()
  if STATE ~= "ready" then return end
  frame_n = frame_n + 1
  if frame_n % FRAMESKIP ~= 0 then return end
  if freeze_barrels then
    for _, addr in ipairs(FREEZE_STATUS_ADDRS) do space:write_u8(addr, 0) end
  end
  socket:write(build_obs())
  local raw = read_exact(1)
  if not raw then reopen_socket(); return end
  local action = string.byte(raw)
  if EXTRACT then
    -- playback drives the inputs; the byte from Python is just an advance ack.
    return
  end
  if action == ACT_RESET then
    local ok, err = pcall(function() mac:soft_reset() end)
    if not ok then log("[bridge] soft_reset failed: " .. tostring(err)) end
  elseif action == ACT_QUIT then
    -- Clean exit so the -record .inp file is finalized/flushed.
    log("[bridge] quit requested; exiting")
    pcall(function() mac:exit() end)
  elseif action == ACT_SAVE then
    -- Per-port slot so parallel MAME instances don't clobber one state file.
    local ok, err = pcall(function() mac:save("dk_" .. PORT) end)
    log("[bridge] save -> " .. tostring(ok) .. " " .. tostring(err))
  elseif action == ACT_LOAD then
    local ok, err = pcall(function() mac:load("dk_" .. PORT) end)
    if not ok then log("[bridge] load failed: " .. tostring(err)) end
  elseif action == 0xF8 then  -- freeze barrels+fireball (training-wheels mode)
    freeze_barrels = true
  elseif action == 0xF7 then  -- unfreeze barrels+fireball
    freeze_barrels = false
  elseif action >= 0xE0 and action <= 0xEF then
    -- Load a shared curriculum start-state curric_<idx> (made by the expert
    -- demo); 0xE0+idx. Lets training start episodes partway up the board.
    local ok, err = pcall(function() mac:load("curric_" .. (action - 0xE0)) end)
    if not ok then log("[bridge] curric load failed: " .. tostring(err)) end
  else
    apply_action(action)
  end
end)

log("[bridge] loaded; port=" .. PORT .. " frameskip=" .. FRAMESKIP)
