-- AudioSearchTool REAPER bridge (protocol asb1), companion of
-- src/audiosearch/integrations/reaper. Runs as a deferred ReaScript.
--
-- Safety rules:
--  * Starts only if <resource path>/audiosearch-bridge.cfg exists AND names this
--    exact resource path, so it does nothing in a production REAPER profile.
--  * Talks only through the channel directory named in that config: small
--    key=value files, written to a temp name then renamed. No network.
--  * Every command must carry the per-launch token, this instance id and the
--    current session id, and must not be expired. Checked again right before
--    executing, after the command left the queue.
--  * Bounded work per defer tick; never busy-waits; no decoding or analysis.
--  * Stops only the preview objects it created (never CF_Preview_StopAll).
--  * Source files are only opened for playback / referenced by new items.

local PROTOCOL = "asb1"
local SCRIPT_VERSION = "asb1-0.1"

local r = reaper

------------------------------------------------------------------ helpers
local function norm(p)
  return (p or ""):gsub("\\", "/"):gsub("/+$", ""):lower()
end

local function join(a, b)
  return a .. "/" .. b
end

local function encode(v)
  v = tostring(v)
  return (v:gsub("[%%\r\n]", function(c) return string.format("%%%02X", c:byte()) end))
end

local function decode(v)
  return (v:gsub("%%(%x%x)", function(h) return string.char(tonumber(h, 16)) end))
end

local function read_kv(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local text = f:read("a")
  f:close()
  local t = {}
  for line in text:gmatch("[^\n]+") do
    local k, v = line:match("^([^=]+)=(.*)$")
    if k then t[k] = decode(v) end
  end
  return t, text
end

local function fmt(v)
  if type(v) == "number" then return string.format("%.17g", v) end
  if type(v) == "boolean" then return v and "1" or "0" end
  return tostring(v)
end

-- Write key=value lines to a temp file, then move into place. Returns true only
-- when open, every write, close and the rename all succeeded (else false, reason).
local function write_kv(path, list)
  local tmp = path .. ".tmp"
  local f, err = io.open(tmp, "wb")
  if not f then return false, "open: " .. tostring(err) end
  local ok = true
  for i = 1, #list, 2 do
    if not f:write(list[i], "=", encode(fmt(list[i + 1])), "\n") then ok = false; break end
  end
  if ok and not f:write("end=1\n") then ok = false end
  local closed = f:close()
  if not ok or not closed then
    os.remove(tmp)
    return false, "write/close failed"
  end
  if r.file_exists(path) then os.remove(path) end
  local renamed, rerr = os.rename(tmp, path)
  if not renamed then
    os.remove(tmp)
    return false, "rename: " .. tostring(rerr)
  end
  return true
end

-- Size of a file in bytes via a read-only open (nil when it cannot be opened).
local function file_size(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local size = f:seek("end")
  f:close()
  return size
end

------------------------------------------------------------------ config
local resource = r.GetResourcePath()
local cfg_path = join(resource, "audiosearch-bridge.cfg")
local cfg = read_kv(cfg_path)
if not cfg then
  -- Not a configured test profile: do nothing at all (not even a log line).
  return
end

-- Startup diagnostics, only inside a configured test profile.
local LOG_PATH = join(resource, "audiosearch-bridge.log")
local function blog(msg)
  local f = io.open(LOG_PATH, "ab")
  if f then
    f:write(os.date("!%Y-%m-%dT%H:%M:%SZ"), " ", tostring(msg), "\n")
    f:close()
  end
end

local problem
if cfg.protocol ~= PROTOCOL then problem = "protocol mismatch"
elseif not cfg.token or #cfg.token < 32 then problem = "token missing/short"
elseif not cfg.channel or not cfg.instance_id then problem = "channel/instance missing"
elseif norm(cfg.resource_path) ~= norm(resource) then
  problem = "resource path mismatch: " .. norm(cfg.resource_path) .. " vs " .. norm(resource)
end
if problem then
  blog("not started: " .. problem)
  return
end
blog("starting " .. SCRIPT_VERSION .. " resource=" .. resource)

local CHANNEL = cfg.channel
local INBOX = join(CHANNEL, "inbox")
local OUTBOX = join(CHANNEL, "outbox")
local JOURNAL = join(CHANNEL, "journal")
local STATUS = join(CHANNEL, "status.txt")
local TEST_COMMANDS = cfg.test_commands == "1"
local POLL = tonumber(cfg.poll_interval) or 0.03
local HEARTBEAT = tonumber(cfg.heartbeat_interval) or 0.25
local MAX_PER_TICK = tonumber(cfg.max_per_tick) or 4
for _, d in ipairs({ CHANNEL, INBOX, OUTBOX, JOURNAL }) do r.RecursiveCreateDirectory(d, 0) end

------------------------------------------------------------------ clock
-- Epoch seconds with sub-second precision: calibrate time_precise() against
-- the moment os.time() ticks over. Until calibrated, use os.time() (coarse).
local clock = { base_epoch = nil, base_tp = nil, last_sec = os.time() }
local function epoch()
  if clock.base_epoch then return clock.base_epoch + (r.time_precise() - clock.base_tp) end
  return os.time()
end
-- Upper bound of "now" for expiry checks: before calibration os.time() may lag
-- by up to one second, so never accept a command that might already be expired.
local function epoch_upper()
  if clock.base_epoch then return epoch() end
  return os.time() + 1
end
local function calibrate()
  if clock.base_epoch then return end
  local s = os.time()
  if s ~= clock.last_sec then
    clock.base_epoch, clock.base_tp = s, r.time_precise()
  end
  clock.last_sec = s
end

------------------------------------------------------------------ session
math.randomseed(math.floor(r.time_precise() * 1000000) % 2147483647 + os.time())
local function uuid4()
  local t = {}
  for i = 1, 16 do t[i] = math.random(0, 255) end
  t[7] = (t[7] & 0x0f) | 0x40
  t[9] = (t[9] & 0x3f) | 0x80
  local h = {}
  for i = 1, 16 do h[i] = string.format("%02x", t[i]) end
  return table.concat(h, "", 1, 4) .. "-" .. table.concat(h, "", 5, 6) .. "-" ..
      table.concat(h, "", 7, 8) .. "-" .. table.concat(h, "", 9, 10) .. "-" ..
      table.concat(h, "", 11, 16)
end

local SESSION = uuid4()
local started_epoch = os.time()
local seq = 0
local has_preview = r.APIExists("CF_CreatePreview") and r.APIExists("CF_Preview_Play")
    and r.APIExists("CF_Preview_Stop") and r.APIExists("CF_Preview_GetValue")
    and r.APIExists("CF_Preview_SetValue")
local sws_version = r.APIExists("CF_GetSWSVersion") and r.CF_GetSWSVersion() or ""
local js_version = ""
if r.APIExists("JS_ReaScriptAPI_Version") then js_version = tostring(r.JS_ReaScriptAPI_Version()) end

------------------------------------------------------------------ projects
-- project_id is issued per (ReaProject pointer, project file) pair and is only
-- valid inside SESSION. Loading another file into a tab issues a new id.
local projects = { by_key = {}, by_id = {}, serial = 0 }
local discovery_version = 0
local last_discovery_sig = ""

local function scan_projects()
  local list = {}
  local current = r.EnumProjects(-1)
  local i = 0
  while true do
    local proj, fn = r.EnumProjects(i)
    if not proj then break end
    local key = tostring(proj) .. "|" .. (fn or "")
    local id = projects.by_key[key]
    if not id then
      projects.serial = projects.serial + 1
      id = SESSION:sub(1, 8) .. "-p" .. projects.serial
      projects.by_key[key] = id
    end
    projects.by_id[id] = { proj = proj, key = key }
    local tracks = {}
    for t = 0, r.CountTracks(proj) - 1 do
      local tr = r.GetTrack(proj, t)
      local _, name = r.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
      tracks[#tracks + 1] = { id = r.GetTrackGUID(tr), name = name or "" }
    end
    list[#list + 1] = {
      id = id, proj = proj, fn = fn or "", name = r.GetProjectName(proj) or "",
      current = proj == current, tracks = tracks,
      state = r.GetProjectStateChangeCount(proj),
    }
    i = i + 1
  end
  -- forget ids whose project is gone
  for id, entry in pairs(projects.by_id) do
    local alive = false
    for _, p in ipairs(list) do if p.id == id then alive = true end end
    if not alive then projects.by_id[id] = nil; projects.by_key[entry.key] = nil end
  end
  local sig = {}
  for _, p in ipairs(list) do
    sig[#sig + 1] = p.id
    for _, t in ipairs(p.tracks) do sig[#sig + 1] = t.id end
  end
  sig = table.concat(sig, ",")
  if sig ~= last_discovery_sig then
    discovery_version = discovery_version + 1
    last_discovery_sig = sig
  end
  return list
end

local function find_project(project_id)
  local entry = projects.by_id[project_id]
  if not entry or not r.ValidatePtr(entry.proj, "ReaProject*") then return nil end
  -- still the same file in that tab?
  local i = 0
  while true do
    local proj, fn = r.EnumProjects(i)
    if not proj then return nil end
    if proj == entry.proj then
      if tostring(proj) .. "|" .. (fn or "") ~= entry.key then return nil end
      return proj
    end
    i = i + 1
  end
end

local function find_track(proj, guid)
  for t = 0, r.CountTracks(proj) - 1 do
    local tr = r.GetTrack(proj, t)
    if r.GetTrackGUID(tr) == guid then return tr end
  end
  return nil
end

------------------------------------------------------------------ preview
local preview = { ptr = nil, gen = nil, rid = "", state = "idle", asset = "", rev = "",
  start = nil, stop_at = nil, position = nil, overshoot = nil, ended = "" }
local max_gen = nil

local function preview_alive()
  if not preview.ptr then return false end
  local ok, pos = r.CF_Preview_GetValue(preview.ptr, "D_POSITION")
  if ok then preview.position = pos end
  return ok
end

local function preview_halt(reason)
  if preview.ptr and preview_alive() then r.CF_Preview_Stop(preview.ptr) end
  preview.ptr = nil
  if preview.state == "playing" then preview.state = "stopped"; preview.ended = reason end
end

local function preview_tick()
  if preview.state ~= "playing" then return end
  if not preview_alive() then
    preview.ptr = nil; preview.state = "stopped"; preview.ended = "source_end"
    return
  end
  if preview.stop_at and preview.position >= preview.stop_at then
    preview.overshoot = preview.position - preview.stop_at
    preview_halt("range_end")
  end
end

------------------------------------------------------------------ status
local perf = { ticks = 0, max_tick_ms = 0, commands = 0, max_gap_ms = 0, max_cmd_ms = 0, max_cmd = "" }
local cmd_t0 = nil
local last_heartbeat = -1
local scanned = {}

local function write_status(state)
  seq = seq + 1
  scanned = scan_projects()
  local _, mode = r.GetAudioDeviceInfo("MODE")
  local _, out = r.GetAudioDeviceInfo("IDENT_OUT")
  local _, srate = r.GetAudioDeviceInfo("SRATE")
  local _, bsize = r.GetAudioDeviceInfo("BSIZE")
  local kv = {
    "protocol", PROTOCOL, "script_version", SCRIPT_VERSION, "state", state,
    "instance_id", cfg.instance_id, "session_id", SESSION, "seq", seq,
    "epoch", epoch(), "calibrated", clock.base_epoch ~= nil, "started_epoch", started_epoch,
    "reaper_version", r.GetAppVersion(), "sws_version", sws_version, "js_version", js_version,
    "cf_preview", has_preview, "test_commands", TEST_COMMANDS,
    "audio_running", r.Audio_IsRunning(), "audio_mode", mode or "", "audio_out", out or "",
    "audio_srate", srate or "", "audio_bsize", bsize or "",
    "discovery_version", discovery_version, "nprojects", #scanned,
    "pb_generation", preview.gen or "", "pb_state", preview.state, "pb_request_id", preview.rid,
    "pb_asset_id", preview.asset, "pb_revision", preview.rev,
    "pb_start", preview.start or "", "pb_end", preview.stop_at or "",
    "pb_position", preview.position or "", "pb_overshoot", preview.overshoot or "",
    "pb_ended", preview.ended,
    "perf_ticks", perf.ticks, "perf_max_tick_ms", perf.max_tick_ms, "perf_commands", perf.commands,
    "perf_max_gap_ms", perf.max_gap_ms, "perf_max_cmd_ms", perf.max_cmd_ms, "perf_max_cmd", perf.max_cmd,
  }
  for pi, p in ipairs(scanned) do
    local pre = "p" .. pi .. "_"
    kv[#kv + 1] = pre .. "id"; kv[#kv + 1] = p.id
    kv[#kv + 1] = pre .. "state"; kv[#kv + 1] = p.state
    kv[#kv + 1] = pre .. "current"; kv[#kv + 1] = p.current
    kv[#kv + 1] = pre .. "name"; kv[#kv + 1] = p.name
    kv[#kv + 1] = pre .. "ntracks"; kv[#kv + 1] = #p.tracks
    for ti, t in ipairs(p.tracks) do
      kv[#kv + 1] = pre .. "t" .. ti .. "_id"; kv[#kv + 1] = t.id
      kv[#kv + 1] = pre .. "t" .. ti .. "_name"; kv[#kv + 1] = t.name
    end
    -- Where an insert would land without the operator filling anything in: the
    -- play/edit cursor and the selected track of the live project. Empty
    -- sel_track means "no track selected" — the caller may then ask for a new one.
    if p.current then
      -- Guarded: older API names and the test doubles may not expose these, and
      -- a missing read must degrade to "unknown", never break the heartbeat.
      local ecursor
      if r.GetCursorPositionEx then
        ecursor = r.GetCursorPositionEx(p.proj)
      elseif r.GetCursorPosition then
        ecursor = r.GetCursorPosition()
      end
      kv[#kv + 1] = pre .. "ecursor"; kv[#kv + 1] = ecursor or 0
      local sel = r.GetSelectedTrack and r.GetSelectedTrack(p.proj, 0) or nil
      kv[#kv + 1] = pre .. "sel_track"; kv[#kv + 1] = sel and r.GetTrackGUID(sel) or ""
    end
  end
  write_kv(STATUS, kv)
end

------------------------------------------------------------------ receipts
local function receipt_path(rid) return join(OUTBOX, rid .. ".rcpt") end

local function reply(rid, status, code, message, extra)
  local kv = { "protocol", PROTOCOL, "request_id", rid, "session_id", SESSION, "status", status,
    "code", code or "", "message", message or "", "exec_epoch", epoch(),
    "exec_ms", cmd_t0 and (r.time_precise() - cmd_t0) * 1000 or "" }
  if extra then for i = 1, #extra do kv[#kv + 1] = extra[i] end end
  write_kv(receipt_path(rid), kv)
end

local function reject(rid, code, message) reply(rid, "rejected", code, message) end

------------------------------------------------------------------ commands
local function num(v) return tonumber(v or "") end

-- Execution-side version check, done before REAPER opens the file. The service
-- holds a no-write/no-delete handle on the verified version while the command
-- is pending; this re-check also catches a file changed while unguarded.
-- Returns nil when fine, else an error message.
local function source_version_problem(c)
  local expect = num(c.expect_size)
  if not expect then return "命令缺少文件版本（大小）信息，无法确认版本" end
  if not r.file_exists(c.path) then return nil end -- reported as offline by the caller
  local size = file_size(c.path)
  if not size then return "无法读取源文件以确认版本" end
  if size ~= expect then return "源文件大小与已验证的版本不一致" end
  return nil
end

local function cmd_preview_start(c)
  if not has_preview then return reject(c.request_id, "invalid_request", "SWS CF_Preview 不可用") end
  local gen = num(c.generation)
  if not gen then return reject(c.request_id, "invalid_request", "缺少代次") end
  if max_gen and gen <= max_gen then
    return reject(c.request_id, "request_expired", "预览代次已被更新的预览取代")
  end
  if not r.file_exists(c.path) then return reject(c.request_id, "asset_offline", "源文件不可读") end
  local problem = source_version_problem(c)
  if problem then return reject(c.request_id, "asset_changed", problem) end
  local src = r.PCM_Source_CreateFromFile(c.path)
  if not src then return reject(c.request_id, "asset_offline", "源文件无法打开") end
  local len = r.GetMediaSourceLength(src)
  local start, stop_at = num(c.start) or 0, num(c.stop_at)
  local expect_len = num(c.expect_duration)
  if len <= 0 or (expect_len and math.abs(len - expect_len) > 0.01) then
    r.PCM_Source_Destroy(src)
    return reject(c.request_id, "asset_changed", "源文件时长与目录记录不一致")
  end
  if start >= len or (stop_at and stop_at > len + 0.001) then
    r.PCM_Source_Destroy(src)
    return reject(c.request_id, "invalid_request", "区间超出源文件时长")
  end
  preview_halt("replaced")
  local p = r.CF_CreatePreview(src)
  r.PCM_Source_Destroy(src) -- the preview keeps its own duplicate
  if not p then return reject(c.request_id, "internal_error", "无法创建预览") end
  r.CF_Preview_SetValue(p, "D_POSITION", start)
  if not r.CF_Preview_Play(p) then
    return reject(c.request_id, "internal_error", "预览未能开始（音频设备未运行？）")
  end
  max_gen = gen
  preview = { ptr = p, gen = gen, rid = c.request_id, state = "playing", asset = c.asset_id or "",
    rev = c.revision or "", start = start, stop_at = stop_at, position = start, overshoot = nil,
    ended = "" }
  reply(c.request_id, "completed", "", "", { "generation", gen, "position", start,
    "audio_running", r.Audio_IsRunning() })
end

local function check_generation(c)
  local gen = num(c.generation)
  if not gen or not max_gen or gen > max_gen then
    reject(c.request_id, "invalid_request", "未知的预览代次"); return false
  end
  if gen < max_gen then
    reject(c.request_id, "request_expired", "该预览代次已被取代，未影响当前预览"); return false
  end
  return true
end

local function cmd_preview_stop(c)
  if not check_generation(c) then return end
  local was = preview.state
  preview_halt("stopped")
  preview.state = "stopped"
  reply(c.request_id, "completed", "", "", { "was", was, "position", preview.position or "" })
end

local function cmd_preview_seek(c)
  if not check_generation(c) then return end
  local pos = num(c.position)
  if preview.state ~= "playing" or not preview_alive() then
    return reject(c.request_id, "request_expired", "该代次预览已停止")
  end
  if not pos or pos < (preview.start or 0) or (preview.stop_at and pos >= preview.stop_at) then
    return reject(c.request_id, "invalid_request", "定位位置不在当前预览区间内")
  end
  r.CF_Preview_SetValue(preview.ptr, "D_POSITION", pos)
  preview.position = pos
  reply(c.request_id, "completed", "", "", { "position", pos })
end

local function item_info(item)
  local take = r.GetActiveTake(item)
  local src = take and r.GetMediaItemTake_Source(take)
  local _, guid = r.GetSetMediaItemInfo_String(item, "GUID", "", false)
  local _, tag = r.GetSetMediaItemInfo_String(item, "P_EXT:audiosearch_request", "", false)
  return {
    "item_id", guid,
    "item_position", r.GetMediaItemInfo_Value(item, "D_POSITION"),
    "item_length", r.GetMediaItemInfo_Value(item, "D_LENGTH"),
    "take_offset", take and r.GetMediaItemTakeInfo_Value(take, "D_STARTOFFS") or "",
    "take_playrate", take and r.GetMediaItemTakeInfo_Value(take, "D_PLAYRATE") or "",
    "source_file", src and r.GetMediaSourceFileName(src) or "",
    "source_length", src and r.GetMediaSourceLength(src) or "",
    "source_channels", src and r.GetMediaSourceNumChannels(src) or "",
    "track_id", r.GetTrackGUID(r.GetMediaItem_Track(item)),
    "item_tag", tag or "",
  }
end

local function cmd_insert(c)
  local rid = c.request_id
  local proj = find_project(c.project_id)
  if not proj then return reject(rid, "target_changed", "目标工程已不存在或已切换内容") end
  -- An empty track_id is the operator's "nothing selected" case: the item gets a
  -- new track of its own, inside the same undo block as the insert. Creating a
  -- track is only possible in the current project, so any other target stays a
  -- refusal rather than a surprise in another tab.
  local make_track = c.track_id == nil or c.track_id == ""
  local track
  if make_track then
    if proj ~= r.EnumProjects(-1) then
      return reject(rid, "target_changed", "只能在当前工程里新建轨道")
    end
  else
    track = find_track(proj, c.track_id)
    if not track then return reject(rid, "target_changed", "目标轨道已不在该工程中") end
  end
  local pos, offset, length = num(c.position), num(c.offset), num(c.length)
  if not pos or not offset or pos < 0 or offset < 0 or (length and length <= 0) then
    return reject(rid, "invalid_request", "插入参数不完整")
  end
  if not r.file_exists(c.path) then return reject(rid, "asset_offline", "源文件不可读") end
  local problem = source_version_problem(c)
  if problem then return reject(rid, "asset_changed", problem) end
  local src = r.PCM_Source_CreateFromFile(c.path)
  if not src then return reject(rid, "asset_offline", "源文件无法打开") end
  local len = r.GetMediaSourceLength(src)
  local expect_len, expect_ch = num(c.expect_duration), num(c.expect_channels)
  if len <= 0 or (expect_len and math.abs(len - expect_len) > 0.01)
      or (expect_ch and r.GetMediaSourceNumChannels(src) ~= expect_ch) then
    r.PCM_Source_Destroy(src)
    return reject(rid, "asset_changed", "源文件时长/声道与目录记录不一致")
  end
  if not length then length = len - offset end -- whole file / open end
  if length <= 0 or offset + length > len + 0.001 then
    r.PCM_Source_Destroy(src)
    return reject(rid, "invalid_request", "区间超出源文件时长")
  end
  -- Mark "started" first: if REAPER dies before the receipt is written, a
  -- later look-up answers "uncertain" instead of pretending nothing happened.
  -- Without a confirmed marker the project is not touched at all.
  local marked, why = write_kv(join(JOURNAL, rid .. ".begin"), { "request_id", rid, "epoch", epoch() })
  if not marked or not r.file_exists(join(JOURNAL, rid .. ".begin")) then
    r.PCM_Source_Destroy(src)
    return reject(rid, "internal_error", "无法写入执行保护记录，未修改工程：" .. tostring(why))
  end
  local desc = "AudioSearch 插入 " .. rid
  local source_owned = false
  local item
  r.PreventUIRefresh(1)
  local block_ok = pcall(r.Undo_BeginBlock2, proj)
  local ok, err = xpcall(function()
    if make_track then
      local index = r.CountTracks(proj)
      r.InsertTrackAtIndex(index, true)
      track = r.GetTrack(proj, index)
      if not track then error("无法在当前工程新建轨道") end
    end
    item = r.AddMediaItemToTrack(track)
    local take = r.AddTakeToMediaItem(item)
    r.SetMediaItemTake_Source(take, src)
    source_owned = true -- the take owns src from here on
    r.SetMediaItemInfo_Value(item, "D_POSITION", pos)
    r.SetMediaItemInfo_Value(item, "D_LENGTH", length)
    r.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS", offset)
    r.GetSetMediaItemTakeInfo_String(take, "P_NAME", c.take_name or "", true)
    r.GetSetMediaItemInfo_String(item, "P_EXT:audiosearch_request", rid, true)
    r.GetSetMediaItemInfo_String(item, "P_EXT:audiosearch_asset",
      (c.asset_id or "") .. "@" .. (c.revision or ""), true)
    r.UpdateItemInProject(item)
  end, debug.traceback)
  -- Always close what was opened, whatever happened in between.
  if block_ok then pcall(r.Undo_EndBlock2, proj, desc, -1) end
  pcall(r.PreventUIRefresh, -1)
  pcall(r.UpdateArrange)
  if not ok then
    if not source_owned then pcall(r.PCM_Source_Destroy, src) end
    -- Parts of the item may exist: never claim rejected/completed, never retry.
    return reply(rid, "uncertain", "execution_uncertain",
      "插入中途出错，工程可能已部分修改（可用撤销“" .. desc .. "”恢复）：" .. tostring(err))
  end
  local extra = item_info(item)
  extra[#extra + 1] = "undo_desc"; extra[#extra + 1] = desc
  reply(rid, "completed", "", "", extra)
end

------------------------------------------------------------------ test-only commands
local function find_item(proj, guid)
  for i = 0, r.CountMediaItems(proj) - 1 do
    local item = r.GetMediaItem(proj, i)
    local _, g = r.GetSetMediaItemInfo_String(item, "GUID", "", false)
    if g == guid then return item end
  end
  return nil
end

local function cmd_test(c)
  local rid = c.request_id
  local proj = find_project(c.project_id or "")
  if c.command == "test.inspect" then
    if not proj then return reject(rid, "target_changed", "工程不存在") end
    local item = c.item_id and c.item_id ~= "" and find_item(proj, c.item_id)
    local extra = { "project_items", r.CountMediaItems(proj), "undo_top", r.Undo_CanUndo2(proj) or "",
      "dirty", r.IsProjectDirty(proj), "state", r.GetProjectStateChangeCount(proj) }
    if c.track_id and c.track_id ~= "" then
      local tr = find_track(proj, c.track_id)
      extra[#extra + 1] = "track_items"; extra[#extra + 1] = tr and r.CountTrackMediaItems(tr) or -1
    end
    extra[#extra + 1] = "item_found"; extra[#extra + 1] = item and true or false
    if item then for _, v in ipairs(item_info(item)) do extra[#extra + 1] = v end end
    return reply(rid, "completed", "", "", extra)
  elseif c.command == "test.undo" then
    if not proj then return reject(rid, "target_changed", "工程不存在") end
    local top = r.Undo_CanUndo2(proj) or ""
    if top ~= c.expect_desc then
      return reject(rid, "target_changed", "撤销栈顶不是预期的插入：" .. top)
    end
    local ok = r.Undo_DoUndo2(proj)
    return reply(rid, ok ~= 0 and "completed" or "rejected", ok ~= 0 and "" or "internal_error",
      ok ~= 0 and "" or "撤销失败", { "undone", top })
  elseif c.command == "test.add_track" then
    if not proj then return reject(rid, "target_changed", "工程不存在") end
    local n = r.CountTracks(proj)
    r.InsertTrackInProject(proj, n, 0)
    local tr = r.GetTrack(proj, n)
    r.GetSetMediaTrackInfo_String(tr, "P_NAME", c.name or "", true)
    return reply(rid, "completed", "", "", { "track_id", r.GetTrackGUID(tr) })
  elseif c.command == "test.delete_track" then
    if not proj then return reject(rid, "target_changed", "工程不存在") end
    local tr = find_track(proj, c.track_id or "")
    if not tr then return reject(rid, "target_changed", "轨道不存在") end
    r.DeleteTrack(tr)
    return reply(rid, "completed", "", "", {})
  elseif c.command == "test.open_project" then
    local path = c.path or ""
    local root = norm(cfg.test_root or "")
    if root == "" or norm(path):sub(1, #root + 1) ~= root .. "/" or not r.file_exists(path) then
      return reject(rid, "invalid_request", "只允许打开任务目录内的测试工程")
    end
    r.Main_openProject("noprompt:" .. path)
    return reply(rid, "completed", "", "", {})
  end
  return reject(rid, "invalid_request", "未知测试命令")
end

------------------------------------------------------------------ dispatch
local HANDLERS = {
  ["preview.start"] = cmd_preview_start,
  ["preview.stop"] = cmd_preview_stop,
  ["preview.seek"] = cmd_preview_seek,
  ["insert"] = cmd_insert,
}

local unreadable = {} -- path -> failed read attempts
local MAX_READ_ATTEMPTS = 20

local function handle(path)
  local c = read_kv(path)
  if not c or c["end"] ~= "1" then
    -- Usually transient on Windows (antivirus/indexer holds the new file):
    -- leave it in place and retry on the next poll. Never claim unread work.
    unreadable[path] = (unreadable[path] or 0) + 1
    if unreadable[path] < MAX_READ_ATTEMPTS then return end
    if not os.remove(path) then return end
    unreadable[path] = nil
    local rid = path:match("%-(%x+%-%x+%-%x+%-%x+%-%x+)%.cmd$")
    if rid and #rid == 36 and not r.file_exists(receipt_path(rid)) then
      reject(rid, "invalid_request", "命令文件无法读取，未执行")
    end
    return
  end
  unreadable[path] = nil
  -- Claim the command by deleting it. If the delete fails it was either
  -- withdrawn by the service (gone: must not run) or is momentarily locked
  -- (still there: retried on the next poll). Either way, do nothing now.
  local ok = os.remove(path)
  if not ok then return end
  local rid = c.request_id
  if not rid or not rid:match("^[%x%-]+$") or #rid ~= 36 then return end
  if r.file_exists(receipt_path(rid)) then return end -- already answered
  if r.file_exists(join(JOURNAL, rid .. ".begin")) then
    return reply(rid, "uncertain", "execution_uncertain", "此前已开始执行但未写回执，需核对工程现场")
  end
  if c.protocol ~= PROTOCOL then return reject(rid, "invalid_request", "协议版本不符") end
  if c.token ~= cfg.token then return reject(rid, "invalid_request", "凭据不符，未执行") end
  if c.instance_id ~= cfg.instance_id then return reject(rid, "target_changed", "实例不符") end
  if c.session_id ~= SESSION then return reject(rid, "target_changed", "会话不符（旧会话命令不回放）") end
  local expires = num(c.expires_epoch)
  if not expires or epoch_upper() >= expires then return reject(rid, "request_expired", "命令已过期，未执行") end
  local fn = HANDLERS[c.command]
  if not fn and TEST_COMMANDS and (c.command or ""):sub(1, 5) == "test." then fn = cmd_test end
  if not fn then return reject(rid, "invalid_request", "未知命令") end
  perf.commands = perf.commands + 1
  cmd_t0 = r.time_precise()
  local okc, err = pcall(fn, c)
  local cmd_ms = (r.time_precise() - cmd_t0) * 1000
  cmd_t0 = nil
  if cmd_ms > perf.max_cmd_ms then perf.max_cmd_ms, perf.max_cmd = cmd_ms, tostring(c.command) end
  if not okc then
    if c.command == "insert" then
      reply(rid, "uncertain", "execution_uncertain", "执行中出错：" .. tostring(err))
    else
      reject(rid, "internal_error", "执行出错：" .. tostring(err))
    end
  end
end

local function poll_inbox()
  local names = {}
  r.EnumerateFiles(INBOX, -1) -- invalidate the directory cache
  local i = 0
  while true do
    local name = r.EnumerateFiles(INBOX, i)
    if not name then break end
    if name:sub(-4) == ".cmd" then names[#names + 1] = name end
    i = i + 1
  end
  table.sort(names) -- names start with a zero-padded sequence number
  for n = 1, math.min(#names, MAX_PER_TICK) do handle(join(INBOX, names[n])) end
end

------------------------------------------------------------------ loop
local last_poll = -1
local running = true

local errors = 0

local last_tick = nil

local function step()
  local t0 = r.time_precise()
  if last_tick then
    local gap = (t0 - last_tick) * 1000
    if gap > perf.max_gap_ms then perf.max_gap_ms = gap end
  end
  last_tick = t0
  calibrate()
  preview_tick()
  if t0 - last_poll >= POLL then
    last_poll = t0
    poll_inbox()
  end
  if t0 - last_heartbeat >= HEARTBEAT then
    last_heartbeat = t0
    write_status("running")
  end
  perf.ticks = perf.ticks + 1
  local ms = (r.time_precise() - t0) * 1000
  if ms > perf.max_tick_ms then perf.max_tick_ms = ms end
end

local function loop()
  if not running then return end
  local ok, err = xpcall(step, debug.traceback)
  if not ok then
    errors = errors + 1
    blog("tick error: " .. tostring(err))
    if errors >= 20 then
      running = false
      blog("too many errors; bridge stopped")
      pcall(preview_halt, "bridge_error")
      pcall(write_status, "error")
      return
    end
  end
  r.defer(loop)
end

r.atexit(function()
  running = false
  preview_halt("bridge_exit")
  write_status("exited")
end)

write_status("running")
blog("running session=" .. SESSION)
r.defer(loop)
