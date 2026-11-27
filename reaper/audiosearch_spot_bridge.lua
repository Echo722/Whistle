-- AudioSearchTool lightweight Spot-to-REAPER bridge (protocol spot1).
-- Completely independent from reaper_mcp_bridge.lua.
-- Polls %LOCALAPPDATA%/AudioSearchTool/reaper-spot/inbox for double-click / Enter
-- spot commands and inserts media items non-destructively at REAPER's edit cursor.

local PROTOCOL = "spot1"
local r = reaper

local MY_ID = string.format("%d-%d", os.time(), math.floor(r.time_precise() * 1000000))
r.SetExtState("AudioSearchSpot", "instance_id", MY_ID, false)

local function norm_sep(p)
  return (p or ""):gsub("\\", "/"):gsub("/+$", "")
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
  return t
end

local function fmt(v)
  if type(v) == "number" then return string.format("%.17g", v) end
  if type(v) == "boolean" then return v and "1" or "0" end
  return tostring(v)
end

local function write_kv(path, list)
  local tmp = path .. ".tmp"
  local f, err = io.open(tmp, "wb")
  if not f then return false, "open: " .. tostring(err) end
  local ok = true
  for i = 1, #list, 2 do
    if not f:write(list[i], "=", encode(fmt(list[i + 1])), "\n") then
      ok = false
      break
    end
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

local BASE_DIR = norm_sep(os.getenv("LOCALAPPDATA") or r.GetResourcePath())
local CHANNEL = join(BASE_DIR, "AudioSearchTool/reaper-spot")
local INBOX = join(CHANNEL, "inbox")
local OUTBOX = join(CHANNEL, "outbox")
local HEARTBEAT_PATH = join(CHANNEL, "heartbeat.txt")

r.RecursiveCreateDirectory(CHANNEL, 0)
r.RecursiveCreateDirectory(INBOX, 0)
r.RecursiveCreateDirectory(OUTBOX, 0)

local function selected_track_label(proj)
  local tr = r.GetSelectedTrack(proj, 0)
  if not tr then return "(新建轨道)" end
  local _, name = r.GetSetMediaTrackInfo_String(tr, "P_NAME", "", false)
  if name and name ~= "" then return name end
  return "Track " .. tostring(math.floor(r.GetMediaTrackInfo_Value(tr, "IP_TRACKNUMBER")))
end

local function project_label(proj)
  local _, pname = r.GetProjectName(proj, "")
  if pname and pname ~= "" then return pname end
  return "(未命名工程)"
end

local function write_heartbeat(alive)
  local proj = 0
  write_kv(HEARTBEAT_PATH, {
    "protocol", alive and PROTOCOL or "closed",
    "epoch", os.time(),
    "project", project_label(proj),
    "track_name", selected_track_label(proj),
    "cursor", r.GetCursorPositionEx(proj),
  })
end

local function write_receipt(rid, ok, msg, extra)
  local fields = {
    "protocol", PROTOCOL,
    "request_id", rid,
    "ok", ok,
    "message", msg or "",
  }
  if extra then
    for i = 1, #extra do
      fields[#fields + 1] = extra[i]
    end
  end
  write_kv(join(OUTBOX, rid .. ".rcpt"), fields)
end

local function exec_spot(c)
  local rid = c.request_id or ""
  local path = c.path or ""
  if path == "" or not r.file_exists(path) then
    return write_receipt(rid, false, "源文件不存在或不可读")
  end

  local src = r.PCM_Source_CreateFromFile(path)
  if not src then
    return write_receipt(rid, false, "REAPER 无法打开该音频文件")
  end
  local src_len = r.GetMediaSourceLength(src)
  if not src_len or src_len <= 0 then
    r.PCM_Source_Destroy(src)
    return write_receipt(rid, false, "音频时长无效")
  end

  local start_sec = math.max(0.0, tonumber(c.start_sec) or 0.0)
  if start_sec >= src_len then
    start_sec = math.max(0.0, src_len - 0.005)
  end
  local raw_end = tonumber(c.end_sec)
  local end_sec = (raw_end and raw_end > start_sec) and math.min(raw_end, src_len) or src_len
  local span_sec = math.max(0.001, end_sec - start_sec)

  local playrate = 1.0
  local take_pitch = 0.0
  local b_ppitch = 1
  if c.apply_tone == "1" then
    local pitch = math.max(-24.0, math.min(24.0, tonumber(c.pitch_semitones) or 0.0))
    local rate = math.max(0.25, math.min(4.0, tonumber(c.rate) or 1.0))
    if c.preserve_pitch == "1" then
      playrate = rate
      take_pitch = pitch
      b_ppitch = 1
    else
      playrate = rate * (2.0 ^ (pitch / 12.0))
      take_pitch = 0.0
      b_ppitch = 0
    end
  end
  local item_len = span_sec / math.max(playrate, 0.01)
  local take_name = (c.take_name and c.take_name ~= "") and c.take_name or path:match("([^/\\]+)$") or "Audio"

  local proj = 0
  local pos = r.GetCursorPositionEx(proj)
  local track = r.GetSelectedTrack(proj, 0)
  local source_owned = false
  local track_name = ""

  r.PreventUIRefresh(1)
  r.Undo_BeginBlock2(proj)
  local ok, err = xpcall(function()
    if not track then
      local idx = r.CountTracks(proj)
      r.InsertTrackAtIndex(idx, true)
      track = r.GetTrack(proj, idx)
      if not track then error("无法新建轨道") end
      r.SetOnlyTrackSelected(track)
    end

    local item = r.AddMediaItemToTrack(track)
    if not item then error("无法创建 Media Item") end
    local take = r.AddTakeToMediaItem(item)
    if not take then error("无法创建 Take") end

    r.SetMediaItemTake_Source(take, src)
    source_owned = true
    r.SetMediaItemInfo_Value(item, "D_POSITION", pos)
    r.SetMediaItemInfo_Value(item, "D_LENGTH", item_len)
    r.SetMediaItemInfo_Value(item, "B_LOOPSRC", 0)
    r.SetMediaItemTakeInfo_Value(take, "D_STARTOFFS", start_sec)
    r.SetMediaItemTakeInfo_Value(take, "D_PLAYRATE", playrate)
    r.SetMediaItemTakeInfo_Value(take, "D_PITCH", take_pitch)
    r.SetMediaItemTakeInfo_Value(take, "B_PPITCH", b_ppitch)
    r.GetSetMediaItemTakeInfo_String(take, "P_NAME", take_name, true)
    r.SelectAllMediaItems(proj, false)
    r.SetMediaItemSelected(item, true)
    r.UpdateItemInProject(item)

    if c.move_cursor ~= "0" then
      r.SetEditCurPos2(proj, pos + item_len, true, false)
    end

    local _, tname = r.GetSetMediaTrackInfo_String(track, "P_NAME", "", false)
    if tname and tname ~= "" then
      track_name = tname
    else
      track_name = "Track " .. tostring(math.floor(r.GetMediaTrackInfo_Value(track, "IP_TRACKNUMBER")))
    end
  end, debug.traceback)
  r.Undo_EndBlock2(proj, "AudioSearch Spot: " .. take_name, -1)
  r.PreventUIRefresh(-1)
  r.UpdateArrange()

  if not ok then
    if not source_owned then pcall(r.PCM_Source_Destroy, src) end
    return write_receipt(rid, false, "插入 REAPER 失败: " .. tostring(err))
  end

  -- Trigger REAPER async missing-peak build so newly spotted items render waveforms immediately.
  pcall(r.Main_OnCommand, 40047, 0)

  write_receipt(rid, true, "ok", {
    "track_name", track_name,
    "position", pos,
    "length", item_len,
  })
end

local unreadable = {}
local function poll_inbox()
  local names = {}
  r.EnumerateFiles(INBOX, -1)
  local i = 0
  while true do
    local name = r.EnumerateFiles(INBOX, i)
    if not name then break end
    if name:sub(-4) == ".cmd" then names[#names + 1] = name end
    i = i + 1
  end
  table.sort(names)
  for n = 1, math.min(#names, 4) do
    local full = join(INBOX, names[n])
    local c = read_kv(full)
    if not c or c["end"] ~= "1" then
      unreadable[full] = (unreadable[full] or 0) + 1
      if unreadable[full] >= 15 then
        os.remove(full)
        unreadable[full] = nil
      end
    else
      unreadable[full] = nil
      if os.remove(full) then
        if c.protocol == PROTOCOL and c.command == "spot" and c.request_id then
          local ok, err = pcall(exec_spot, c)
          if not ok then
            write_receipt(c.request_id, false, "桥脚本异常: " .. tostring(err))
          end
        elseif c.request_id then
          write_receipt(c.request_id, false, "无效命令")
        end
      end
    end
  end
end

local last_hb = -1.0
local last_poll = -1.0

local function loop()
  if r.GetExtState("AudioSearchSpot", "instance_id") ~= MY_ID then
    return
  end
  local now = r.time_precise()
  if now - last_poll >= 0.03 then
    last_poll = now
    pcall(poll_inbox)
  end
  if now - last_hb >= 0.8 then
    last_hb = now
    pcall(write_heartbeat, true)
  end
  r.defer(loop)
end

r.atexit(function()
  if r.GetExtState("AudioSearchSpot", "instance_id") == MY_ID then
    pcall(write_heartbeat, false)
  end
end)

pcall(write_heartbeat, true)
r.defer(loop)
