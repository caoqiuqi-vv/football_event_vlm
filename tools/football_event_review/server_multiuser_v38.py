#!/usr/bin/env python3
"""v37 streaming with invisible targeted seeks and video-level team settings."""

from __future__ import annotations

import ssl
from pathlib import Path
from urllib.parse import urlparse

import server_hierarchical as hierarchical
import server_multiuser_v32 as v32
import server_multiuser_v36 as v36
import server_multiuser_v37 as v37


_html_v37 = hierarchical.patch_hierarchical_html
_js_v37 = hierarchical.patch_hierarchical_js


TEAM_PANEL_V37 = r'''<section id="teamSetupPanel" class="team-setup-panel">
          <button id="teamSetupToggle" class="team-setup-summary" type="button">
            <span><b id="teamSetupStatus">队伍信息待确认</b><small>上衣主色 · 左右半场</small></span>
            <span id="teamSetupBadges" class="team-setup-badges"></span><i>▾</i>
          </button>
          <div id="teamSetupEditor" class="team-setup-editor">
            <div class="team-sample-row"><button id="playTeamSample" type="button">▶ 播放队伍抽样片段</button><small id="teamSampleRange"></small></div>
            <div class="team-colour-grid">
              <label><span>队伍 A 上衣主色</span><div><input id="teamAColor" type="color"><input id="teamAName" type="text" maxlength="60" placeholder="队伍 A"></div></label>
              <label><span>队伍 B 上衣主色</span><div><input id="teamBColor" type="color"><input id="teamBName" type="text" maxlength="60" placeholder="队伍 B"></div></label>
            </div>
            <div class="colour-presets" id="colourPresets"><small>点选色块修改当前队伍颜色</small></div>
            <div class="period-side-grid">
              <label><span>上半场左侧</span><select id="firstPeriodLeftTeam"><option value="unknown">不明确</option><option value="teamA">队伍 A</option><option value="teamB">队伍 B</option></select></label>
              <label><span>半场切换时间</span><div><input id="periodSplitSec" type="number" min="0" step="0.1"><button id="useCurrentAsSplit" type="button">当前画面</button></div></label>
              <label><span>下半场左侧</span><select id="secondPeriodLeftTeam"><option value="unknown">不明确</option><option value="teamA">队伍 A</option><option value="teamB">队伍 B</option></select></label>
            </div>
            <div class="team-setup-actions"><button id="swapTeamColours" type="button">交换 A/B 颜色</button><button id="confirmTeamSetup" type="button">确认队伍与半场信息</button></div>
          </div>
        </section>'''


TEAM_UI_V38 = r'''<div class="video-level-settings">
          <div class="video-level-copy"><small>本视频配置</small><span>通常每个视频只需设置一次</span></div>
          <button id="teamSetupToggle" class="team-config-launch" type="button" aria-haspopup="dialog">
            <span class="team-config-icon" aria-hidden="true">⚙</span>
            <span><b id="teamSetupStatus">队伍信息待确认</b><small>队伍颜色与半场</small></span>
            <span id="teamSetupBadges" class="team-setup-badges"></span>
            <i aria-hidden="true">›</i>
          </button>
        </div>

        <div id="teamSetupModal" class="team-modal hidden" role="dialog" aria-modal="true" aria-labelledby="teamSetupTitle">
          <button id="teamSetupBackdrop" class="team-modal-backdrop" type="button" aria-label="关闭队伍设置"></button>
          <section id="teamSetupPanel" class="team-setup-panel team-settings-drawer">
            <header class="team-drawer-header">
              <div><small>视频级设置</small><h2 id="teamSetupTitle">队伍颜色与半场</h2><p>此设置应用于当前整段视频，不需要逐事件修改。</p></div>
              <button id="closeTeamSetup" class="team-modal-close" type="button" aria-label="关闭">×</button>
            </header>
            <div id="teamSetupEditor" class="team-setup-editor">
              <div class="team-sample-row"><button id="playTeamSample" type="button">▶ 播放队伍抽样片段</button><small id="teamSampleRange"></small></div>
              <div class="team-colour-grid">
                <label><span>队伍 A 上衣主色</span><div><input id="teamAColor" type="color"><input id="teamAName" type="text" maxlength="60" placeholder="队伍 A"></div></label>
                <label><span>队伍 B 上衣主色</span><div><input id="teamBColor" type="color"><input id="teamBName" type="text" maxlength="60" placeholder="队伍 B"></div></label>
              </div>
              <div class="colour-presets" id="colourPresets"><small>先选择 A/B 输入框，再点色块快速修改</small></div>
              <div class="period-side-grid">
                <label><span>上半场左侧</span><select id="firstPeriodLeftTeam"><option value="unknown">不明确</option><option value="teamA">队伍 A</option><option value="teamB">队伍 B</option></select></label>
                <label><span>半场切换时间</span><div><input id="periodSplitSec" type="number" min="0" step="0.1"><button id="useCurrentAsSplit" type="button">当前画面</button></div></label>
                <label><span>下半场左侧</span><select id="secondPeriodLeftTeam"><option value="unknown">不明确</option><option value="teamA">队伍 A</option><option value="teamB">队伍 B</option></select></label>
              </div>
              <div class="team-setup-actions"><button id="swapTeamColours" type="button">交换 A/B 颜色</button><button id="confirmTeamSetup" type="button">保存并关闭</button></div>
            </div>
          </section>
        </div>'''


def patch_html_v38(text: str) -> str:
    text = _html_v37(text).replace(
        "multiuser-v37-direct-range-seek-20260908",
        "multiuser-v38-targeted-seek-team-drawer-20260908",
        1,
    )
    text = text.replace(
        "hierarchical-v24-bottom-confirm-20260818",
        "multiuser-v38-targeted-seek-team-drawer-20260908",
        1,
    )
    seek_controls = r'''<div id="absoluteSeekBar" class="absolute-seek-bar">
            <input id="absoluteSeekInput" type="text" inputmode="decimal" autocomplete="off" placeholder="00:00:00 或秒数" aria-label="跳转到固定时间">
            <button id="absoluteSeekButton" type="button">跳转</button>
            <small>按需 Range 流</small>
          </div>'''
    if seek_controls not in text:
        raise RuntimeError("v38 fixed-seek controls missing")
    text = text.replace(seek_controls + "\n          ", "", 1)
    if TEAM_PANEL_V37 not in text:
        raise RuntimeError("v38 team panel anchor missing")
    return text.replace(TEAM_PANEL_V37, TEAM_UI_V38, 1)


def patch_js_v38(text: str) -> str:
    text = _js_v37(text)
    text = text.replace("teamSetupOpen: true,", "teamSetupOpen: false,", 1)

    old_seek = r'''function seekAbsolute(timeSec, autoPlay = false) {
  if (!state.video) return;
  const target = Math.max(0, Math.min(Number(state.video.duration_sec || Infinity), Number(timeSec)));
  if (!Number.isFinite(target)) { toast("请输入秒数或 HH:MM:SS"); return; }
  const player = $("#player");
  const videoId = state.currentVideoId;
  state.pendingAbsoluteSeek = { videoId, target, autoPlay };
  const apply = () => {
    const pending = state.pendingAbsoluteSeek;
    if (!pending || pending.videoId !== state.currentVideoId) return;
    state.pendingAbsoluteSeek = null;
    try { player.currentTime = pending.target; } catch (_) { return; }
    $("#absoluteSeekInput").value = formatTime(pending.target);
    if (pending.autoPlay) player.play().catch(() => toast("请点击画面开始播放"));
  };
  if (!player.getAttribute("src")) {
    // The media fragment tells the browser the first useful timestamp before
    // it opens the file, avoiding an unnecessary fetch from t=0.
    player.src = mediaUrlAt(target);
    player.load();
  }
  if (player.readyState >= 1) apply();
  else player.addEventListener("loadedmetadata", apply, { once: true });
}'''
    new_seek = r'''function targetIsBuffered(player, target) {
  for (let index = 0; index < player.buffered.length; index += 1) {
    if (target >= player.buffered.start(index) - 0.25 && target <= player.buffered.end(index) + 0.25) return true;
  }
  return false;
}

function seekAbsolute(timeSec, autoPlay = false) {
  if (!state.video) return;
  const target = Math.max(0, Math.min(Number(state.video.duration_sec || Infinity), Number(timeSec)));
  if (!Number.isFinite(target)) return;
  const player = $("#player");
  const videoId = state.currentVideoId;
  const playAtTarget = () => {
    try { player.currentTime = target; } catch (_) { return; }
    if (autoPlay) player.play().catch(() => toast("请点击画面开始播放"));
  };

  // Nearby/replayed events use the browser buffer immediately. For a distant
  // event, abort the old transfer and reopen the same authenticated resource
  // with a media-fragment target. The browser then asks the server for the
  // required byte range instead of downloading from the previous position.
  if (player.getAttribute("src") && player.readyState >= 1 && targetIsBuffered(player, target)) {
    state.pendingAbsoluteSeek = null;
    playAtTarget();
    return;
  }
  state.pendingAbsoluteSeek = { videoId, target, autoPlay };
  player.pause();
  player.removeAttribute("src");
  player.load();
  player.src = mediaUrlAt(target);
  const apply = () => {
    const pending = state.pendingAbsoluteSeek;
    if (!pending || pending.videoId !== state.currentVideoId || Math.abs(pending.target - target) > 0.001) return;
    state.pendingAbsoluteSeek = null;
    try { player.currentTime = pending.target; } catch (_) { return; }
    if (pending.autoPlay) player.play().catch(() => toast("请点击画面开始播放"));
  };
  player.addEventListener("loadedmetadata", apply, { once: true });
  player.load();
}'''
    if old_seek not in text:
        raise RuntimeError("v38 seek implementation anchor missing")
    text = text.replace(old_seek, new_seek, 1)

    old_binding = r'''  const submitAbsoluteSeek = () => {
    const target = parseAbsoluteSeek($("#absoluteSeekInput").value);
    seekAbsolute(target, false);
  };
  $("#absoluteSeekButton").onclick = submitAbsoluteSeek;
  $("#absoluteSeekInput").onkeydown = (event) => {
    if (event.key === "Enter") { event.preventDefault(); submitAbsoluteSeek(); }
  };
'''
    if old_binding not in text:
        raise RuntimeError("v38 fixed-seek binding anchor missing")
    text = text.replace(old_binding, "", 1)

    old_render = '  $("#teamSetupPanel").classList.toggle("confirmed", confirmed);\n  $("#teamSetupEditor").classList.toggle("hidden", !state.teamSetupOpen);'
    new_render = '  $("#teamSetupPanel").classList.toggle("confirmed", confirmed);\n  $("#teamSetupModal").classList.toggle("hidden", !state.teamSetupOpen);\n  document.body.classList.toggle("team-modal-open", state.teamSetupOpen);'
    if old_render not in text:
        raise RuntimeError("v38 team render anchor missing")
    text = text.replace(old_render, new_render, 1)

    old_toggle = '  $("#teamSetupToggle").onclick = () => { state.teamSetupOpen = !state.teamSetupOpen; renderTeamSetup(); };'
    new_toggle = r'''  const closeTeamSetup = () => { state.teamSetupOpen = false; renderTeamSetup(); };
  $("#teamSetupToggle").onclick = () => { state.teamSetupOpen = true; renderTeamSetup(); };
  $("#closeTeamSetup").onclick = closeTeamSetup;
  $("#teamSetupBackdrop").onclick = closeTeamSetup;'''
    if old_toggle not in text:
        raise RuntimeError("v38 team toggle anchor missing")
    text = text.replace(old_toggle, new_toggle, 1)

    old_sample = '    player.currentTime = Number(profile.sample_start_sec); state.contextEnd = Number(profile.sample_end_sec);\n    player.play().catch(() => toast("请点击画面开始播放"));'
    new_sample = '    state.teamSetupOpen = false; renderTeamSetup();\n    state.contextEnd = Number(profile.sample_end_sec);\n    seekAbsolute(Number(profile.sample_start_sec), true);'
    if old_sample not in text:
        raise RuntimeError("v38 team sample anchor missing")
    text = text.replace(old_sample, new_sample, 1)

    old_auto_open = '  state.teamSetupOpen = state.video.team_profile?.status !== "confirmed";'
    if old_auto_open not in text:
        raise RuntimeError("v38 team auto-open anchor missing")
    text = text.replace(old_auto_open, '  state.teamSetupOpen = false;', 1)

    text = text.replace(
        '  player.currentTime = Math.max(0, Number(event.support_start_sec ?? event.start_sec));\n  state.contextEnd = Math.min(state.video.duration_sec, Number(event.support_end_sec ?? event.end_sec));\n  player.play().catch(() => toast("浏览器阻止了自动播放，请点击画面或按 Space"));',
        '  state.contextEnd = Math.min(state.video.duration_sec, Number(event.support_end_sec ?? event.end_sec));\n  seekAbsolute(Math.max(0, Number(event.support_start_sec ?? event.start_sec)), true);',
        1,
    )
    text = text.replace(
        '  player.currentTime = Math.max(0, first.timeSec - 3);\n  state.contextEnd = Math.min(state.video.duration_sec, first.timeSec + 5);\n  player.play().catch(() => {});',
        '  state.contextEnd = Math.min(state.video.duration_sec, first.timeSec + 5);\n  seekAbsolute(Math.max(0, first.timeSec - 3), true);',
        1,
    )
    text = text.replace(
        '$("#playContext").onclick = () => { const event = currentEvent(); if (!event) return; const player = $("#player"); player.currentTime = Math.max(0, event.time_sec - 5); state.contextEnd = event.time_sec + 8; player.play(); };',
        '$("#playContext").onclick = () => { const event = currentEvent(); if (!event) return; state.contextEnd = event.time_sec + 8; seekAbsolute(Math.max(0, event.time_sec - 5), true); };',
        1,
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v38
hierarchical.patch_hierarchical_js = patch_js_v38
hierarchical.HIERARCHICAL_CSS += r'''
.absolute-seek-bar{display:none!important}
.video-level-settings{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:7px 0 9px;padding:7px 8px 7px 10px;border:1px solid #2d3b49;border-radius:9px;background:linear-gradient(135deg,#111a24,#121e29)}
.video-level-copy{display:grid;gap:1px;min-width:0}.video-level-copy small{color:#91a2b2;font-size:8px;font-weight:800;letter-spacing:.08em}.video-level-copy span{color:#627384;font-size:8px;white-space:nowrap}
.team-config-launch{display:flex;align-items:center;gap:7px;min-height:36px;padding:5px 8px;color:#dce6ef;border:1px solid #405267;border-radius:8px;background:#182431;cursor:pointer;text-align:left;transition:border-color .15s,background .15s,transform .15s}.team-config-launch:hover{border-color:#70869c;background:#1c2b39;transform:translateY(-1px)}.team-config-launch>span:nth-child(2){display:grid;gap:1px}.team-config-launch b{font-size:9px;white-space:nowrap}.team-config-launch small{color:#7f90a0;font-size:7px;white-space:nowrap}.team-config-launch>i{color:#8ea0b1;font-size:17px;font-style:normal}.team-config-icon{display:grid;width:24px;height:24px;place-items:center;border-radius:7px;background:#253546;color:#aebdca;font-size:12px}
.team-modal{position:fixed;inset:0;z-index:1000;display:flex;justify-content:flex-end}.team-modal.hidden{display:none}.team-modal-backdrop{position:absolute;inset:0;border:0;background:#03070bad;backdrop-filter:blur(3px);cursor:default}.team-settings-drawer{position:relative;z-index:1;box-sizing:border-box;width:min(520px,94vw);height:100%;margin:0;border:0;border-left:1px solid #425466;border-radius:0;background:linear-gradient(180deg,#14202b,#0e171f);box-shadow:-20px 0 60px #0008;overflow-y:auto;animation:teamDrawerIn .18s ease-out}.team-drawer-header{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;padding:22px 20px 15px;border-bottom:1px solid #2c3a47}.team-drawer-header small{color:#87a0b7;font-size:9px;font-weight:800;letter-spacing:.12em}.team-drawer-header h2{margin:4px 0 3px;color:#f1f6fa;font-size:20px}.team-drawer-header p{margin:0;color:#748799;font-size:10px}.team-modal-close{display:grid;width:34px;height:34px;flex:none;place-items:center;color:#b9c7d3;border:1px solid #3e5061;border-radius:9px;background:#1c2935;cursor:pointer;font-size:22px;line-height:1}.team-modal-close:hover{color:#fff;border-color:#708398;background:#263746}.team-settings-drawer .team-setup-editor{gap:14px;padding:18px 20px}.team-settings-drawer .team-colour-grid,.team-settings-drawer .period-side-grid{gap:10px}.team-settings-drawer .team-setup-actions{margin-top:5px}.team-settings-drawer .team-setup-actions button{min-height:40px}
body.team-modal-open{overflow:hidden}@keyframes teamDrawerIn{from{transform:translateX(28px);opacity:.4}to{transform:translateX(0);opacity:1}}
@media(max-width:760px){.video-level-copy span{display:none}.team-config-launch{max-width:72%}.team-config-launch .team-setup-badges b{max-width:72px}.team-settings-drawer{width:100vw}.team-drawer-header{padding:18px 16px 13px}.team-settings-drawer .team-setup-editor{padding:15px 16px}}
'''


class TargetedSeekHandler(v37.DirectRangeStreamingHandler):
    server_version = "FootballEventReview/3.8"

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/healthz":
            return self.send_json({"ok": True, "version": "v38-targeted-range-team-drawer"})
        return super().do_GET()


def main() -> None:
    args = v32.parse_args()
    store = v36.TeamCalibrationStore(args.manifest, args.db, args.access_config, args.proxy_root)
    server = v32.ReviewThreadingHTTPServer((args.host, args.port), TargetedSeekHandler)
    server.store = store  # type: ignore[attr-defined]
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ValueError("--tls-cert and --tls-key must be provided together")
    scheme = "http"
    if args.tls_cert and args.tls_key:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        server.is_tls = True  # type: ignore[attr-defined]
        scheme = "https"
    else:
        server.is_tls = False  # type: ignore[attr-defined]
    print(f"Multi-user football review UI v38: {scheme}://{args.host}:{args.port}", flush=True)
    print("Media transport: invisible buffered-or-targeted HTTP/1.1 range seek", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
