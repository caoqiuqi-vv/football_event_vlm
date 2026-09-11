#!/usr/bin/env python3
"""v38 with a collision-free event header and semantic team-colour names."""

from __future__ import annotations

import ssl
from urllib.parse import urlparse

import server_hierarchical as hierarchical
import server_multiuser_v32 as v32
import server_multiuser_v36 as v36
import server_multiuser_v38 as v38


_html_v38 = hierarchical.patch_hierarchical_html
_js_v38 = hierarchical.patch_hierarchical_js


def patch_html_v39(text: str) -> str:
    text = _html_v38(text)
    old = "multiuser-v38-targeted-seek-team-drawer-20260908"
    new = "multiuser-v39-clean-header-colour-names-20260908"
    occurrences = text.count(old)
    if occurrences != 2:
        raise RuntimeError(f"v39 expected two static version anchors, found {occurrences}")
    return text.replace(old, new)


def patch_js_v39(text: str) -> str:
    text = _js_v38(text)
    old_presets = 'const TEAM_COLOUR_PRESETS = ["#e53935", "#fb8c00", "#fdd835", "#43a047", "#00acc1", "#1e88e5", "#8e24aa", "#f5f5f5", "#9e9e9e", "#212121"];'
    new_presets = r'''const TEAM_COLOUR_PRESETS = [
  { hex: "#e53935", name: "红色" }, { hex: "#fb8c00", name: "橙色" },
  { hex: "#fdd835", name: "黄色" }, { hex: "#43a047", name: "绿色" },
  { hex: "#00acc1", name: "青色" }, { hex: "#1e88e5", name: "蓝色" },
  { hex: "#8e24aa", name: "紫色" }, { hex: "#f5f5f5", name: "白色" },
  { hex: "#9e9e9e", name: "灰色" }, { hex: "#212121", name: "黑色" },
];

function colourRgb(hex) {
  const value = String(hex || "").replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(value)) return [0, 0, 0];
  return [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16));
}

function colourNameForHex(hex) {
  const rgb = colourRgb(hex);
  return TEAM_COLOUR_PRESETS.reduce((best, item) => {
    const sample = colourRgb(item.hex);
    const distance = sample.reduce((sum, channel, index) => sum + (channel - rgb[index]) ** 2, 0);
    return !best || distance < best.distance ? { name: item.name, distance } : best;
  }, null)?.name || "队伍";
}

function setTeamColour(team, hex) {
  const prefix = team === "teamA" ? "teamA" : "teamB";
  $(`#${prefix}Color`).value = hex;
  $(`#${prefix}Name`).value = colourNameForHex(hex);
  document.querySelectorAll("#colourPresets [data-preset-colour]").forEach((button) =>
    button.classList.toggle("selected", button.dataset.presetColour.toLowerCase() === String(hex).toLowerCase())
  );
}'''
    if old_presets not in text:
        raise RuntimeError("v39 colour preset anchor missing")
    text = text.replace(old_presets, new_presets, 1)

    old_render = '''  $("#colourPresets").innerHTML = `<small>当前修改 ${state.activeTeamColour === 'teamA' ? '队伍 A' : '队伍 B'}</small>` + TEAM_COLOUR_PRESETS.map((colour) => `<button data-preset-colour="${colour}" style="background:${colour}" title="${colour}"></button>`).join("");
  $("#colourPresets").querySelectorAll("button").forEach((button) => button.onclick = () => { $(`#${state.activeTeamColour === 'teamA' ? 'teamAColor' : 'teamBColor'}`).value = button.dataset.presetColour; });'''
    new_render = '''  $("#colourPresets").innerHTML = `<small>当前修改 ${state.activeTeamColour === 'teamA' ? '队伍 A' : '队伍 B'}</small>` + TEAM_COLOUR_PRESETS.map((item) => `<button data-preset-colour="${item.hex}" data-preset-name="${item.name}" style="background:${item.hex}" title="${item.name} · ${item.hex}" aria-label="选择${item.name}"></button>`).join("");
  $("#colourPresets").querySelectorAll("button").forEach((button) => button.onclick = () => setTeamColour(state.activeTeamColour, button.dataset.presetColour));'''
    if old_render not in text:
        raise RuntimeError("v39 colour render anchor missing")
    text = text.replace(old_render, new_render, 1)

    old_bind = '''  $("#teamAColor").onclick = () => { state.activeTeamColour = "teamA"; };
  $("#teamBColor").onclick = () => { state.activeTeamColour = "teamB"; };'''
    new_bind = '''  $("#teamAColor").onclick = () => { state.activeTeamColour = "teamA"; };
  $("#teamBColor").onclick = () => { state.activeTeamColour = "teamB"; };
  $("#teamAColor").oninput = (event) => setTeamColour("teamA", event.target.value);
  $("#teamBColor").oninput = (event) => setTeamColour("teamB", event.target.value);'''
    if old_bind not in text:
        raise RuntimeError("v39 colour input binding anchor missing")
    return text.replace(old_bind, new_bind, 1)


hierarchical.patch_hierarchical_html = patch_html_v39
hierarchical.patch_hierarchical_js = patch_js_v39
hierarchical.HIERARCHICAL_CSS += r'''
/* v39: an auto-height information grid replaces the legacy fixed 22px row. */
.event-heading{
  position:relative!important;display:block!important;min-height:0!important;
  margin:0 0 10px!important;padding:2px 0 1px;
}
.event-heading .prediction-block{
  display:grid!important;width:100%!important;min-width:0;
  grid-template-rows:auto auto!important;gap:12px!important;align-items:start!important;
}
.prediction-kicker{
  box-sizing:border-box;display:grid!important;width:100%;min-width:0;min-height:0;
  grid-template-columns:max-content minmax(80px,1fr) max-content;
  grid-template-areas:"source evidence evidence" "merge pending whistle";
  align-items:center!important;justify-content:stretch!important;
  column-gap:7px!important;row-gap:6px!important;padding-right:82px;
}
#taskSourceCaption{grid-area:source;min-width:0;white-space:nowrap}
.model-evidence-summary{grid-area:evidence;box-sizing:border-box;min-width:0;max-width:100%!important;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.canonical-merge-notice{grid-area:merge;min-width:0;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.prediction-pending-button{grid-area:pending;justify-self:start;height:24px;min-height:24px;padding-inline:7px}
.header-whistle-badge{grid-area:whistle;justify-self:end;margin:0!important}
.prediction-result{
  display:flex!important;width:100%!important;min-width:0;
  flex-flow:row wrap;align-items:center!important;justify-content:flex-start!important;
  gap:8px 12px!important;
}
.prediction-result .event-class.multi{display:flex;width:auto!important;max-width:100%;flex:0 1 auto;overflow:visible;padding:0;outline-offset:2px}
.prediction-result .event-label{padding:5px 9px;border-radius:6px}
.prediction-result .time-edit-button{flex:0 0 auto;margin:0;padding:3px 5px}
.event-heading .event-context{
  position:absolute;z-index:2;top:1px;right:0;display:flex!important;
  width:72px;align-items:flex-end!important;gap:5px!important;padding:0!important;
}
.event-heading .evaluation-badge{box-sizing:border-box;max-width:72px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.event-heading .event-position{font-size:10px}
.colour-presets button{position:relative;transition:transform .12s,box-shadow .12s}.colour-presets button:hover{transform:scale(1.1)}.colour-presets button.selected{box-shadow:0 0 0 2px #101820,0 0 0 4px #7dd3c7;transform:scale(1.06)}
@media(max-width:520px){
  .prediction-kicker{grid-template-columns:max-content minmax(60px,1fr) max-content;padding-right:76px;column-gap:5px!important}
  .canonical-merge-notice,.model-evidence-summary{font-size:9px}
  .prediction-result #eventTime{font-size:22px}
}
'''


class CleanReviewHandler(v38.TargetedSeekHandler):
    server_version = "FootballEventReview/3.9"

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/healthz":
            return self.send_json({"ok": True, "version": "v39-clean-header-colour-names"})
        return super().do_GET()


def main() -> None:
    args = v32.parse_args()
    store = v36.TeamCalibrationStore(args.manifest, args.db, args.access_config, args.proxy_root)
    server = v32.ReviewThreadingHTTPServer((args.host, args.port), CleanReviewHandler)
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
    print(f"Multi-user football review UI v39: {scheme}://{args.host}:{args.port}", flush=True)
    print("UI: collision-free event header + semantic team-colour names", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
