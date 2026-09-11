#!/usr/bin/env python3
"""Ergonomic segment review UI with a compact continuous action rail."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v9  # noqa: F401


_html_v9 = hierarchical.patch_hierarchical_html
_js_v9 = hierarchical.patch_hierarchical_js


def patch_html_v10(text: str) -> str:
    text = _html_v9(text).replace(
        "hierarchical-v9-segment-multilabel-20260818",
        "hierarchical-v10-ergonomic-actions-20260818",
    )

    def take_select(select_id: str) -> str:
        start = text.index(f'          <select id="{select_id}"')
        end = text.index("          </select>", start) + len("          </select>")
        return text[start:end]

    evaluation = take_select("evaluationFilter")
    source = take_select("sourceFilter")
    conflict = '          <label class="conflict"><input type="checkbox" id="conflictOnly" /> 仅看证据冲突</label>'
    for block in (evaluation, source, conflict):
        if block not in text:
            raise RuntimeError("advanced filter anchor missing")
        text = text.replace(block + "\n", "", 1)
    advanced = f'''          <details class="advanced-filters">
            <summary>高级筛选</summary>
            <div class="advanced-filter-popover">
{evaluation}
{source}
{conflict}
            </div>
          </details>'''
    filter_end = '        </div>\n\n        <div id="emptyState"'
    if filter_end not in text:
        raise RuntimeError("filter row end anchor missing")
    text = text.replace(
        filter_end,
        advanced + '\n        </div>\n\n        <div id="emptyState"',
        1,
    )

    replay = '            <button id="replayEventBtn" class="rapid-replay"><span>↻ 重播</span><kbd>R</kbd></button>'
    pause = '            <button id="pauseEventBtn" class="rapid-pause"><span>⏸ 暂停画面</span><kbd>P</kbd></button>'
    if replay not in text:
        raise RuntimeError("replay action anchor missing")
    text = text.replace(replay, replay + "\n" + pause, 1)
    text = text.replace('<span>确认保留</span><kbd>Enter</kbd>', '<span>确认所选并下一个</span><kbd>Enter</kbd>', 1)
    text = text.replace('<span>删除误报</span><kbd>X</kbd>', '<span>删除整段并下一个</span><kbd>X</kbd>', 1)
    text = text.replace('<span>← 上一个</span>', '<span>← 上一片段</span>', 1)
    text = text.replace('<span>下一个 →</span>', '<span>下一片段 →</span>', 1)
    text = text.replace(
        '<span class="shortcut-hint">J/K 上/下一个 · Space 播放</span>',
        '<span class="shortcut-hint">J/K 上/下片段 · R 重播 · P 暂停 · Space 播放</span>',
        1,
    )
    return text


def patch_js_v10(text: str) -> str:
    text = _js_v9(text)
    replay_handler = '  $("#replayEventBtn").onclick = () => { const event = currentEvent(); if (event) playEventContext(event); };'
    pause_handler = '''
  $("#pauseEventBtn").onclick = () => {
    const player = $("#player");
    player.pause();
    state.contextEnd = null;
    toast(`已暂停在 ${formatTime(player.currentTime)}`);
  };'''
    if replay_handler not in text:
        raise RuntimeError("pause handler anchor missing")
    text = text.replace(replay_handler, replay_handler + pause_handler, 1)
    replay_key = '    else if (["r", "R"].includes(event.key)) { const current = currentEvent(); if (current) playEventContext(current); }'
    pause_key = '    else if (["p", "P"].includes(event.key)) { player.pause(); state.contextEnd = null; toast(`已暂停在 ${formatTime(player.currentTime)}`); }'
    if replay_key not in text:
        raise RuntimeError("pause keyboard anchor missing")
    text = text.replace(replay_key, replay_key + "\n" + pause_key, 1)
    return text


hierarchical.patch_hierarchical_html = patch_html_v10
hierarchical.patch_hierarchical_js = patch_js_v10
hierarchical.HIERARCHICAL_CSS += r'''
/* v10: keep frequent decisions contiguous and demote infrequent filters. */
.filter-row { grid-template-columns: minmax(130px, 1fr) minmax(130px, 1fr) auto; align-items: center; }
.advanced-filters { position: relative; justify-self: end; }
.advanced-filters > summary { padding: 5px 7px; color: #718092; border: 1px solid #2b3540; border-radius: 6px; background: #10161d; cursor: pointer; font-size: 9px; list-style: none; user-select: none; }
.advanced-filters > summary::-webkit-details-marker { display: none; }
.advanced-filters[open] > summary { color: #c4ced8; border-color: #465463; }
.advanced-filter-popover { position: absolute; top: calc(100% + 5px); right: 0; z-index: 30; width: 245px; display: grid; gap: 6px; padding: 8px; border: 1px solid #3b4857; border-radius: 8px; background: #111820f7; box-shadow: 0 12px 32px #000a; }
.advanced-filter-popover select { width: 100%; }
.advanced-filter-popover .conflict { padding: 4px 2px; color: #8e9aa7; font-size: 10px; }
.rapid-actions { grid-template-columns: .62fr .68fr .8fr 1.38fr 1.3fr .65fr; gap: 4px; margin-top: 4px; }
.rapid-actions button { min-width: 0; height: 48px; padding: 4px 5px; }
.rapid-actions button span { white-space: normal; line-height: 1.15; }
.rapid-pause { color: #f2d378; background: #292313; border-color: #665725; }
.rapid-pause:hover { color: #111; background: #f2d378; }
#saveSegmentLabelsBtn { display: none; }
.segment-label-editor { margin-bottom: 3px; }
@media (max-width: 1250px) {
  .rapid-actions { grid-template-columns: repeat(3, 1fr); }
  .rapid-actions button { height: 42px; }
}
'''


if __name__ == "__main__":
    hierarchical.main()
