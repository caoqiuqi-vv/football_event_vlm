#!/usr/bin/env python3
"""Add throw-in as a persistent human correction/confounder label.

The production model still predicts shot/save/set_piece only. ``throw_in`` is
stored as a human-added primary label so reviewed set-piece false positives can
be exported directly as discriminative supervision without changing old rows.
"""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v28  # noqa: F401


hierarchical.PRIMARY_LABELS = (
    "shot",
    "save",
    "set_piece",
    "back_pass",
    "throw_in",
)

_html_v28 = hierarchical.patch_hierarchical_html
_js_v28 = hierarchical.patch_hierarchical_js


def patch_html_v29(text: str) -> str:
    return _html_v28(text).replace(
        "hierarchical-v28-ssh-friendly-streaming-20260901",
        "hierarchical-v29-throw-in-label-20260902",
        1,
    )


def patch_js_v29(text: str) -> str:
    text = _js_v28(text)
    label_anchor = 'back_pass: "回传" };'
    if label_anchor not in text:
        raise RuntimeError("v29 label-name anchor missing")
    text = text.replace(
        label_anchor,
        'back_pass: "回传", throw_in: "界外球" };',
        1,
    )

    key_anchor = '''else if (["1", "2", "3", "4"].includes(event.key)) {
      const labels = state.bootstrap.primary_labels || ["shot", "save", "set_piece", "back_pass"];
      const label = labels[Number(event.key) - 1];'''
    key_replacement = '''else if (["1", "2", "3", "4", "5"].includes(event.key)) {
      const labels = state.bootstrap.primary_labels || ["shot", "save", "set_piece", "back_pass", "throw_in"];
      const label = labels[Number(event.key) - 1];'''
    if key_anchor not in text:
        raise RuntimeError("v29 number-key anchor missing")
    text = text.replace(key_anchor, key_replacement, 1)

    role_anchor = 'const roles = { shot: "射门方", save: "扑救方", set_piece: "定位球执行方" };'
    if role_anchor not in text:
        raise RuntimeError("v29 attribution-role anchor missing")
    return text.replace(
        role_anchor,
        'const roles = { shot: "射门方", save: "扑救方", set_piece: "定位球执行方", back_pass: "回传方", throw_in: "掷球队" };',
        1,
    )


hierarchical.patch_hierarchical_html = patch_html_v29
hierarchical.patch_hierarchical_js = patch_js_v29
hierarchical.HIERARCHICAL_CSS += r'''
:root { --throw-in:#80cfa9; }
.primary-label-buttons button.label-throw_in.selected,
.segment-label-buttons button.selected.label-throw_in {
  color:#0d1b14; background:var(--throw-in); border-color:var(--throw-in);
}
.event-class.throw_in,
.event-label.label-throw_in,
.queue-label.label-throw_in,
.label-throw_in { color:#0d1b14; background:var(--throw-in); }
.segment-label-card.label-throw_in { color:#0d1b14; background:var(--throw-in); }
.attribution-card.label-throw_in { border-left-color:var(--throw-in); }
.segment-label-buttons { grid-template-columns:repeat(5,minmax(0,1fr)); }
'''


if __name__ == "__main__":
    hierarchical.main()
