#!/usr/bin/env python3
"""Single-source multi-label UI with a one-line high-frequency action rail."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v10  # noqa: F401


_html_v10 = hierarchical.patch_hierarchical_html


def patch_html_v11(text: str) -> str:
    text = _html_v10(text).replace(
        "hierarchical-v10-ergonomic-actions-20260818",
        "hierarchical-v11-single-label-source-20260818",
    )
    replacements = {
        "← 上一片段": "← 上一个",
        "⏸ 暂停画面": "⏸ 暂停",
        "确认所选并下一个": "✓ 确认并下一个",
        "删除整段并下一个": "× 删除并下一个",
        "下一片段 →": "跳过 →",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"v11 action label anchor missing: {old}")
        text = text.replace(old, new, 1)
    return text


hierarchical.patch_hierarchical_html = patch_html_v11
hierarchical.HIERARCHICAL_CSS += r'''
/* v11: the segment multi-label selector is the only primary-label source. */
.label-flow-heading, .primary-label-buttons, #saveModify { display: none !important; }
.modify-block { margin-top: 6px; padding-top: 6px; }
.segment-label-buttons { grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }

/* Keep all six frequent operations on one row on normal review widths. */
.rapid-actions {
  display: grid;
  grid-template-columns: 82px 82px 82px minmax(158px, 1.18fr) minmax(150px, 1.08fr) 82px;
  gap: 4px;
  width: 100%;
}
.rapid-actions button { min-width: 0; padding: 4px 6px; }
.rapid-actions button span { white-space: nowrap; font-size: 14px; line-height: 1; }
.rapid-actions button kbd { margin-top: 5px; font-size: 10px; }
.team-buttons button span, .goal-side-buttons button { white-space: nowrap; font-size: 13px; }

@media (max-width: 700px) {
  .rapid-actions { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .rapid-actions button span { white-space: normal; }
}
'''


if __name__ == "__main__":
    hierarchical.main()
