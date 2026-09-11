#!/usr/bin/env python3
"""Expose segment-level whistle evidence beside the primary model prediction."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v15  # noqa: F401


_html_v15 = hierarchical.patch_hierarchical_html
_js_v15 = hierarchical.patch_hierarchical_js


def patch_html_v16(text: str) -> str:
    text = _html_v15(text).replace(
        "hierarchical-v15-three-label-flow-20260818",
        "hierarchical-v16-header-whistle-20260818",
    )
    old = '<small class="prediction-caption">模型预测</small><span id="eventClass" class="event-class">射门</span>'
    new = (
        '<small class="prediction-caption">模型预测</small>'
        '<span id="headerWhistleBadge" class="header-whistle-badge no-whistle" '
        'title="当前片段附近的哨声检测结果">未检测到哨声</span>'
        '<span id="eventClass" class="event-class">射门</span>'
    )
    if old not in text:
        raise RuntimeError("v16 prediction heading anchor missing")
    return text.replace(old, new, 1)


def patch_js_v16(text: str) -> str:
    text = _js_v15(text)
    old = '''  const whistleScore = Number(event.whistle_score || 0);
  $("#whistleEvidence").classList.toggle("hidden", whistleScore <= 0);'''
    new = '''  const whistleScore = Math.max(0, ...group.map((item) => Number(item.whistle_score || 0)));
  const headerWhistleBadge = $("#headerWhistleBadge");
  headerWhistleBadge.className = `header-whistle-badge ${whistleScore > 0 ? "has-whistle" : "no-whistle"}`;
  headerWhistleBadge.textContent = whistleScore > 0 ? `🔔 有哨声 ${whistleScore.toFixed(2)}` : "未检测到哨声";
  headerWhistleBadge.title = whistleScore > 0
    ? `当前片段附近检测到哨声，置信度 ${whistleScore.toFixed(3)}`
    : "当前片段附近未检测到哨声";
  $("#whistleEvidence").classList.toggle("hidden", whistleScore <= 0);'''
    if old not in text:
        raise RuntimeError("v16 whistle render anchor missing")
    return text.replace(old, new, 1)


hierarchical.patch_hierarchical_html = patch_html_v16
hierarchical.patch_hierarchical_js = patch_js_v16
hierarchical.HIERARCHICAL_CSS += r'''
/* v16: whistle evidence is visible at the decision point, not only below. */
.header-whistle-badge {
  display: inline-flex;
  align-items: center;
  flex: 0 0 auto;
  min-height: 22px;
  margin: 0 7px 0 2px;
  padding: 2px 8px;
  border: 1px solid;
  border-radius: 999px;
  font-size: 10px;
  font-weight: 850;
  line-height: 1;
  white-space: nowrap;
}
.header-whistle-badge.has-whistle {
  color: #e6ff67;
  border-color: #bddb36;
  background: #7b8e1f4a;
  box-shadow: 0 0 0 2px #d7ff3812;
}
.header-whistle-badge.no-whistle {
  color: #84909e;
  border-color: #45515f;
  background: #17202a;
}
@media (max-width: 1180px) {
  .header-whistle-badge { margin-right: 4px; padding-inline: 6px; }
}
'''


if __name__ == "__main__":
    hierarchical.main()
