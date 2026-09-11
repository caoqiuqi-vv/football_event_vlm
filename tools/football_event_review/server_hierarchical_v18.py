#!/usr/bin/env python3
"""Clean two-level prediction header and balanced review action rail."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v17  # noqa: F401


_html_v17 = hierarchical.patch_hierarchical_html
_js_v17 = hierarchical.patch_hierarchical_js


def patch_html_v18(text: str) -> str:
    text = _html_v17(text).replace(
        "hierarchical-v17-prefetch-scroll-icon-20260818",
        "hierarchical-v18-clean-header-actions-20260818",
    )
    old = '''<div><small class="prediction-caption">模型预测</small><span id="headerWhistleBadge" class="header-whistle-badge no-whistle" role="img" aria-label="未检测到哨声" title="当前片段附近未检测到哨声"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/><path class="whistle-slash" d="M4 4l16 16"/></svg></span><span id="eventClass" class="event-class">射门</span><strong id="eventTime">00:00.000</strong></div>'''
    new = '''<div class="prediction-block"><div class="prediction-kicker"><small class="prediction-caption">模型预测</small><span id="headerWhistleBadge" class="header-whistle-badge no-whistle" role="img" aria-label="未检测到哨声" title="当前片段附近未检测到哨声"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/><path class="whistle-slash" d="M4 4l16 16"/></svg></span></div><div class="prediction-result"><span id="eventClass" class="event-class">射门</span><strong id="eventTime">00:00.000</strong></div></div>'''
    if old not in text:
        raise RuntimeError("v18 event heading anchor missing")
    return text.replace(old, new, 1)


def patch_js_v18(text: str) -> str:
    text = _js_v17(text)
    old = 'const EVALUATION_NAMES = { fp: "模型 FP · 待人工核验", fn: "模型 FN · GT 漏检", matched: "已匹配 GT", unlabeled: "该类别未标注", whistle_rescue: "哨声补漏 · 待人工确认" };'
    new = 'const EVALUATION_NAMES = { fp: "FP · 待核验", fn: "FN · GT 漏检", matched: "已匹配 GT", unlabeled: "类别未标注", whistle_rescue: "哨声补漏" };'
    if old not in text:
        raise RuntimeError("v18 evaluation names anchor missing")
    return text.replace(old, new, 1)


hierarchical.patch_hierarchical_html = patch_html_v18
hierarchical.patch_hierarchical_js = patch_js_v18
hierarchical.HIERARCHICAL_CSS += r'''
/* v18: separate metadata from the actual prediction so neither competes for width. */
.event-heading {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 14px;
  min-height: 82px;
  margin: 0;
  padding: 4px 3px 11px;
}
.event-heading .prediction-block {
  min-width: 0;
  display: grid;
  grid-template-rows: 22px minmax(36px, auto);
  align-items: center;
  justify-content: start;
  gap: 6px;
}
.prediction-kicker {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 7px;
}
.prediction-caption {
  width: auto;
  color: #768494;
  font-size: 10px;
  line-height: 1;
  letter-spacing: .12em;
  white-space: nowrap;
}
.prediction-result {
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 13px;
}
.prediction-result .event-class.multi {
  flex: 0 1 auto;
  min-width: 0;
  flex-wrap: nowrap;
  white-space: nowrap;
}
.prediction-result .event-label { flex: 0 0 auto; white-space: nowrap; }
.prediction-result #eventTime {
  flex: 0 0 auto;
  color: #f1f5f9;
  font-size: 25px;
  letter-spacing: .025em;
  white-space: nowrap;
}
.event-heading .event-context {
  min-width: 105px;
  max-width: 148px;
  display: flex;
  align-items: flex-end !important;
  justify-content: center;
  gap: 6px !important;
}
.evaluation-badge {
  max-width: 100%;
  padding: 5px 9px;
  border-radius: 999px;
  font-size: 10px;
  line-height: 1.1;
  text-align: center;
  white-space: nowrap;
}
.event-position { padding-right: 4px; font-size: 11px; white-space: nowrap; }

/* One calm, balanced action strip; destructive action stays distinct but not oversized. */
.rapid-actions {
  grid-template-columns: minmax(96px, .72fr) minmax(108px, .82fr) minmax(174px, 1.25fr) minmax(174px, 1.25fr) minmax(92px, .7fr);
  gap: 6px;
  margin: 0 -4px 7px;
  padding: 8px 4px;
}
.rapid-actions button { height: 50px; padding: 5px 8px; border-radius: 9px; }
.rapid-actions button span { overflow: hidden; font-size: 13px; line-height: 1.1; text-overflow: ellipsis; white-space: nowrap; }
.rapid-actions button kbd { margin-top: 4px; font-size: 9px; }
.rapid-nav { background: #1a232d; }
.rapid-replay { background: #67c9d8; }
.rapid-decision.accept { background: #59d39c; }
.rapid-decision.delete { background: #903447; }

@media (max-width: 760px) {
  .event-heading { grid-template-columns: minmax(0, 1fr) auto; gap: 8px; }
  .prediction-result { gap: 8px; }
  .prediction-result #eventTime { font-size: 21px; }
  .rapid-actions { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .rapid-actions .rapid-decision.accept { grid-column: 1 / 2; }
  .rapid-actions button span { white-space: nowrap; }
}
'''


if __name__ == "__main__":
    hierarchical.main()
