#!/usr/bin/env python3
"""Add back-pass as a first-class human correction label.

The model still predicts shot/save/set_piece only. back_pass is a human
confounder label used to repair high-confidence false-positive candidates and
is persisted/exported exactly like the existing primary labels.
"""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v24  # noqa: F401


# Stores in v15+ consult this module-level value at request time, and bootstrap
# reads it when the server starts. Keeping the model labels unchanged avoids
# implying that the current checkpoint has a fourth output head.
hierarchical.PRIMARY_LABELS = ("shot", "save", "set_piece", "back_pass")

_html_v24 = hierarchical.patch_hierarchical_html
_js_v24 = hierarchical.patch_hierarchical_js


def patch_html_v25(text: str) -> str:
    return _html_v24(text).replace(
        "hierarchical-v24-bottom-confirm-20260818",
        "hierarchical-v25-back-pass-label-20260901",
        1,
    )


def patch_js_v25(text: str) -> str:
    text = _js_v24(text)
    label_anchor = 'own_goal: "乌龙球" };'
    if label_anchor not in text:
        raise RuntimeError("v25 label-name anchor missing")
    text = text.replace(label_anchor, 'own_goal: "乌龙球", back_pass: "回传" };', 1)

    # The inherited number-key handler predates the segment-level multi-label
    # editor. Make 1..4 operate on the actual selected label set.
    key_anchor = 'else if (["1", "2", "3", "4", "5", "6"].includes(event.key)) { state.selectedLabel = state.bootstrap.review_labels[Number(event.key) - 1]; renderClassButtons(); }'
    key_replacement = '''else if (["1", "2", "3", "4"].includes(event.key)) {
      const labels = state.bootstrap.primary_labels || ["shot", "save", "set_piece", "back_pass"];
      const label = labels[Number(event.key) - 1];
      if (label) {
        if (state.selectedSegmentLabels.has(label)) state.selectedSegmentLabels.delete(label);
        else state.selectedSegmentLabels.add(label);
        state.selectedLabel = label;
        renderSegmentLabelEditor(); renderClassButtons(); renderTeamAttribution();
      }
    }'''
    if key_anchor not in text:
        raise RuntimeError("v25 number-key anchor missing")
    return text.replace(key_anchor, key_replacement, 1)


hierarchical.patch_hierarchical_html = patch_html_v25
hierarchical.patch_hierarchical_js = patch_js_v25
hierarchical.HIERARCHICAL_CSS += r'''
:root { --back-pass:#e6b86a; }
.primary-label-buttons button.label-back_pass.selected,
.segment-label-buttons button.selected.label-back_pass {
  color:#19130a; background:var(--back-pass); border-color:var(--back-pass);
}
.event-class.back_pass,
.event-label.label-back_pass,
.queue-label.label-back_pass,
.label-back_pass { color:#19130a; background:var(--back-pass); }
.segment-label-card.label-back_pass { color:#19130a; background:var(--back-pass); }
.attribution-card.label-back_pass { border-left-color:var(--back-pass); }
.segment-label-buttons { grid-template-columns:repeat(4,minmax(0,1fr)); }
'''


if __name__ == "__main__":
    hierarchical.main()
