#!/usr/bin/env python3
"""Collision-free prediction header for one to three model labels."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v21  # noqa: F401

_html_v21 = hierarchical.patch_hierarchical_html


def patch_html_v22(text: str) -> str:
    return _html_v21(text).replace(
        "hierarchical-v21-compact-complete-editing-20260818",
        "hierarchical-v22-label-capacity-header-20260818",
    )


hierarchical.patch_hierarchical_html = patch_html_v22
hierarchical.HIERARCHICAL_CSS += r'''
/* v22: labels and time own separate rows, so 1–3 labels can never overlap time. */
.event-heading { min-height: 112px; align-items: start; }
.event-heading .prediction-block {
  width: 100%;
  min-width: 0;
  grid-template-rows: 22px auto;
  justify-content: stretch;
}
.prediction-result {
  width: 100%;
  min-width: 0;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: auto auto;
  align-items: start;
  justify-items: start;
  gap: 7px;
}
.prediction-result .event-class.multi {
  width: 100%;
  min-width: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 5px;
  overflow: hidden;
  white-space: normal;
}
.prediction-result .event-label { flex: 0 0 auto; white-space: nowrap; }
.prediction-result .time-edit-button { justify-self: start; max-width: 100%; }
.prediction-result #eventTime { font-size: 24px; }
.event-heading .event-context { padding-top: 4px; }
'''

if __name__ == "__main__":
    hierarchical.main()
