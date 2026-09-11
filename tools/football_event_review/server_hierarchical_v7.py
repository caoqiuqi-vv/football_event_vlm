#!/usr/bin/env python3
"""Full-height, single-scroll review layout without bottom dead space."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v6  # noqa: F401


_html_v6 = hierarchical.patch_hierarchical_html


def patch_html_v7(text: str) -> str:
    return _html_v6(text).replace("hierarchical-v6-darkscroll-20260813", "hierarchical-v7-fullheight-20260813")


hierarchical.patch_hierarchical_html = patch_html_v7
hierarchical.HIERARCHICAL_CSS += """
/* v7: exactly fill the viewport. The prior 680px short-screen override left a bottom gap. */
html, body { height: 100%; min-height: 100%; }
body { display: flex; flex-direction: column; }
.topbar { flex: 0 0 70px; }
.workspace {
  flex: 1 0 620px;
  width: 100%;
  height: calc(100vh - 70px);
  min-height: 620px;
  overflow: hidden;
}
.left-column {
  height: 100%;
  grid-template-rows: minmax(320px, 1fr) 286px;
}
.timeline-panel { min-height: 0; overflow: hidden; }
.review-panel {
  height: 100%;
  min-height: 0;
  display: block;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior-y: contain;
}
.queue-toolbar { position: sticky; top: 0; z-index: 12; background: #12171ef5; }
.filter-row { position: sticky; top: 56px; z-index: 11; background: #12171ef5; }
.event-card {
  min-height: 0;
  max-height: none;
  overflow: visible;
}
.queue-list-wrap {
  min-height: 260px;
  height: auto;
  display: block;
  overflow: visible;
}
.queue-title { position: sticky; top: 166px; z-index: 10; background: #12171ef5; }
.queue-list { max-height: none; overflow: visible; }

/* Override the obsolete short-screen fixed heights from v5. */
@media (max-height: 850px) and (min-width: 1001px) {
  .workspace { height: calc(100vh - 70px); min-height: 620px; }
  .left-column { height: 100%; grid-template-rows: minmax(320px, 1fr) 286px; }
  .review-panel { height: 100%; }
}
@media (max-height: 700px) and (min-width: 1001px) {
  .workspace { height: 620px; min-height: 620px; }
  .left-column { height: 620px; grid-template-rows: 334px 286px; }
  .review-panel { height: 620px; }
}
@media (max-width: 1000px) {
  body { display: block; }
  .topbar { height: auto; min-height: 58px; }
  .workspace { display: block; height: auto; min-height: 0; overflow: visible; }
  .left-column { height: auto; grid-template-rows: 55vh 286px; }
  .review-panel { height: auto; min-height: 700px; overflow: visible; }
  .queue-toolbar, .filter-row, .queue-title { position: static; }
}
"""


if __name__ == "__main__":
    hierarchical.main()
