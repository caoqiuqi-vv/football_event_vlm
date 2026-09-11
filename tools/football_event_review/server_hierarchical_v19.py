#!/usr/bin/env python3
"""Keep event review controls in view when candidates refresh or advance."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v18  # noqa: F401


_html_v18 = hierarchical.patch_hierarchical_html
_js_v18 = hierarchical.patch_hierarchical_js


def patch_html_v19(text: str) -> str:
    return _html_v18(text).replace(
        "hierarchical-v18-clean-header-actions-20260818",
        "hierarchical-v19-stable-review-scroll-20260818",
    )


def patch_js_v19(text: str) -> str:
    text = _js_v18(text)
    old_queue_scroll = '  requestAnimationFrame(() => list.querySelector(".current")?.scrollIntoView({ block: "nearest" }));'
    new_queue_scroll = '''  // Never use scrollIntoView here: the queue is part of the outer review panel,
  // so doing so moves the annotator away from the active event controls.
  const currentItem = list.querySelector(".current");
  if (currentItem && list.scrollHeight > list.clientHeight) {
    list.scrollTop = Math.max(0, currentItem.offsetTop - list.clientHeight / 2 + currentItem.offsetHeight / 2);
  }'''
    if old_queue_scroll not in text:
        raise RuntimeError("v19 queue scroll anchor missing")
    text = text.replace(old_queue_scroll, new_queue_scroll, 1)

    old_event_scroll = '  if (changed) requestAnimationFrame(() => { $("#eventCard").scrollTop = 0; });'
    new_event_scroll = '''  if (changed) requestAnimationFrame(() => {
    $("#eventCard").scrollTop = 0;
    const reviewPanel = document.querySelector(".review-panel");
    if (reviewPanel) reviewPanel.scrollTop = 0;
  });'''
    if old_event_scroll not in text:
        raise RuntimeError("v19 event scroll anchor missing")
    return text.replace(old_event_scroll, new_event_scroll, 1)


hierarchical.patch_hierarchical_html = patch_html_v19
hierarchical.patch_hierarchical_js = patch_js_v19


if __name__ == "__main__":
    hierarchical.main()
