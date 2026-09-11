#!/usr/bin/env python3
"""Cache-busted launcher for the hierarchical review UI."""

from __future__ import annotations

import server_hierarchical as hierarchical


_patch_html = hierarchical.patch_hierarchical_html


def patch_html_v2(text: str) -> str:
    text = _patch_html(text)
    text = text.replace('/styles.css"', '/styles.css?v=hierarchical-v2-20260812"')
    text = text.replace('/app.js"', '/app.js?v=hierarchical-v2-20260812"')
    text = text.replace(
        "<strong>足球事件质检台</strong>",
        "<strong>足球事件质检台 · 分层标签 v2</strong>",
    )
    return text


hierarchical.patch_hierarchical_html = patch_html_v2


if __name__ == "__main__":
    hierarchical.main()
