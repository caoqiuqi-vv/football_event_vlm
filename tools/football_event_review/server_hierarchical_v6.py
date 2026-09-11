#!/usr/bin/env python3
"""Remove root horizontal overflow and white scrollbar/gutter artifacts."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v5  # noqa: F401


_html_v5 = hierarchical.patch_hierarchical_html


def patch_html_v6(text: str) -> str:
    return _html_v5(text).replace("hierarchical-v5-scroll-20260813", "hierarchical-v6-darkscroll-20260813")


hierarchical.patch_hierarchical_html = patch_html_v6
hierarchical.HIERARCHICAL_CSS += """
html, body {
  width: 100%;
  max-width: 100%;
  min-height: 100%;
  margin: 0;
  background: #0b0e12;
  color-scheme: dark;
  overflow-x: clip;
  scrollbar-color: #4b5664 #0b0e12;
}
body { min-height: 100vh; }
.workspace { width: 100%; max-width: 100vw; overflow-x: clip; }
* { scrollbar-color: #4b5664 #11171e; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: #0b0e12; }
*::-webkit-scrollbar-thumb { background: #4b5664; border: 2px solid #0b0e12; border-radius: 8px; }
"""


if __name__ == "__main__":
    hierarchical.main()
