#!/usr/bin/env python3
"""Use manifest GT subtype suggestions as editable defaults for new reviews."""

from __future__ import annotations

import server_hierarchical as hierarchical
import server_hierarchical_v26  # noqa: F401


_html_v26 = hierarchical.patch_hierarchical_html
_js_v26 = hierarchical.patch_hierarchical_js


def patch_html_v27(text: str) -> str:
    return _html_v26(text).replace(
        "hierarchical-v26-boundary-navigation-20260901",
        "hierarchical-v27-gt-subtype-defaults-20260901",
        1,
    )


def patch_js_v27(text: str) -> str:
    text = _js_v26(text)
    old = (
        '  state.secondaryLabels = new Set(group.flatMap((item) => '
        'item.review.secondary_labels || []));'
    )
    new = '''  const reviewedSecondaryLabels = group.flatMap((item) => item.review.secondary_labels || []);
  const suggestedSecondaryLabels = group.every((item) => item.review.status === "unreviewed")
    ? group.flatMap((item) => item.suggested_secondary_labels || [])
    : [];
  state.secondaryLabels = new Set(reviewedSecondaryLabels.length ? reviewedSecondaryLabels : suggestedSecondaryLabels);'''
    if old not in text:
        raise RuntimeError("v27 secondary-label initialization anchor missing")
    return text.replace(old, new, 1)


hierarchical.patch_hierarchical_html = patch_html_v27
hierarchical.patch_hierarchical_js = patch_js_v27


if __name__ == "__main__":
    hierarchical.main()
