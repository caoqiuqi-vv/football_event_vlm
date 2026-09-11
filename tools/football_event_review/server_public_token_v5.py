#!/usr/bin/env python3
"""Token-protected public launcher for the viewport-safe v5 UI."""

from __future__ import annotations

import server_public_token as public
import server_hierarchical_v5  # noqa: F401  # installs v5 UI patches


if __name__ == "__main__":
    public.main()
