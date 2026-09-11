#!/usr/bin/env python3
"""Token-protected launcher for the header-whistle v16 UI."""

from __future__ import annotations

import server_public_token_v15 as public_v15
import server_hierarchical_v16  # noqa: F401


if __name__ == "__main__":
    public_v15.public.main()
