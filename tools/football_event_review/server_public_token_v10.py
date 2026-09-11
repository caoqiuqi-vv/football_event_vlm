#!/usr/bin/env python3
"""Token-protected launcher for the ergonomic v10 review UI."""

from __future__ import annotations

import server_public_token as public
import server_hierarchical_v10  # noqa: F401


if __name__ == "__main__":
    public.main()
