#!/usr/bin/env python3
"""Token-protected launcher for the clean-layout v18 review UI."""

from __future__ import annotations

import server_public_token_v17 as public_v17
import server_hierarchical_v18  # noqa: F401


if __name__ == "__main__":
    public_v17.public_v15.public.main()
