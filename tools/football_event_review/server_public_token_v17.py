#!/usr/bin/env python3
"""Token-protected launcher for the responsive v17 review UI."""

from __future__ import annotations

import server_public_token_v15 as public_v15
import server_hierarchical_v17  # noqa: F401


_TokenHandlerV15 = public_v15.public.TokenProtectedHandler


class CachedMediaTokenHandler(_TokenHandlerV15):
    """Let the browser reuse previously fetched MP4 byte ranges during review."""

    def end_headers(self) -> None:
        if self.path.split("?", 1)[0].startswith("/media/"):
            self.send_header("Cache-Control", "private, max-age=86400")
        return super().end_headers()


public_v15.public.TokenProtectedHandler = CachedMediaTokenHandler


if __name__ == "__main__":
    public_v15.public.main()
