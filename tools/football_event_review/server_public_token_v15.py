#!/usr/bin/env python3
"""Token-protected launcher for the three-primary-label v15 UI."""

from __future__ import annotations

import json
from http import HTTPStatus
from urllib.parse import unquote, urlparse

import server_public_token as public
import server_hierarchical as hierarchical
import server_hierarchical_v15  # noqa: F401


class TokenProtectedV15Handler(public.TokenProtectedHandler):
    def do_POST(self) -> None:
        if not self.has_access():
            return self.deny()
        path = unquote(urlparse(self.path).path)
        if path.startswith("/api/events/") and path.endswith("/segment-decision"):
            try:
                event_id = path.split("/")[3]
                return self.send_json(self.store.update_segment_labels(event_id, self.read_json()))
            except (ValueError, json.JSONDecodeError) as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            except KeyError as error:
                return self.send_json({"error": f"Not found: {error}"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                return self.send_json({"error": str(error)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return super().do_POST()


public.TokenProtectedHandler = TokenProtectedV15Handler


if __name__ == "__main__":
    public.main()
