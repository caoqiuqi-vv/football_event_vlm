#!/usr/bin/env python3
"""Original-quality streaming profile for the multi-user review UI."""

from __future__ import annotations

import server as base

# The largest test video is about 49.4 Mbit/s. Keep server pacing above that
# while retaining bounded byte ranges so several reviewers can seek at once.
base.MAX_OPEN_ENDED_RANGE_BYTES = 32 * 1024 * 1024
base.MEDIA_RATE_LIMIT_BYTES_PER_SEC = 12 * 1024 * 1024

import server_multiuser_v34 as v34  # noqa: E402


_video_payload = v34.v33.v32.ProxyReviewStore.video_payload
_video_payload_for_user = v34.v33.v32.ProxyReviewStore.video_payload_for_user


def original_stream_payload(self, video_id: str) -> dict:
    payload = _video_payload(self, video_id)
    # The URL previously referred to a 540p proxy. A versioned query prevents
    # browsers from combining cached low-resolution ranges with the new file.
    payload["media_url"] = f"{payload['media_url']}?stream=original-faststart-v1"
    return payload


v34.v33.v32.ProxyReviewStore.video_payload = original_stream_payload


def original_stream_payload_for_user(self, user_id: str, video_id: str) -> dict:
    payload = _video_payload_for_user(self, user_id, video_id)
    payload["media_url"] = f"{payload['media_url']}?stream=original-faststart-v1"
    return payload


v34.v33.v32.ProxyReviewStore.video_payload_for_user = original_stream_payload_for_user


if __name__ == "__main__":
    v34.v33.v32.main()
