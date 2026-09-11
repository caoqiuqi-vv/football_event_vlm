#!/usr/bin/env python3
"""Create four secure reviewer accounts and assign whole videos by duration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-credentials", type=Path, required=True)
    parser.add_argument("--base-url", default="https://review.example.com")
    parser.add_argument("--reviewers", default="质检员1,质检员2,质检员3,质检员4")
    parser.add_argument("--seed", default="football-full-review-v30")
    return parser.parse_args()


def password_digest(password: str, salt: bytes) -> str:
    return hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32
    ).hex()


def stable_tie(seed: str, video_id: str) -> str:
    return hashlib.sha256(f"{seed}|{video_id}".encode()).hexdigest()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    reviewers = [item.strip() for item in args.reviewers.split(",") if item.strip()]
    if len(reviewers) != 4:
        raise RuntimeError("exactly four reviewer display names are required")
    videos = []
    for video in manifest.get("videos", []):
        video_id = str(video["video_id"])
        duration = float(video.get("duration_sec") or 0.0)
        if duration <= 0:
            raise RuntimeError(f"invalid duration for {video_id}: {duration}")
        videos.append((video_id, duration))
    if len(videos) != len({video_id for video_id, _ in videos}):
        raise RuntimeError("manifest contains duplicate videos")

    # Longest-processing-time scheduling balances total long-video context while
    # keeping every video indivisible and owned by exactly one reviewer.
    buckets = [
        {"display_name": display_name, "seconds": 0.0, "video_ids": []}
        for display_name in reviewers
    ]
    for video_id, duration in sorted(
        videos, key=lambda item: (-item[1], stable_tie(args.seed, item[0]))
    ):
        bucket = min(
            buckets,
            key=lambda item: (float(item["seconds"]), len(item["video_ids"])),
        )
        bucket["video_ids"].append(video_id)
        bucket["seconds"] += duration

    config_users = []
    credential_users = []
    base_url = args.base_url.rstrip("/")
    for index, bucket in enumerate(buckets, start=1):
        user_id = f"reviewer_{index:02d}"
        slug = f"r{index}-{secrets.token_urlsafe(12)}"
        password = secrets.token_urlsafe(15)
        salt = secrets.token_bytes(16)
        config_users.append({
            "user_id": user_id,
            "slug": slug,
            "display_name": bucket["display_name"],
            "password_salt": salt.hex(),
            "password_hash": password_digest(password, salt),
            "video_ids": bucket["video_ids"],
        })
        credential_users.append({
            "user_id": user_id,
            "display_name": bucket["display_name"],
            "url": f"{base_url}/u/{slug}",
            "password": password,
            "video_count": len(bucket["video_ids"]),
            "video_hours": round(float(bucket["seconds"]) / 3600.0, 3),
            "video_ids": bucket["video_ids"],
        })

    created_at = datetime.now(timezone.utc).isoformat()
    config = {
        "schema_version": 1,
        "created_at": created_at,
        "assignment_unit": "whole_video",
        "manifest": str(args.manifest.resolve()),
        "users": config_users,
    }
    credentials = {
        "schema_version": 1,
        "created_at": created_at,
        "warning": "PLAINTEXT SECRETS: deliver each row separately, then remove or archive securely",
        "users": credential_users,
    }
    for path, payload in (
        (args.output_config, config),
        (args.output_credentials, credentials),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    print(json.dumps({
        "videos": len(videos),
        "assignment_unit": "whole_video",
        "reviewers": [
            {
                "display_name": item["display_name"],
                "video_count": item["video_count"],
                "video_hours": item["video_hours"],
            }
            for item in credential_users
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
