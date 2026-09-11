from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CANONICAL_SCHEMA = "football_longform_v2.canonical_media.v1"


@dataclass(frozen=True)
class CanonicalSplit:
    media_ids: tuple[str, ...]
    annotation_id_by_media_id: dict[str, str]
    source_video_by_media_id: dict[str, str]


def load_canonical_split(path: str | Path, split: str) -> CanonicalSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != CANONICAL_SCHEMA:
        raise ValueError("unsupported canonical media manifest")
    policy = payload.get("policy", {})
    if not policy.get("one_media_one_timeline"):
        raise ValueError("canonical manifest must guarantee one media per timeline")
    entries = payload.get(split)
    if not isinstance(entries, list):
        raise ValueError(f"canonical manifest has no {split} entries")
    media_ids: list[str] = []
    annotation_ids: dict[str, str] = {}
    source_videos: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"invalid canonical {split} entry")
        media_id = str(entry.get("media_id", ""))
        annotation_id = str(entry.get("annotation_id", ""))
        if not media_id or not annotation_id or media_id in annotation_ids:
            raise ValueError(f"invalid or duplicate canonical media id in {split}: {media_id}")
        source_video = str(entry.get("source_video", ""))
        if not source_video:
            raise ValueError(f"canonical {split} entry has no source_video: {media_id}")
        media_ids.append(media_id)
        annotation_ids[media_id] = annotation_id
        source_videos[media_id] = source_video
    return CanonicalSplit(tuple(media_ids), annotation_ids, source_videos)
