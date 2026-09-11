#!/usr/bin/env python3
"""Split a validated final-label export into one immutable JSON per video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("errors") or source.get("warnings"):
        raise ValueError("source final-label export contains errors or warnings")
    videos = source.get("videos")
    if not isinstance(videos, dict) or not videos:
        raise ValueError("source final-label export has no videos")

    source_hash = sha256(args.input)
    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    index = []
    try:
        for video_id, video in sorted(videos.items()):
            events = list(video.get("events", []))
            for event in events:
                event_video_id = event.get("video_id")
                if event_video_id is not None and str(event_video_id) != video_id:
                    raise ValueError(
                        f"{video_id}: event belongs to video {event_video_id}"
                    )
            events.sort(
                key=lambda row: (
                    float(row["time_sec"]),
                    str(row.get("semantic_label") or row.get("label") or ""),
                    str(row.get("source_id") or row.get("id") or ""),
                )
            )
            counts = Counter(
                str(row.get("semantic_label") or row.get("label") or "unknown")
                for row in events
            )
            payload = {
                "schema_version": "football_final_video_labels_v1",
                "video_id": video_id,
                "created_at": source.get("created_at"),
                "source_final_labels": str(args.input.resolve()),
                "source_final_labels_sha256": source_hash,
                "video_source": video.get("source", {}),
                "summary": {
                    "events": len(events),
                    "events_by_label": dict(sorted(counts.items())),
                },
                "events": events,
            }
            path = temporary / f"{video_id}.json"
            write_json(path, payload)
            index.append(
                {
                    "video_id": video_id,
                    "file": path.name,
                    "events": len(events),
                    "events_by_label": dict(sorted(counts.items())),
                    "sha256": sha256(path),
                }
            )

        manifest = {
            "schema_version": "football_final_per_video_manifest_v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_final_labels": str(args.input.resolve()),
            "source_final_labels_sha256": source_hash,
            "videos": len(index),
            "events": sum(row["events"] for row in index),
            "files": index,
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(args.output_dir)
        if args.seal:
            for path in args.output_dir.iterdir():
                os.chmod(path, 0o444)
            os.chmod(args.output_dir, 0o555)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "videos": len(index),
                "events": sum(row["events"] for row in index),
                "sealed": args.seal,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
