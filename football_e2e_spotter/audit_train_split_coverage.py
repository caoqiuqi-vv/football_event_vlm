#!/usr/bin/env python3
"""Read-only audit of canonical long-video split coverage in the clip loader."""

from __future__ import annotations

import argparse
import json

from train_football_events import load_config, load_long_video_records, read_video_id_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--split-file", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config, [
        f"data.long_video.split_files.{args.split}=[{args.split_file}]",
        f"data.long_video.roots[0].annotations_dir={args.annotations}",
    ])
    records, events = load_long_video_records(cfg, args.split)
    expected = set(read_video_id_files([args.split_file]))
    present = {record.video_id for record in records}
    by_video = {video_id: values for (_, video_id), values in events.items()}
    missing = sorted(expected - present)
    detail = {
        video_id: {
            "events": len(by_video.get(video_id, [])),
            "usable_events": sum(not event.is_ignored for event in by_video.get(video_id, [])),
        }
        for video_id in missing
    }
    print(json.dumps({
        "canonical_videos": len(expected), "record_videos": len(present),
        "missing_from_records": missing, "detail": detail,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
