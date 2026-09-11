from __future__ import annotations

"""Create media-ID keyed annotation views from the reviewed canonical manifest."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "calibration"])
    args = parser.parse_args()

    manifest_path = Path(args.canonical_manifest).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "football_longform_v2.canonical_media.v1":
        raise ValueError("unsupported canonical manifest schema")
    output_root = Path(args.output_root).expanduser().resolve()
    summary: dict[str, dict] = {}
    combined_root = output_root / "all"
    combined_root.mkdir(parents=True, exist_ok=True)
    combined_names: set[str] = set()
    for split in args.splits:
        entries = payload.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"canonical manifest has no split={split}")
        split_root = output_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        expected_names = {f"{entry['media_id']}.json" for entry in entries}
        stale = [path for path in split_root.glob("*.json") if path.name not in expected_names]
        if stale:
            raise RuntimeError(f"refusing stale canonical annotation files: {stale[:5]}")
        overrides = 0
        for entry in entries:
            media_id = str(entry["media_id"])
            annotation_id = str(entry["annotation_id"])
            source = Path(entry["annotation_path"]).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = split_root / f"{media_id}.json"
            if destination.is_symlink() or destination.exists():
                if destination.resolve() != source:
                    raise RuntimeError(
                        f"canonical annotation target mismatch: {destination} -> "
                        f"{destination.resolve()}, expected {source}"
                    )
            else:
                os.symlink(source, destination)
            combined = combined_root / f"{media_id}.json"
            if combined.name in combined_names:
                raise RuntimeError(f"media ID appears in multiple canonical splits: {media_id}")
            combined_names.add(combined.name)
            if combined.is_symlink() or combined.exists():
                if combined.resolve() != source:
                    raise RuntimeError(f"combined annotation target mismatch: {combined}")
            else:
                os.symlink(source, combined)
            overrides += int(media_id != annotation_id)
        summary[split] = {
            "count": len(entries),
            "annotation_id_overrides": overrides,
            "directory": str(split_root),
        }
    provenance = {
        "schema_version": "football_longform_v2.canonical_annotation_view.v1",
        "canonical_manifest": str(manifest_path),
        "combined_directory": str(combined_root),
        "combined_count": len(combined_names),
        "splits": summary,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
