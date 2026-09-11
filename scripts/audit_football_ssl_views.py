#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

from dinov3.data.datasets.football_video_ssl import FootballVideoSSL


def panel(image: Image.Image | None, label: str, size: tuple[int, int] = (480, 270)) -> Image.Image:
    if image is None:
        result = Image.new("RGB", size, (30, 30, 30))
    else:
        result = image.copy().convert("RGB")
        result.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, (18, 18, 18))
        canvas.paste(result, ((size[0] - result.width) // 2, (size[1] - result.height) // 2))
        result = canvas
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, size[0], 26), fill=(0, 0, 0))
    draw.text((8, 6), label, fill=(255, 255, 255))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Render football SSL detector/motion/global views.")
    parser.add_argument("--manifest", default="outputs/football_ssl/manifests/football_all_valid_v2.jsonl")
    parser.add_argument("--data-config", default="configs/football_ssl/football_detector_local_v2.yaml")
    parser.add_argument("--output", default="outputs/football_ssl/audits/detector_local_v2")
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = FootballVideoSSL(
        root=args.manifest,
        extra=args.data_config,
        stride_sec=2.0,
        edge_sec=1.0,
        cache_size=1,
    )
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.count, len(dataset)))
    records = []
    contact_rows = []
    for position, index in enumerate(indices):
        views, _ = dataset[index]
        row_panels = [
            panel(views.anchor, f"anchor {views.video_id} @ {views.time_sec:.1f}s"),
            panel(views.neighbor, "temporal neighbor"),
            panel(
                views.detector_local,
                f"detector valid={views.detector_valid} reason={views.detector_reason}",
            ),
            panel(views.motion_local, "motion fallback"),
        ]
        row = Image.new("RGB", (960, 540), (0, 0, 0))
        for panel_index, image in enumerate(row_panels):
            row.paste(image, ((panel_index % 2) * 480, (panel_index // 2) * 270))
        file_name = f"{position:03d}_{views.video_id}_{views.time_sec:.1f}.jpg"
        row.save(output / file_name, quality=92)
        records.append(
            {
                "index": index,
                "video_id": views.video_id,
                "time_sec": views.time_sec,
                "detector_valid": views.detector_valid,
                "detector_reason": views.detector_reason,
                "motion_valid": views.motion_local is not None,
                "image": file_name,
            }
        )
        if len(contact_rows) < 12:
            contact_rows.append(row.resize((480, 270), Image.Resampling.LANCZOS))

    index_root = Path(dataset.data_config["spatial_crop"]["index_root"])
    manifest_video_ids = {Path(str(video.get("name") or video["path"])).stem for video in dataset.videos}
    indexed_video_ids = {path.stem for path in index_root.glob("*.pt")}
    covered = manifest_video_ids & indexed_video_ids
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "data_config": str(Path(args.data_config).resolve()),
        "virtual_samples": len(dataset),
        "videos": len(manifest_video_ids),
        "detector_index_videos": len(covered),
        "detector_video_coverage": len(covered) / max(len(manifest_video_ids), 1),
        "rendered_samples": len(records),
        "rendered_detector_valid": sum(record["detector_valid"] for record in records),
        "records": records,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if contact_rows:
        sheet = Image.new("RGB", (960, ((len(contact_rows) + 1) // 2) * 270), (0, 0, 0))
        for index, row in enumerate(contact_rows):
            sheet.paste(row, ((index % 2) * 480, (index // 2) * 270))
        sheet.save(output / "contact_sheet.jpg", quality=92)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()

