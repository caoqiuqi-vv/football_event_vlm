from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from football_longform_v2.backbones import export_supervised_ema_backbone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the EMA-selected supervised football LoRA as plain DINO.")
    parser.add_argument("--task-checkpoint", required=True, type=Path)
    parser.add_argument("--official-weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--image-size", nargs=2, default=(224, 384), type=int, metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args()
    report = export_supervised_ema_backbone(
        task_checkpoint=args.task_checkpoint,
        official_weights=args.official_weights,
        output_path=args.output,
        device=args.device,
        verify_image_size=tuple(args.image_size),
    )
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
