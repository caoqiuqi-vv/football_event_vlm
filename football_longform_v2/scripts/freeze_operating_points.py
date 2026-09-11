from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.deployment import build_frozen_operating_points


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze calibration-only event operating points.")
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    selection_path = Path(args.selection).expanduser().resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    report_path = Path(str(selected.get("report", ""))).expanduser().resolve()
    checkpoint_path = Path(
        str(selected.get("copied_checkpoint", selected.get("checkpoint", "")))
    ).expanduser().resolve()
    if not report_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("selected calibration report or checkpoint is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    operating_points = build_frozen_operating_points(
        report, checkpoint=checkpoint_path, checkpoint_sha256=sha256_file(checkpoint_path),
        report_sha256=sha256_file(report_path),
    )
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else selection_path.with_name("operating_points.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(operating_points, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output), "checkpoint": str(checkpoint_path),
        "labels": operating_points["labels"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
