from __future__ import annotations

"""Mine detector-free A0 peaks as hard negatives for the A1 verifier."""

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.annotations import load_event_annotations  # noqa: E402
from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.decoding import decode_proposals  # noqa: E402
from football_longform_v2.evaluation import infer_continuous_timeline  # noqa: E402
from football_longform_v2.feature_store import load_aligned_npz  # noqa: E402
from football_longform_v2.models import TemporalLocator  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def aggregate(shards: list[Path], output: Path) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    expected = set(range(int(payloads[0]["num_shards"])))
    actual = {int(payload["shard_index"]) for payload in payloads}
    if actual != expected:
        raise RuntimeError(f"incomplete shard set: expected={expected} actual={actual}")
    checkpoint = payloads[0]["source_checkpoint"]
    if any(payload["source_checkpoint"] != checkpoint for payload in payloads):
        raise RuntimeError("hard-negative shards use different checkpoints")
    items = [item for payload in payloads for item in payload["hard_negatives"]]
    result = {
        "schema_version": "football_longform_v2.a1_hard_negatives.v1",
        "source_checkpoint": checkpoint,
        "proposal_route": "shared_max_class_logit_nms_0.5s",
        "video_count": len({item["video_id"] for item in items}),
        "hard_negative_count": len(items),
        "hard_negatives": sorted(
            items, key=lambda row: (row["video_id"], -float(row["score"]))
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "video_count": result["video_count"],
        "hard_negative_count": result["hard_negative_count"],
        "output": str(output),
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--safety-seconds", type=float, default=8.0)
    parser.add_argument("--max-per-video", type=int, default=96)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate", nargs="*")
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if args.aggregate is not None:
        if not args.aggregate:
            raise ValueError("--aggregate requires shard paths")
        aggregate([Path(path).expanduser().resolve() for path in args.aggregate], output_path)
        return
    if not args.config or not args.checkpoint:
        raise ValueError("--config and --checkpoint are required")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")

    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    canonical = load_canonical_split(resolve(root, paths["canonical_manifest"]), "train")
    all_ids = read_video_ids(resolve(root, paths["train_ids"]))
    if canonical.media_ids != all_ids:
        raise RuntimeError("canonical train split mismatch")
    ids = all_ids[args.shard_index :: args.num_shards]
    labels = tuple(config["task"]["output_labels"])
    feature_root = resolve(root, paths["feature_store"]) / "train"
    annotation_root = resolve(root, paths["annotations"])

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = TemporalLocator.from_config(config)
    model.load_state_dict(state["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    timeline_hz = float(config["features"]["timeline_hz"])
    mined: list[dict] = []
    for video_id in ids:
        timeline = load_aligned_npz(feature_root / video_id / "timeline.npz")
        annotations = load_event_annotations(
            annotation_root / f"{canonical.annotation_id_by_media_id[video_id]}.json"
        )
        forbidden = [
            event.timestamp for event in (*annotations.accepted, *annotations.rejected)
        ]
        prediction = infer_continuous_timeline(
            model,
            timeline,
            device=device,
            core_steps=int(round(120.0 * timeline_hz)),
            context_steps=int(round(30.0 * timeline_hz)),
        )
        if prediction.class_logits is None:
            raise RuntimeError("A0 checkpoint did not emit class logits")
        probabilities = prediction.class_logits.sigmoid()
        proposals = decode_proposals(
            prediction.class_logits.max(dim=-1, keepdim=True).values.unsqueeze(0),
            timeline.timestamps.unsqueeze(0),
            ("any_event",),
            threshold=0.0,
            nms_radius_seconds={"any_event": 0.5},
            max_per_minute={"any_event": 1000.0},
        )[0]
        safe: list[dict] = []
        for proposal in proposals:
            if forbidden and min(
                abs(proposal.timestamp - timestamp) for timestamp in forbidden
            ) <= args.safety_seconds:
                continue
            class_scores = probabilities[proposal.timeline_index]
            top_index = int(class_scores.argmax())
            safe.append({
                "source": "canonical_train",
                "video_id": video_id,
                "center_sec": proposal.timestamp,
                "score": proposal.score,
                "labels": [labels[top_index]],
                "coarse_probabilities": {
                    label: float(class_scores[index])
                    for index, label in enumerate(labels)
                },
            })
        safe.sort(key=lambda row: float(row["score"]), reverse=True)
        selected = safe[: args.max_per_video]
        mined.extend(selected)
        print(
            f"device={device} shard={args.shard_index}/{args.num_shards} "
            f"video={video_id} proposals={len(proposals)} safe={len(safe)} "
            f"kept={len(selected)}",
            flush=True,
        )
    payload = {
        "schema_version": "football_longform_v2.a1_hard_negatives.shard.v1",
        "source_checkpoint": str(checkpoint),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "video_ids": list(ids),
        "hard_negatives": mined,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"output={output_path}")


if __name__ == "__main__":
    main()
