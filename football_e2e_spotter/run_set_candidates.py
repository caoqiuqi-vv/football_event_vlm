#!/usr/bin/env python3
"""Export independent Stage-1 slots and their evidence for Stage-2.

This deliberately exports every low-threshold slot.  It has no NMS, no peak
selection, and no cross-window merge: adjacent events must survive as rows.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from football_e2e_spotter.data import PixelAudioSequenceDataset
from football_e2e_spotter.set_pipeline import core_starts
from football_e2e_spotter.set_spotting import NO_EVENT_INDEX, SET_LABELS, TemporalSetSpotter


def ids(path: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in Path(path).read_text().splitlines() if line.strip())


def dataset(cfg: dict, split: str, video_ids: tuple[str, ...]) -> PixelAudioSequenceDataset:
    return PixelAudioSequenceDataset(
        video_ids, store_root=cfg["data"]["frame_store"], split=split,
        annotations=cfg["data"]["annotations"], sequence_frames=1,
        sequences_per_video=1, positive_probability=0, temporal_jitter_seconds=0,
        seed=0, class_sigma_seconds={}, family_sigma_seconds={},
        ignore_radius_seconds={}, max_offset_seconds=0,
    )


def load_spotter(cfg: dict, checkpoint: str, device: torch.device) -> TemporalSetSpotter:
    data, stage = cfg["data"], cfg["stage1"]
    model = TemporalSetSpotter(sample_fps=data["sample_fps"], core_seconds=stage["core_seconds"], context_seconds=stage["context_seconds"], num_slots=stage["num_slots"], mel_bins=data["audio_mel_bins"], **stage["model"]).to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    return model.eval()


@torch.inference_mode()
def export(model: TemporalSetSpotter, data: PixelAudioSequenceDataset, video_ids: tuple[str, ...], *, device: torch.device, threshold: float) -> list[dict]:
    rows: list[dict] = []
    fps = float(model.sample_fps)
    context = int(round(model.context_seconds * fps))
    count = int(round((model.core_seconds + 2 * model.context_seconds) * fps))
    for video_id in video_ids:
        meta, total = data.metadata[video_id], int(data.metadata[video_id]["frame_count"])
        for core_start in core_starts(float(meta["duration_seconds"]), model.core_seconds):
            first = int(round(core_start * fps)) - context
            raw = list(range(first, first + count))
            valid = torch.tensor([[0 <= index < total for index in raw]], dtype=torch.bool, device=device)
            indices = [min(max(index, 0), total - 1) for index in raw]
            frames = data.normalize_eval_frames(data.read_frame_indices(video_id, indices)).unsqueeze(0).to(device)
            audio = torch.from_numpy(data.audio[video_id][indices].astype("float32", copy=True)).unsqueeze(0).to(device)
            out = model(frames, audio, valid)
            probs, quality = out["class_logits"][0].softmax(-1), out["quality_logits"][0].sigmoid()
            select = torch.linspace(0, out["core_memory"].shape[1] - 1, 64, device=device).round().long()
            shared = out["core_memory"][0, select].float().cpu().tolist()
            core_audio = audio[0, model.core_start_index:model.core_start_index + model.core_frame_count]
            audio_tokens = core_audio[torch.linspace(0, core_audio.shape[0] - 1, 64, device=device).round().long()].float().cpu().tolist()
            for slot in range(model.num_slots):
                label_index = int(probs[slot, :NO_EVENT_INDEX].argmax())
                score = float(probs[slot, label_index] * quality[slot])
                if score < threshold:
                    continue
                rows.append({
                    "video_id": video_id, "source_video": meta["source_video"], "core_start_seconds": core_start,
                    "slot_index": slot, "label": SET_LABELS[label_index],
                    "time_sec": float(core_start + model.core_seconds * out["time_normalized"][0, slot]),
                    "score": score, "eventness": float(quality[slot]),
                    "uncertainty_sec": float(out["log_sigma"][0, slot].exp()),
                    "stage1_logits": out["class_logits"][0, slot].float().cpu().tolist(),
                    "slot_embedding": out["slot_embeddings"][0, slot].float().cpu().tolist(),
                    "shared_tokens": shared, "audio_tokens": audio_tokens,
                    "no_temporal_nms": True,
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ids", required=True); parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True); parser.add_argument("--threshold", type=float, default=.01)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    video_ids = ids(args.ids); data = dataset(cfg, args.split, video_ids)
    rows = export(load_spotter(cfg, args.checkpoint, torch.device(args.device)), data, video_ids, device=torch.device(args.device), threshold=args.threshold)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output.with_suffix(output.suffix + ".meta.json")).write_text(json.dumps({"no_temporal_nms": True, "threshold": args.threshold, "videos": len(video_ids), "candidates": len(rows)}, indent=2), encoding="utf-8")
    print(json.dumps({"candidates": len(rows), "output": str(output), "no_temporal_nms": True}))


if __name__ == "__main__":
    main()
