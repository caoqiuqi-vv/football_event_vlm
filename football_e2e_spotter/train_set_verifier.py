#!/usr/bin/env python3
"""Train the high-resolution DINOv3 verifier from OOF slot candidates.

Rows are never temporally suppressed.  Supervision is assigned with a single
one-to-one matching pass per video, so extra neighbouring slots remain true
deployment-distribution background examples rather than being discarded.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.data import IMAGENET_MEAN, IMAGENET_STD
from football_e2e_spotter.set_data import load_set_events
from football_e2e_spotter.set_spotting import NO_EVENT_INDEX, SET_LABELS, hungarian_assignment
from football_e2e_spotter.verifier import DinoVerifier, configure_dinov3_vitl16


def read_rows(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def assign_oof_targets(rows: list[dict], annotations: str, tolerance: float = 3.0) -> list[dict]:
    """Attach one GT to at most one candidate, independent of candidate class."""
    by_video: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows): by_video[row["video_id"]].append(index)
    for video_id, indices in by_video.items():
        events = load_set_events(Path(annotations) / f"{video_id}.json")
        for index in indices:
            rows[index].update(target_label=NO_EVENT_INDEX, target_delta=0.0, matched=False)
        if not events or not indices: continue
        costs = torch.tensor([[abs(rows[index]["time_sec"] - time) + (.25 if rows[index]["label"] != SET_LABELS[label] else 0.0) for index in indices] for label, time in events])
        if costs.shape[0] <= costs.shape[1]:
            event_rows, candidate_cols = hungarian_assignment(costs)
        else:  # Rare low-recall failure: retain one-to-one semantics by transposing.
            candidate_cols, event_rows = hungarian_assignment(costs.t())
        for event_index, candidate_index in zip(event_rows.tolist(), candidate_cols.tolist()):
            if costs[event_index, candidate_index] <= tolerance + .25:
                index, (label, time) = indices[candidate_index], events[event_index]
                rows[index].update(target_label=label, target_delta=float(time - rows[index]["time_sec"]), matched=True)
    return rows


def letterbox(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


class VerifierCandidateDataset(Dataset):
    def __init__(self, rows: list[dict], height: int, width: int) -> None:
        self.rows, self.height, self.width = rows, height, width

    def __len__(self) -> int: return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]; cap = cv2.VideoCapture(row["source_video"])
        if not cap.isOpened(): raise RuntimeError(f"cannot open {row['source_video']}")
        offsets = np.concatenate((np.linspace(-1, 1, 17), np.asarray([-8, -6, -4, -2, 2, 4, 6, 8], dtype=np.float32)))
        frames = []
        for offset in offsets:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, 1000 * (float(row["time_sec"]) + float(offset))))
            ok, frame = cap.read()
            if not ok: frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            else: frame = cv2.cvtColor(letterbox(frame, self.height, self.width), cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        cap.release()
        global_frames = torch.stack(frames).float().div_(255.0)
        global_frames = (global_frames - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
        return {
            "frames": global_frames, "candidate": torch.tensor(row["slot_embedding"], dtype=torch.float32),
            "shared": torch.tensor(row["shared_tokens"], dtype=torch.float32),
            "audio": torch.tensor(row["audio_tokens"], dtype=torch.float32),
            "label": torch.tensor(row["target_label"], dtype=torch.long), "delta": torch.tensor(row["target_delta"], dtype=torch.float32),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True); parser.add_argument("--annotations", required=True)
    parser.add_argument("--weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--output", required=True); parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=2); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device)
    rows = assign_oof_targets(read_rows(args.candidates), args.annotations)
    if not rows: raise RuntimeError("no OOF candidates")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    (output / "oof_target_summary.json").write_text(json.dumps({"rows": len(rows), "matched": sum(x["matched"] for x in rows), "no_temporal_nms": True}, indent=2))
    loader = DataLoader(VerifierCandidateDataset(rows, 512, 896), batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    backbone = dinov3_vitl16(pretrained=True, weights=args.weights, check_hash=False)
    model = DinoVerifier(backbone, audio_dim=64).to(device)
    configure_dinov3_vitl16(backbone, warmup=False, lora_rank=16)
    trainable_backbone = {name for name, value in backbone.named_parameters() if value.requires_grad}
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.02)
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            for name, parameter in backbone.named_parameters(): parameter.requires_grad_(name in trainable_backbone)
        model.train(); total = 0.0
        for batch in loader:
            values = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(values["frames"], values["candidate"], values["shared"], values["audio"])
                label, positive = values["label"], values["label"] != NO_EVENT_INDEX
                loss = F.cross_entropy(out["class_logits"], label) + .5 * F.binary_cross_entropy_with_logits(out["quality_logits"], positive.float())
                if positive.any(): loss = loss + 2 * F.smooth_l1_loss(out["time_delta_sec"][positive], values["delta"][positive].clamp(-2, 2))
                loss = loss + .01 * F.relu(.15 - (out["crop_boxes"][:, 0, :2] - out["crop_boxes"][:, 1, :2]).norm(dim=-1)).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += float(loss.detach())
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "trainable_backbone": sorted(trainable_backbone), "no_temporal_nms": True}
        torch.save(checkpoint, output / "last.pt")
        print(json.dumps({"epoch": epoch, "loss": total / max(len(loader), 1), "oof_rows": len(rows), "no_temporal_nms": True}), flush=True)


if __name__ == "__main__":
    main()
