#!/usr/bin/env python3
"""Calibrate and evaluate Stage-1 -> independent slots -> Verifier, no NMS."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.set_spotting import NO_EVENT_INDEX, SET_LABELS
from football_e2e_spotter.train_set_verifier import VerifierCandidateDataset, assign_oof_targets, read_rows
from football_e2e_spotter.verifier import DinoVerifier, configure_dinov3_vitl16


def one_to_one(rows: list[dict], label: int, threshold: float, targets: dict[str, list[float]], tolerance: float = 3.0) -> dict[str, float]:
    available = {key: list(value) for key, value in targets.items()}; tp = fp = 0
    selected = sorted((row for row in rows if row["final_label"] == label and row["final_score"] >= threshold), key=lambda row: row["final_score"], reverse=True)
    for row in selected:
        values = available.get(row["video_id"], [])
        if values:
            nearest = min(range(len(values)), key=lambda i: abs(values[i] - row["time_sec"]))
            if abs(values[nearest] - row["time_sec"]) <= tolerance:
                tp += 1; values.pop(nearest); continue
        fp += 1
    fn = sum(map(len, available.values()))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidates", required=True); parser.add_argument("--annotations", required=True); parser.add_argument("--verifier", required=True); parser.add_argument("--output", required=True); parser.add_argument("--weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); device = torch.device(args.device); rows = assign_oof_targets(read_rows(args.candidates), args.annotations)
    backbone = dinov3_vitl16(pretrained=True, weights=args.weights, check_hash=False); model = DinoVerifier(backbone, audio_dim=64).to(device); configure_dinov3_vitl16(backbone, warmup=False, lora_rank=16)
    model.load_state_dict(torch.load(args.verifier, map_location="cpu", weights_only=False)["model"], strict=True); model.eval()
    loader = DataLoader(VerifierCandidateDataset(rows, 512, 896), batch_size=1, shuffle=False, num_workers=0)
    with torch.inference_mode():
        for row, batch in zip(rows, loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch["frames"], batch["candidate"], batch["shared"], batch["audio"])
            row["verifier_logits"] = out["class_logits"][0].float().cpu().tolist(); row["verifier_quality"] = float(out["quality_logits"][0].sigmoid()); row["time_sec"] += float(out["time_delta_sec"][0])
    # Per-class logistic calibration: stage-1 class logit, verifier class logit, verifier quality.
    coefficients = []
    for label in range(len(SET_LABELS)):
        x = torch.tensor([[row["stage1_logits"][label], row["verifier_logits"][label], row["verifier_quality"]] for row in rows], dtype=torch.float32)
        y = torch.tensor([row["target_label"] == label for row in rows], dtype=torch.float32)
        linear = torch.nn.Linear(3, 1); optimizer = torch.optim.LBFGS(linear.parameters(), lr=.3, max_iter=100)
        def closure():
            optimizer.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(linear(x).squeeze(1), y); loss.backward(); return loss
        optimizer.step(closure); coefficients.append({"weight": linear.weight.detach().flatten().tolist(), "bias": float(linear.bias.detach())})
        for row, score in zip(rows, linear(x).detach().squeeze(1).tolist()): row.setdefault("final_logits", []).append(score)
    for row in rows:
        row["final_label"] = int(torch.tensor(row["final_logits"]).argmax()); row["final_score"] = float(torch.tensor(row["final_logits"])[row["final_label"]].sigmoid())
    targets: dict[int, dict[str, list[float]]] = {label: defaultdict(list) for label in range(len(SET_LABELS))}
    for row in rows:
        if row["matched"]: targets[row["target_label"]][row["video_id"]].append(row["time_sec"] + row["target_delta"])
    recall_targets = [.90, .85, .85, .85, .85]; report = {"no_temporal_nms": True, "coefficients": coefficients, "classes": {}}
    for label, name in enumerate(SET_LABELS):
        best = {"threshold": 1.0, "precision": 0.0, "recall": 0.0}
        for threshold in sorted({row["final_score"] for row in rows if row["final_label"] == label}, reverse=True):
            metric = one_to_one(rows, label, threshold, targets[label])
            if metric["recall"] >= recall_targets[label] and metric["precision"] >= best["precision"]: best = {"threshold": threshold, **metric}
        report["classes"][name] = best
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True); (output / "calibration_eval.json").write_text(json.dumps(report, indent=2)); (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows)); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
