#!/usr/bin/env python3
"""Four-GPU strict one-to-one calibration for the no-ROI/no-NMS verifier."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.set_data import load_set_events
from football_e2e_spotter.set_spotting import SET_LABELS
from football_e2e_spotter.verifier_data import VerifierCandidateDataset, assign_oof_targets, read_rows
from football_e2e_spotter.verifier_fullimage import FullImageDinoVerifier
from football_e2e_spotter.verifier_lora_runtime import configure


class Indexed(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset[index]
        item["index"] = torch.tensor(index)
        return item


def metric(rows: list[dict], label: int, threshold: float, targets: dict[str, list[float]]) -> dict[str, float | int]:
    remaining = {video: list(times) for video, times in targets.items()}
    true_positive = false_positive = 0
    accepted = sorted(
        (row for row in rows if row["final_label"] == label and row["final_score"] >= threshold),
        key=lambda row: row["final_score"], reverse=True,
    )
    for row in accepted:
        values = remaining.get(row["video_id"], [])
        if values:
            nearest = min(range(len(values)), key=lambda index: abs(values[index] - row["time_sec"]))
            if abs(values[nearest] - row["time_sec"]) <= 3:
                true_positive += 1
                values.pop(nearest)
                continue
        false_positive += 1
    false_negative = sum(len(values) for values in remaining.values())
    return {"tp": true_positive, "fp": false_positive, "fn": false_negative,
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rows = assign_oof_targets(read_rows(args.candidates), args.annotations)
    backbone = dinov3_vitl16(pretrained=True, weights=args.weights, check_hash=False)
    model = FullImageDinoVerifier(backbone, audio_dim=64).to(device)
    configure(backbone, warmup=False)
    checkpoint = torch.load(args.verifier, map_location="cpu", weights_only=False)
    if checkpoint.get("verifier_variant") != "full_image_only":
        raise ValueError("expected a full-image-only verifier checkpoint")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dataset = Indexed(VerifierCandidateDataset(rows, 512, 896))
    loader = DataLoader(dataset, batch_size=1, sampler=DistributedSampler(dataset, shuffle=False), num_workers=0)
    local_predictions: list[tuple[int, list[float], float, float]] = []
    with torch.inference_mode():
        for batch in loader:
            index = int(batch.pop("index"))
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["frames"], batch["candidate"], batch["shared"], batch["audio"])
            local_predictions.append((index, result["class_logits"][0].float().cpu().tolist(), float(result["quality_logits"][0].sigmoid()), float(result["time_delta_sec"][0])))
    gathered: list[list[tuple[int, list[float], float, float]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_predictions)
    if rank == 0:
        for partition in gathered:
            assert partition is not None
            for index, logits, quality, delta in partition:
                rows[index].update(verifier_logits=logits, verifier_quality=quality, time_sec=rows[index]["time_sec"] + delta)
        coefficients = []
        for label in range(5):
            features = torch.tensor([[row["stage1_logits"][label], row["verifier_logits"][label], row["verifier_quality"]] for row in rows])
            target = torch.tensor([row["target_label"] == label for row in rows], dtype=torch.float)
            linear = torch.nn.Linear(3, 1)
            optimizer = torch.optim.LBFGS(linear.parameters(), lr=.3, max_iter=100)
            def closure() -> torch.Tensor:
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(linear(features).squeeze(), target)
                loss.backward()
                return loss
            optimizer.step(closure)
            coefficients.append({"weight": linear.weight.detach().flatten().tolist(), "bias": float(linear.bias.detach())})
            for row, score in zip(rows, linear(features).detach().squeeze().tolist()):
                row.setdefault("final_logits", []).append(score)
        for row in rows:
            row["final_label"] = int(torch.tensor(row["final_logits"]).argmax())
            row["final_score"] = float(torch.tensor(row["final_logits"])[row["final_label"]].sigmoid())
        targets: dict[int, defaultdict[str, list[float]]] = {label: defaultdict(list) for label in range(5)}
        for video, annotation in {row["video_id"]: str(row.get("annotation_id", row["video_id"])) for row in rows}.items():
            for label, time_sec in load_set_events(Path(args.annotations) / f"{annotation}.json"):
                targets[label][video].append(time_sec)
        required = (.90, .85, .85, .85, .85)
        report: dict[str, object] = {"no_temporal_nms": True, "tolerance_seconds": 3., "verifier_variant": "full_image_only", "roi_branch": False, "coefficients": coefficients, "classes": {}}
        for label, name in enumerate(SET_LABELS):
            best = None
            for threshold in sorted({row["final_score"] for row in rows if row["final_label"] == label}, reverse=True):
                value = metric(rows, label, threshold, targets[label])
                if value["recall"] >= required[label] and (best is None or value["precision"] > best["precision"]):
                    best = {"threshold": threshold, **value}
            report["classes"][name] = best or {"threshold": 1., **metric(rows, label, 1., targets[label])}
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "calibration_eval.json").write_text(json.dumps(report, indent=2) + "\n")
        (output / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        print(json.dumps(report, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
