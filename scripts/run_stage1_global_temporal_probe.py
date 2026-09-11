#!/usr/bin/env python
"""Causal event probe for the Stage-1 localization-adapted DINO backbone.

The official test18 split is an untouched holdout.  This program only uses the
predeclared validation calibration/development split from the Stage2 manifest.
It runs:
  A. original DINO + original epoch-3 temporal head (reference),
  B. Stage-1-adapted DINO + the same frozen temporal head (zero-shot),
  C. Stage-1-adapted DINO + trained post-DINO temporal head.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_football_events as football
from football_localization_full import atomic_json, digest
from football_stage2_metrics import EventCurves, LABELS
from football_stage2_optional import FrozenStage2Extractor, WindowFrames

OUT = ROOT / "outputs/football_localization_stage2/720p_stage1_global_temporal_probe_20260909"
BASE = ROOT / "outputs/football_localization_stage2/720p_optional_ball_roi_from_stage1e2_20260909"
PYTHON = "/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"
TEST_IDS = ROOT / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt"


def make_config() -> dict:
    return {
        "output_dir": str(OUT),
        "base_manifest": str(BASE / "manifest.json"),
        "base_index": str(BASE / "arrays/index.json"),
        "original_features": str(BASE / "arrays/global_features.npy"),
        "event_config": str(BASE / "event_config.json"),
        "source_checkpoint": str(
            ROOT / "outputs/football_localization_stage1/720p_full_native_kl_from720best_20260907/source_720p_best.pt"
        ),
        "stage1_checkpoint": str(
            ROOT / "outputs/football_localization_stage1/720p_full_native_kl_from720best_20260907/best_adapt.pt"
        ),
        "control_checkpoint": str(BASE / "weights/frozen_control_epoch1.pt"),
        "gpus": [1, 3, 4, 6],
        "seeds": [42, 43],
        "epochs": 6,
        "batch_size": 256,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "pos_weight": [1.0, 2.5, 4.0],
        "recall_floors": [0.97, 0.92, 0.90],
        "cache_workers": 2,
        "cache_frame_chunk": 8,
        "minimum_free_gpu_memory_mb": 28000,
        "feature_dtype": "float16",
        "selection_metric": "calibration_window_macro_ap",
        "trainable_modules": ["frame_proj", "temporal", "head"],
        "event_protocol": {
            "clip_sec": 10.0,
            "stride_sec": 5.0,
            "tolerance_sec": 2.0,
            "no_nms": True,
        },
        "scope": (
            "validation-only causal representation probe; calibration thresholds and checkpoint selection use "
            "7 validation videos; development reporting uses 8 validation videos; official test18 is untouched"
        ),
    }


def setup() -> tuple[dict, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = make_config()
    manifest = json.loads(Path(cfg["base_manifest"]).read_text())
    index = json.loads(Path(cfg["base_index"]).read_text())
    unique = {r["key"]: r for rows in manifest["splits"].values() for r in rows}
    expected = [r["key"] for r in sorted(unique.values(), key=lambda r: (r["video_id"], r["start_sec"], r["key"]))]
    assert index["keys"] == expected, "base cache index and manifest differ"
    test_ids = {x.strip() for x in TEST_IDS.read_text().splitlines() if x.strip()}
    used = {r["video_id"] for rows in manifest["splits"].values() for r in rows}
    overlap = sorted(test_ids & used)
    assert not overlap, f"official test18 leakage: {overlap}"
    provenance = {
        "created_unix": time.time(),
        "config": cfg,
        "manifest_sha256": digest(cfg["base_manifest"]),
        "index_sha256": digest(cfg["base_index"]),
        "source_sha256": digest(cfg["source_checkpoint"]),
        "stage1_sha256": digest(cfg["stage1_checkpoint"]),
        "test18_ids_sha256": digest(TEST_IDS),
        "test18_overlap": overlap,
        "test18_used_for_training_calibration_selection_or_reporting": False,
        "counts": manifest["counts"],
    }
    if (OUT / "config.json").exists():
        assert json.loads((OUT / "config.json").read_text()) == cfg
    else:
        atomic_json(OUT / "config.json", cfg)
        atomic_json(OUT / "PROVENANCE.json", provenance)
    return cfg, manifest


def record_map(manifest: dict) -> dict:
    return {r["key"]: r for rows in manifest["splits"].values() for r in rows}


def adapted_path(final: bool = True) -> Path:
    name = "adapted_global_features.npy" if final else "adapted_global_features.partial.npy"
    return OUT / "arrays" / name


def initialize_cache(cfg: dict) -> None:
    arrays = OUT / "arrays"
    arrays.mkdir(exist_ok=True)
    if adapted_path().exists() or adapted_path(False).exists():
        return
    index = json.loads(Path(cfg["base_index"]).read_text())
    mm = np.lib.format.open_memmap(
        adapted_path(False), mode="w+", dtype=np.float16,
        shape=(len(index["keys"]), 16, 2048),
    )
    mm.flush()


def cache_rank(cfg: dict, manifest: dict, rank: int) -> None:
    lock = (OUT / f"cache_rank{rank}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    index = json.loads(Path(cfg["base_index"]).read_text())
    records_by_key = record_map(manifest)
    assigned = list(range(rank, len(index["keys"]), len(cfg["gpus"])))
    progress_path = OUT / f"cache_rank{rank}_progress.json"
    completed = 0
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        assert state["assigned"] == len(assigned)
        completed = int(state["completed"])
    if completed >= len(assigned):
        atomic_json(OUT / f"cache_rank{rank}_complete.json", {"complete": True, "assigned": len(assigned)})
        return
    chosen_indices = assigned[completed:]
    records = [records_by_key[index["keys"][i]] for i in chosen_indices]
    model = FrozenStage2Extractor(cfg).cuda().eval()
    loader = DataLoader(
        WindowFrames(records), batch_size=1, num_workers=cfg["cache_workers"],
        pin_memory=True, prefetch_factor=1, multiprocessing_context="spawn",
    )
    bank = np.load(adapted_path(False), mmap_mode="r+")
    started = time.time()
    for local_step, (frames, local_index) in enumerate(loader, 1):
        position = chosen_indices[int(local_index[0])]
        clip = frames[0].cuda(non_blocking=True)
        pieces = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(clip), cfg["cache_frame_chunk"]):
                pieces.append(model.extract_global_pair(clip[start:start + cfg["cache_frame_chunk"]])["adapted_global"])
        value = torch.cat(pieces).float().cpu().numpy().astype(np.float16)
        assert np.isfinite(value).all()
        bank[position] = value
        done = completed + local_step
        if local_step == 1 or local_step % 20 == 0 or done == len(assigned):
            bank.flush()
            state = {
                "phase": "extracting_adapted_global_features",
                "rank": rank,
                "completed": done,
                "assigned": len(assigned),
                "elapsed_sec_this_run": time.time() - started,
                "updated_unix": time.time(),
            }
            atomic_json(progress_path, state)
            atomic_json(OUT / f"cache_rank{rank}_heartbeat.json", state)
            print(json.dumps(state), flush=True)
    bank.flush()
    atomic_json(OUT / f"cache_rank{rank}_complete.json", {"complete": True, "assigned": len(assigned)})


def finalize_cache(cfg: dict) -> None:
    for rank in range(len(cfg["gpus"])):
        assert (OUT / f"cache_rank{rank}_complete.json").exists()
    array = np.load(adapted_path(False), mmap_mode="r")
    for start in range(0, len(array), 512):
        assert np.isfinite(array[start:start + 512]).all()
    del array
    adapted_path(False).replace(adapted_path())
    atomic_json(OUT / "CACHE_COMPLETE.json", {
        "complete": True,
        "sha256": digest(adapted_path()),
        "shape": [len(json.loads(Path(cfg["base_index"]).read_text())["keys"]), 16, 2048],
        "dtype": "float16",
    })


def make_event_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    source = torch.load(cfg["source_checkpoint"], weights_only=True, map_location="cpu", mmap=True)
    event_cfg = football.to_config(json.loads(Path(cfg["event_config"]).read_text()))
    football.configure_label_schema(event_cfg)
    model = football.make_model(event_cfg, use_cached_features=True, device=torch.device("cpu"))
    state = {k: v for k, v in source["model"].items() if not k.startswith("backbone.")}
    model.load_state_dict(state, strict=True)
    return model.to(device)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def window_ap(prob: np.ndarray, rows: list[dict]) -> dict:
    target = np.asarray([r["labels"] for r in rows], dtype=np.float32)
    masks = np.asarray([r["label_mask"] for r in rows], dtype=np.float32)
    result = {}
    for c, label in enumerate(LABELS):
        valid = masks[:, c] > 0
        y = target[valid, c] > 0.5
        score = prob[valid, c]
        positives = int(y.sum())
        if positives == 0:
            value = 0.0
        else:
            order = np.argsort(-score, kind="stable")
            sorted_y = y[order]
            precision = np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)
            value = float(precision[sorted_y].sum() / positives)
        result[label] = value
    result["macro"] = float(np.mean([result[x] for x in LABELS]))
    return result


def group_indices(cfg: dict, manifest: dict) -> dict[str, np.ndarray]:
    index = json.loads(Path(cfg["base_index"]).read_text())
    lookup = {key: i for i, key in enumerate(index["keys"])}
    return {
        group: np.asarray([lookup[r["key"]] for r in manifest["splits"][group]], dtype=np.int64)
        for group in ["train", "calibration", "development"]
    }


@torch.no_grad()
def predict(model: torch.nn.Module, features: np.ndarray, indices: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    outputs = []
    for start in range(0, len(indices), batch_size):
        ids = indices[start:start + batch_size]
        # Apply identical cache quantization to both arms so zero-shot differs
        # only by the Stage-1 LoRA representation, not storage precision.
        x_np = np.asarray(features[ids], dtype=np.float16).astype(np.float32)
        x = torch.from_numpy(x_np).to(device, non_blocking=True)
        outputs.append(model._global_branch_outputs(x)["logits"].float().cpu())
    return torch.cat(outputs).numpy()


def review_workload(prob: np.ndarray, thresholds: list[float], rows: list[dict], manifest: dict) -> dict:
    selected = np.any(prob >= np.asarray(thresholds)[None], axis=1)
    intervals: dict[str, list[tuple[float, float]]] = {}
    for keep, row in zip(selected, rows):
        if keep:
            intervals.setdefault(row["video_id"], []).append((float(row["start_sec"]), float(row["end_sec"])))
    merged_seconds = 0.0
    segments = 0
    for spans in intervals.values():
        spans.sort()
        lo, hi = spans[0]
        for a, b in spans[1:]:
            if a <= hi:
                hi = max(hi, b)
            else:
                merged_seconds += hi - lo
                segments += 1
                lo, hi = a, b
        merged_seconds += hi - lo
        segments += 1
    videos = sorted({r["video_id"] for r in rows})
    total = sum(float(manifest["videos"][v]["duration"]) for v in videos)
    return {
        "selected_windows": int(selected.sum()),
        "total_windows": len(rows),
        "merged_segments": segments,
        "review_seconds": merged_seconds,
        "video_seconds": total,
        "review_duration_ratio": merged_seconds / max(total, 1e-12),
    }


def evaluate_arm(name: str, prob_cal: np.ndarray, prob_dev: np.ndarray, cfg: dict, manifest: dict,
                 common_thresholds: list[float] | None = None) -> dict:
    cal_rows = manifest["splits"]["calibration"]
    dev_rows = manifest["splits"]["development"]
    cal_curves = EventCurves(cal_rows, manifest, cfg["event_protocol"]["tolerance_sec"])
    dev_curves = EventCurves(dev_rows, manifest, cfg["event_protocol"]["tolerance_sec"])
    thresholds, cal_metrics = cal_curves.tune(prob_cal, cfg["recall_floors"])
    result = {
        "name": name,
        "thresholds_calibration_only": thresholds,
        "calibration": cal_metrics,
        "development": dev_curves.evaluate(prob_dev, thresholds),
        "calibration_window_ap": window_ap(prob_cal, cal_rows),
        "development_window_ap": window_ap(prob_dev, dev_rows),
        "development_review_workload": review_workload(prob_dev, thresholds, dev_rows, manifest),
    }
    if common_thresholds is not None:
        result["development_at_original_common_thresholds"] = dev_curves.evaluate(prob_dev, common_thresholds)
        result["development_review_at_original_common_thresholds"] = review_workload(
            prob_dev, common_thresholds, dev_rows, manifest
        )
    return result


def zero_shot(cfg: dict, manifest: dict) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    groups = group_indices(cfg, manifest)
    original = np.load(cfg["original_features"], mmap_mode="r")
    adapted = np.load(adapted_path(), mmap_mode="r")
    model = make_event_model(cfg, device)
    predictions = {}
    for name, bank in [("original", original), ("adapted_zero_shot", adapted)]:
        predictions[name] = {}
        for group in ["calibration", "development"]:
            predictions[name][group] = predict(model, bank, groups[group])
    original_result = evaluate_arm(
        "original_dino_original_head", sigmoid(predictions["original"]["calibration"]),
        sigmoid(predictions["original"]["development"]), cfg, manifest,
    )
    common = original_result["thresholds_calibration_only"]
    adapted_result = evaluate_arm(
        "stage1_adapted_dino_same_frozen_head", sigmoid(predictions["adapted_zero_shot"]["calibration"]),
        sigmoid(predictions["adapted_zero_shot"]["development"]), cfg, manifest, common,
    )
    np.savez_compressed(
        OUT / "zero_shot_predictions.npz",
        original_calibration=predictions["original"]["calibration"],
        original_development=predictions["original"]["development"],
        adapted_calibration=predictions["adapted_zero_shot"]["calibration"],
        adapted_development=predictions["adapted_zero_shot"]["development"],
    )
    atomic_json(OUT / "ZERO_SHOT_RESULT.json", {
        "original": original_result,
        "adapted_zero_shot": adapted_result,
        "only_changed_variable": "last-four-block Stage1 localization LoRA in DINO global features",
        "test18_untouched": True,
    })
    atomic_json(OUT / "ZERO_SHOT_COMPLETE.json", {"complete": True})


def train_temporal(cfg: dict, manifest: dict, seed: int) -> None:
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dest = OUT / f"temporal_seed{seed}"
    dest.mkdir(exist_ok=True)
    lock = (dest / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (dest / "COMPLETE.json").exists():
        return
    device = torch.device("cuda")
    model = make_event_model(cfg, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for module_name in cfg["trainable_modules"]:
        module = getattr(model, module_name)
        module.requires_grad_(True)
        trainable.extend(module.parameters())
    parameter_count = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    adapted = np.load(adapted_path(), mmap_mode="r")
    groups = group_indices(cfg, manifest)
    train_rows = manifest["splits"]["train"]
    target = np.asarray([r["labels"] for r in train_rows], dtype=np.float32)
    masks = np.asarray([r["label_mask"] for r in train_rows], dtype=np.float32)
    pos_weight = torch.tensor(cfg["pos_weight"], device=device)
    zero_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best = {"epoch": 0, "calibration_macro_ap": -1.0, "state": zero_state}
    history = []
    for epoch in range(0, cfg["epochs"] + 1):
        if epoch > 0:
            model.train()
            order = np.random.default_rng(seed * 1000 + epoch).permutation(len(groups["train"]))
            loss_sum = 0.0
            steps = 0
            grad_max = 0.0
            started = time.time()
            for start in range(0, len(order), cfg["batch_size"]):
                local = order[start:start + cfg["batch_size"]]
                ids = groups["train"][local]
                x = torch.from_numpy(np.array(adapted[ids], dtype=np.float32, copy=True)).to(device, non_blocking=True)
                y = torch.from_numpy(target[local]).to(device)
                mask = torch.from_numpy(masks[local]).to(device)
                ratio = ((epoch - 1) * len(order) + start) / max(cfg["epochs"] * len(order), 1)
                lr = cfg["learning_rate"] * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * ratio)))
                optimizer.param_groups[0]["lr"] = lr
                logits = model._global_branch_outputs(x)["logits"]
                loss = (F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight, reduction="none"
                ) * mask).sum() / mask.sum().clamp_min(1.0)
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                assert torch.isfinite(grad)
                optimizer.step()
                loss_sum += float(loss.detach())
                steps += 1
                grad_max = max(grad_max, float(grad))
                if steps == 1 or steps % 25 == 0:
                    heartbeat = {
                        "phase": "training_post_dino_temporal_head", "seed": seed, "epoch": epoch,
                        "step": steps, "steps": math.ceil(len(order) / cfg["batch_size"]),
                        "loss": float(loss.detach()), "lr": lr, "updated_unix": time.time(),
                    }
                    atomic_json(dest / "heartbeat.json", heartbeat)
                    print(json.dumps(heartbeat), flush=True)
            assert grad_max > 0
            train_summary = {
                "loss_mean": loss_sum / steps, "gradient_norm_max": grad_max,
                "elapsed_sec": time.time() - started,
            }
        else:
            train_summary = {"loss_mean": None, "gradient_norm_max": None, "elapsed_sec": 0.0}
        cal_logits = predict(model, adapted, groups["calibration"], cfg["batch_size"])
        cal_ap = window_ap(sigmoid(cal_logits), manifest["splits"]["calibration"])
        row = {"epoch": epoch, "calibration_window_ap": cal_ap, **train_summary}
        history.append(row)
        atomic_json(dest / f"epoch_{epoch:03d}.json", row)
        if cal_ap["macro"] > best["calibration_macro_ap"] + 1e-12:
            best = {
                "epoch": epoch,
                "calibration_macro_ap": cal_ap["macro"],
                "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            }
            torch.save({
                "epoch": epoch, "model": best["state"], "config": cfg, "seed": seed,
                "calibration_window_ap": cal_ap, "trainable_parameter_count": parameter_count,
            }, dest / "best.pt")
    selected = torch.load(dest / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(selected["model"], strict=True)
    cal_logits = predict(model, adapted, groups["calibration"], cfg["batch_size"])
    dev_logits = predict(model, adapted, groups["development"], cfg["batch_size"])
    original_common = json.loads((OUT / "ZERO_SHOT_RESULT.json").read_text())["original"]["thresholds_calibration_only"]
    result = evaluate_arm(
        f"adapted_dino_trained_temporal_seed{seed}", sigmoid(cal_logits), sigmoid(dev_logits),
        cfg, manifest, original_common,
    )
    result.update({
        "seed": seed, "selected_epoch": int(selected["epoch"]),
        "trainable_parameter_count": parameter_count, "history": history,
        "dino_frozen": True, "initialized_from_original_epoch3_head": True,
        "test18_untouched": True,
    })
    np.savez_compressed(dest / "predictions.npz", calibration=cal_logits, development=dev_logits)
    atomic_json(dest / "RESULT.json", result)
    atomic_json(dest / "COMPLETE.json", {"complete": True, "selected_epoch": int(selected["epoch"])})


def report(cfg: dict, manifest: dict) -> None:
    zero = json.loads((OUT / "ZERO_SHOT_RESULT.json").read_text())
    trained = [json.loads((OUT / f"temporal_seed{s}" / "RESULT.json").read_text()) for s in cfg["seeds"]]
    best = max(trained, key=lambda x: x["calibration_window_ap"]["macro"])
    summary = {
        "execution_complete": True,
        "test18_untouched": True,
        "selection_and_threshold_source": "validation calibration split only",
        "report_source": "validation development split only",
        "zero_shot": zero,
        "trained_seeds": trained,
        "selected_trained_seed_by_calibration_only": best["seed"],
        "selected_trained_result": best,
    }
    atomic_json(OUT / "FINAL_SUMMARY.json", summary)
    arms = [zero["original"], zero["adapted_zero_shot"], *trained]
    lines = [
        "# Stage-1 检测适配 DINO：zero-shot 与时序头训练实验", "",
        "本实验不接触 test18。阈值与 checkpoint 仅由 validation-calibration 选择，最终数字来自 validation-development。",
        "10 秒窗口、5 秒步长、±2 秒事件覆盖、无 NMS。阈值召回约束为 shot 97%、save 92%、set-piece 90%。", "",
        "|实验|选中epoch|shot P/R|save P/R|set-piece P/R|宏P/R|宏AP|审核时长占比|", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        d = arm["development"]
        ap = arm["development_window_ap"]["macro"]
        epoch = arm.get("selected_epoch", 0)
        workload = arm["development_review_workload"]["review_duration_ratio"]
        lines.append(
            f"|{arm['name']}|{epoch}|{d['shot']['precision']*100:.2f}% / {d['shot']['recall']*100:.2f}%|"
            f"{d['save']['precision']*100:.2f}% / {d['save']['recall']*100:.2f}%|"
            f"{d['set_piece']['precision']*100:.2f}% / {d['set_piece']['recall']*100:.2f}%|"
            f"{d['macro_precision']*100:.2f}% / {d['macro_recall']*100:.2f}%|{ap*100:.2f}%|{workload*100:.2f}%|"
        )
    base = zero["original"]["development"]
    zs = zero["adapted_zero_shot"]["development"]
    lines += ["", "## 解释边界", "",
              "zero-shot 行是严格因果对照：帧、预处理、原时序头和分类头完全相同，只替换 Stage-1 检测监督适配后的 DINO 表征。",
              "训练时序头行可回答适配表征在重新拟合后是否更有用，但其增益包含表示变化与时序头再优化，不能单独归因于检测监督。",
              f"zero-shot 开发集宏 precision 变化 {(zs['macro_precision']-base['macro_precision'])*100:+.2f}pp，宏 recall 变化 {(zs['macro_recall']-base['macro_recall'])*100:+.2f}pp。",
              "每类阈值、FP窗口/小时、共同原阈值结果、预测数组和各 epoch loss/AP 轨迹见 JSON/NPZ 文件。"]
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(OUT / "PIPELINE_COMPLETE.json", {"complete": True, "report": str(OUT / "FINAL_REPORT.md")})


def run_jobs(cfg: dict, jobs: list[tuple[str, list[str]]], phase: str) -> None:
    queue = list(jobs)
    active = {}
    attempts = {}
    while queue or active:
        raw = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"
        ], text=True)
        free_memory = {int(line.split(",")[0]): int(line.split(",")[1]) for line in raw.splitlines()}
        occupied = {value[1] for value in active.values()}
        free = [gpu for gpu in cfg["gpus"] if gpu not in occupied and free_memory.get(gpu, 0) >= cfg["minimum_free_gpu_memory_mb"]]
        while queue and free:
            tag, args = queue.pop(0)
            gpu = free.pop(0)
            attempts[tag] = attempts.get(tag, 0) + 1
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
            log = (OUT / f"{tag}.log").open("a")
            process = subprocess.Popen(
                [PYTHON, str(Path(__file__).resolve()), *args], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            active[tag] = (process, gpu, log, args)
        atomic_json(OUT / "pipeline_status.json", {
            "phase": phase, "supervisor_pid": os.getpid(),
            "active": {k: {"pid": v[0].pid, "gpu": v[1], "attempt": attempts[k]} for k, v in active.items()},
            "queued": len(queue), "updated_unix": time.time(),
        })
        time.sleep(5)
        for tag, (process, gpu, log, args) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            del active[tag]
            if code != 0:
                if attempts[tag] < 2:
                    queue.append((tag, args))
                else:
                    raise RuntimeError(f"{tag} failed twice; see {OUT / (tag + '.log')}")


def pipeline(cfg: dict, manifest: dict) -> None:
    lock = (OUT / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        if not (OUT / "CACHE_COMPLETE.json").exists():
            initialize_cache(cfg)
            run_jobs(cfg, [(f"cache_rank{r}", ["--phase", "cache", "--rank", str(r)]) for r in range(len(cfg["gpus"]))], "adapted_feature_cache")
            finalize_cache(cfg)
        if not (OUT / "ZERO_SHOT_COMPLETE.json").exists():
            run_jobs(cfg, [("zero_shot", ["--phase", "zero-shot"])], "zero_shot_evaluation")
        train_jobs = [(f"temporal_seed{s}", ["--phase", "train", "--seed", str(s)]) for s in cfg["seeds"]]
        run_jobs(cfg, train_jobs, "temporal_head_training")
        report(cfg, manifest)
        atomic_json(OUT / "pipeline_status.json", {"phase": "complete", "updated_unix": time.time()})
    except Exception as error:
        atomic_json(OUT / "pipeline_status.json", {"phase": "failed", "error": str(error), "updated_unix": time.time()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pipeline", "cache", "zero-shot", "train", "report"], default="pipeline")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg, manifest = setup()
    if args.phase == "pipeline":
        pipeline(cfg, manifest)
    elif args.phase == "cache":
        cache_rank(cfg, manifest, args.rank)
    elif args.phase == "zero-shot":
        zero_shot(cfg, manifest)
    elif args.phase == "train":
        train_temporal(cfg, manifest, args.seed)
    else:
        report(cfg, manifest)


if __name__ == "__main__":
    main()
