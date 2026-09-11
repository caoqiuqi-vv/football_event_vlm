#!/usr/bin/env python
"""Automated VLM quality check for mined dense hard negatives.

A local vision-language model (default: Qwen3-VL-2B-Instruct on this machine)
inspects 3 frames from each of the top-scoring mined windows per class and
vetoes entries that look like *real events* (i.e. likely annotation misses).
Vetoing the top band protects training from the most damaging label noise;
the remaining entries are already protected by label-completeness gating,
safety margins, and the reduced hard-negative loss weight.

Output: a filtered manifest (same schema) plus a JSONL audit trail with the
raw VLM answers.  If anything fails mid-way, the unfiltered manifest is left
untouched and the caller can fall back to it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ["shot", "save", "set_piece"]

PROMPTS = {
    "shot": (
        "这三张图片是一场足球比赛同一个10秒视频片段的第2秒、第5秒、第8秒。"
        "只有当画面中有球员起脚将球射向对方球门时才算射门;普通传球、回传、"
        "带球推进、中场倒脚、向禁区传球都不算射门。"
        "请先用一句话描述画面内容,然后回答是否发生射门:是 / 否 / 不确定。"
    ),
    "save": (
        "这三张图片是一场足球比赛同一个10秒视频片段的第2秒、第5秒、第8秒。"
        "只有当守门员扑出、挡住或没收对方射向球门的球时才算扑救;守门员普通接球、"
        "发球门球、无人射门的站位都不算扑救。"
        "请先用一句话描述画面内容,然后回答是否发生扑救:是 / 否 / 不确定。"
    ),
    "set_piece": (
        "这三张图片是一场足球比赛同一个10秒视频片段的第2秒、第5秒、第8秒。"
        "只有角球、任意球、点球或中圈开球的主罚动作或主罚前的站位准备才算定位球;"
        "运动战中的普通传导、防守站位、门前混战都不算定位球。"
        "请先用一句话描述画面内容,然后回答是否发生定位球:是 / 否 / 不确定。"
    ),
}


def grab_frame(path: str, sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(sec, 0.0) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def parse_verdict(text: str) -> str:
    """Verdict comes after the one-sentence description; parse from the tail.

    The answer usually ends with "是否发生X:是/否/不确定" — extract the text
    after the last colon first (note "是否" itself contains 否), then fall
    back to tail heuristics.
    """
    import re

    matches = re.findall(r"[:：]\s*(不确定|是|否)\s*[。\.\s]*$", text.strip())
    if matches:
        return {"是": "yes", "否": "no", "不确定": "uncertain"}[matches[-1]]
    tail = text.strip()[-12:].replace("是否", "")
    if "不确定" in tail:
        return "uncertain"
    if "否" in tail or "不是" in tail or "没有" in tail or "不算" in tail:
        return "no"
    if "是" in tail:
        return "yes"
    return "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path,
                        help="filtered manifest path")
    parser.add_argument("--audit", type=Path, default=None,
                        help="JSONL audit trail (default: <output>.audit.jsonl)")
    parser.add_argument("--model-path", default="/mnt/data_16t/VLM_model/Qwen3-VL-2B-Instruct")
    parser.add_argument("--top-per-class", type=int, default=150)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoProcessor

    import train_football_events as base

    cfg = base.load_config(args.config, [])
    roots = cfg["data"]["long_video"]["roots"]
    video_dirs = {r["source"]: r["videos_dir"] for r in roots}
    overrides: dict[str, str] = {}
    for r in roots:
        overrides.update(r.get("video_path_overrides", {}) or {})

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]

    # top-scoring entries per class (union)
    to_check: dict[int, list[str]] = {}
    for label in LABELS:
        candidates = [e for e in entries if label in e["mined_for"]]
        candidates.sort(key=lambda e: e["scores"].get(label, 0.0), reverse=True)
        for e in candidates[: args.top_per_class]:
            to_check.setdefault(id(e), e["mined_for"])
    check_entries = [e for e in entries if id(e) in to_check]
    print(f"checking {len(check_entries)} entries with VLM", flush=True)

    if "internvl" in str(args.model_path).lower():
        from transformers import InternVLForConditionalGeneration as ModelCls
    else:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    model = ModelCls.from_pretrained(
        args.model_path, dtype=torch.bfloat16,
    ).to(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()

    audit_path = args.audit or args.output.with_suffix(".audit.jsonl")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    verdicts: dict[int, dict[str, str]] = {}
    audit_f = audit_path.open("w", encoding="utf-8")
    for index, entry in enumerate(check_entries):
        vid = entry["video_id"]
        path = overrides.get(vid) or str(Path(video_dirs[entry["source"]]) / f"{vid}.mp4")
        if not Path(path).exists():
            continue
        times = [
            float(entry["start"]) + 2.0,
            0.5 * (float(entry["start"]) + float(entry["end"])),
            float(entry["end"]) - 2.0,
        ]
        frames = [grab_frame(path, t) for t in times]
        frames = [f for f in frames if f is not None]
        if len(frames) < 3:
            continue
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        entry_verdicts: dict[str, str] = {}
        for label in entry["mined_for"]:
            content: list[dict] = [
                {"type": "image", "image": frame} for frame in frames
            ]
            content.append({"type": "text", "text": PROMPTS[label]})
            messages = [{"role": "user", "content": content}]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            text = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0]
            verdict = parse_verdict(text)
            entry_verdicts[label] = verdict
            audit_f.write(json.dumps({
                "video_id": vid, "start": entry["start"], "label": label,
                "score": entry["scores"].get(label), "answer": text,
                "verdict": verdict,
            }, ensure_ascii=False) + "\n")
        verdicts[id(entry)] = entry_verdicts
        if (index + 1) % 25 == 0:
            audit_f.flush()
            print(f"  checked {index + 1}/{len(check_entries)}", flush=True)
    audit_f.close()

    vetoed = 0
    kept_entries = []
    for entry in entries:
        v = verdicts.get(id(entry))
        if v and any(verdict == "yes" for verdict in v.values()):
            vetoed += 1
            continue
        kept_entries.append(entry)

    manifest["vlm_review"] = {
        "model": args.model_path,
        "top_per_class": args.top_per_class,
        "checked": len(check_entries),
        "vetoed": vetoed,
        "audit": str(audit_path),
    }
    manifest["entries"] = kept_entries
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(f"checked={len(check_entries)} vetoed={vetoed} kept={len(kept_entries)}")
    print(f"filtered_manifest={args.output} audit={audit_path}")


if __name__ == "__main__":
    main()
