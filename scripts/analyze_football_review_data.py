#!/usr/bin/env python
"""Analyze football event review DBs: human verdicts + queue metadata (2026-08-20).

Sources:
  - whistle_v24: /mnt/data_16t/football/eval_outputs/recall_first_d7_frame_whistle_ui/reviews.sqlite3
    (724 candidates, 7 videos, 53 human verdicts on one video)
  - GT: /home/new_users/qiuqi/code/football_events_human_repair/<video_id>.json

Answers:
  1. Human verdicts: GT 漏标率 (accepted on fp), GT 虚标率 (deleted on matched), label edits.
  2. Queue metadata: fp distance-to-GT distribution, cross-class frame evidence on far FPs,
     whistle score on set_piece matched vs fp (kickoff hypothesis), whistle_rescue nature.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

DB = "/mnt/data_16t/football/eval_outputs/recall_first_d7_frame_whistle_ui/reviews.sqlite3"
GT_DIR = Path("/home/new_users/qiuqi/code/football_events_human_repair")
OUT = Path("outputs/football_review_analysis")
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["shot", "save", "set_piece"]

# GT 中文标签 -> (主类, 子类型); set_piece 子类型可直接用于评估端拆分
LABEL_MAP = {
    "射门": ("shot", None),
    "扑救": ("save", None),
    "任意球": ("set_piece", "free_kick"),
    "角球": ("set_piece", "corner"),
    "点球": ("set_piece", "penalty"),
    "中圈开球": ("set_piece", "kickoff"),
}


def parse_ts(ts: str) -> float:
    """'00:00:20.514' -> seconds"""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_gt(video_id: str) -> list[dict]:
    p = GT_DIR / f"{video_id}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    evs = d.get("events", []) if isinstance(d, dict) else d
    out = []
    for e in evs:
        lab = e.get("label", "")
        if lab not in LABEL_MAP:
            continue
        main, sub = LABEL_MAP[lab]
        out.append({"label": main, "subtype": sub, "time_sec": parse_ts(e["timestamp"])})
    return out


def main() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    events = [dict(r) for r in con.execute("SELECT * FROM events")]
    reviews = {r["event_id"]: dict(r) for r in con.execute("SELECT * FROM reviews")}
    con.close()

    rows = []
    for e in events:
        pl = json.loads(e["payload_json"])
        rv = reviews.get(e["id"], {})
        rows.append({**e, "pl": pl, "rv": rv})

    videos = sorted({r["video_id"] for r in rows})
    gt_by_video = {v: load_gt(v) for v in videos}
    report: list[str] = []
    out = report.append

    out("# 足球事件 review 数据分析 (2026-08-20)\n")
    out(f"DB: `{DB}`\n候选数 {len(rows)},视频 {len(videos)}: {videos}\n")

    # ---------------- A. human verdicts ----------------
    verdicts = [r for r in rows if r["rv"].get("status") not in (None, "unreviewed")]
    out(f"\n## A. 人工判定 (n={len(verdicts)})\n")
    if verdicts:
        vtab = Counter((r["rv"]["status"], r["pl"].get("evaluation_status")) for r in verdicts)
        out("| verdict | eval_status | count |")
        out("|---|---|---|")
        for (s, ev), c in sorted(vtab.items()):
            out(f"| {s} | {ev} | {c} |")

        fp_judged = [r for r in verdicts if r["pl"].get("evaluation_status") == "fp"]
        matched_judged = [r for r in verdicts if r["pl"].get("evaluation_status") == "matched"]
        acc_fp = [r for r in fp_judged if r["rv"]["status"] == "accepted"]
        del_matched = [r for r in matched_judged if r["rv"]["status"] == "deleted"]
        out(f"\n**GT 漏标率**: {len(acc_fp)}/{len(fp_judged)} = {len(acc_fp)/len(fp_judged):.1%} 的系统判 fp 候选被人工确认为真事件"
            f" (label: {dict(Counter(r['source_label'] for r in acc_fp))})")
        out(f"**GT 疑似虚标率**: {len(del_matched)}/{len(matched_judged)} = {len(del_matched)/len(matched_judged):.1%} 的 matched 候选被人工删除"
            f" (label: {dict(Counter(r['source_label'] for r in del_matched))})")

        # set_piece subtypes from verdicts
        sec = Counter()
        for r in verdicts:
            try:
                for s in json.loads(r["rv"].get("secondary_labels_json") or "[]"):
                    sec[s] += 1
            except Exception:
                pass
        if sec:
            out(f"\n**set_piece 子类型 (审核中指定)**: {dict(sec)}")

    # ---------------- B. queue metadata ----------------
    out(f"\n## B. 队列元数据 (全部 {len(rows)} 候选)\n")
    src = Counter(r["pl"].get("review_source", "None") for r in rows)
    out(f"review_source 分布: {dict(src)}")
    evtab = Counter(r["pl"].get("evaluation_status", "?") for r in rows)
    out(f"evaluation_status 分布: {dict(evtab)}")

    # distance to nearest GT for fp candidates
    fp_rows = [r for r in rows if r["pl"].get("evaluation_status") == "fp"]
    matched_rows = [r for r in rows if r["pl"].get("evaluation_status") == "matched"]
    out(f"\nfp 候选 {len(fp_rows)} 个, matched {len(matched_rows)} 个")

    dist_by_label: dict[str, list[float]] = defaultdict(list)
    for r in fp_rows:
        gts = [e for e in gt_by_video.get(r["video_id"], []) if e.get("label") == r["source_label"]]
        t = r["pl"]["time_sec"]
        if gts:
            d = min(abs(e.get("time_sec", 1e9) - t) for e in gts)
        else:
            d = None
        dist_by_label[r["source_label"]].append(d)
    out("\n**fp 到最近同类 GT 的距离分布**:")
    out("| label | n | median | p25 | p75 | >10s 占比 | >30s 占比 |")
    out("|---|---|---|---|---|---|---|")
    for lab in LABELS:
        ds = [d for d in dist_by_label[lab] if d is not None]
        n_none = sum(1 for d in dist_by_label[lab] if d is None)
        if not ds:
            out(f"| {lab} | {len(dist_by_label[lab])} | - | - | - | - | - |")
            continue
        ds_s = sorted(ds)
        import statistics
        med = statistics.median(ds_s)
        p25 = ds_s[len(ds_s) // 4]
        p75 = ds_s[3 * len(ds_s) // 4]
        out(f"| {lab} | {len(ds)} | {med:.1f} | {p25:.1f} | {p75:.1f} | "
            f"{sum(1 for d in ds if d > 10)/len(ds):.0%} | {sum(1 for d in ds if d > 30)/len(ds):.0%} |")
        if n_none:
            out(f"  ({lab}: {n_none} 个 fp 所在视频无同类 GT)")

    # cross-class frame evidence on far fp
    out("\n**远 FP (距同类 GT>10s) 的跨类 frame 证据** (frame_detection_scores 中其他类的均值):")
    out("| label | n | 同类的分数均值 | 其他类最大分数均值 |")
    out("|---|---|---|---|")
    for lab in LABELS:
        far = []
        for r in fp_rows:
            if r["source_label"] != lab:
                continue
            ds = dist_by_label[lab]
            # need index alignment; recompute here
            gts = [e for e in gt_by_video.get(r["video_id"], []) if e.get("label") == lab]
            d = min(abs(e.get("time_sec", 1e9) - r["pl"]["time_sec"]) for e in gts) if gts else 1e9
            if d > 10:
                far.append(r)
        if not far:
            out(f"| {lab} | 0 | - | - |")
            continue
        same_scores, other_scores = [], []
        for r in far:
            fds = r["pl"].get("frame_detection_scores", {})
            same_scores.append(fds.get(lab, 0.0))
            others = [v for k, v in fds.items() if k != lab]
            other_scores.append(max(others) if others else 0.0)
        import statistics
        out(f"| {lab} | {len(far)} | {statistics.mean(same_scores):.3f} | {statistics.mean(other_scores):.3f} |")

    # whistle on set_piece: matched vs fp
    sp = [r for r in rows if r["source_label"] == "set_piece"]
    if sp:
        import statistics
        out("\n**set_piece 的 whistle_score 分布 (kickoff 假设)**:")
        out("| status | n | 有 whistle 信号 | whistle_score 均值 |")
        out("|---|---|---|---|")
        for ev in ["matched", "fp", "whistle_rescue"]:
            grp = [r for r in sp if r["pl"].get("evaluation_status") == ev]
            if not grp:
                continue
            with_ws = [r for r in grp if r["pl"].get("whistle_score") is not None]
            means = statistics.mean([r["pl"]["whistle_score"] for r in with_ws]) if with_ws else 0.0
            out(f"| {ev} | {len(grp)} | {len(with_ws)} ({len(with_ws)/len(grp):.0%}) | {means:.3f} |")

    # set_piece subtype composition: nearest GT subtype for matched vs fp vs whistle_rescue
    sp_sub = defaultdict(Counter)
    sp_any = Counter()
    for r in sp:
        gts = [e for e in gt_by_video.get(r["video_id"], []) if e.get("label") == "set_piece"]
        ev = r["pl"].get("evaluation_status")
        sp_any[ev] += 1
        if not gts:
            sp_sub[ev]["no_set_piece_gt"] += 1
            continue
        t = r["pl"]["time_sec"]
        nearest = min(gts, key=lambda e: abs(e["time_sec"] - t))
        sp_sub[ev][nearest["subtype"]] += 1
    if sp_any:
        out("\n**set_piece 最近 GT 子类型构成 (matched vs fp)**:")
        out("| status | n | free_kick | corner | penalty | kickoff | 无 set_piece GT |")
        out("|---|---|---|---|---|---|---|")
        for ev in ["matched", "fp", "whistle_rescue"]:
            c = sp_sub[ev]
            if not c:
                continue
            out(f"| {ev} | {sp_any[ev]} | {c.get('free_kick',0)} | {c.get('corner',0)} | {c.get('penalty',0)} | "
                f"{c.get('kickoff',0)} | {c.get('no_set_piece_gt',0)} |")

    # whistle_rescue nature
    wr = [r for r in rows if r["pl"].get("evaluation_status") == "whistle_rescue"]
    if wr:
        out(f"\n**whistle_rescue 候选 ({len(wr)})**: 距最近 GT (任意类) 距离分布:")
        dists = []
        for r in wr:
            gts = gt_by_video.get(r["video_id"], [])
            if not gts:
                continue
            t = r["pl"]["time_sec"]
            dists.append(min(abs(e.get("time_sec", 1e9) - t) for e in gts))
        if dists:
            import statistics
            ds_s = sorted(dists)
            out(f"  n={len(dists)} median={statistics.median(ds_s):.1f}s p25={ds_s[len(ds_s)//4]:.1f} p75={ds_s[3*len(ds_s)//4]:.1f} "
                f"≤2s(等于已标): {sum(1 for d in dists if d <= 2)/len(dists):.0%}")
        out(f"  label 分布: {dict(Counter(r['source_label'] for r in wr))}")

    txt = "\n".join(report)
    (OUT / "review_analysis_20260820.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
