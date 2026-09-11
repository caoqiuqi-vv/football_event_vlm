#!/usr/bin/env python
"""Summarize a fixed stage1 run without retuning or claiming human GT accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",required=True); args=ap.parse_args()
    out=Path(args.output_dir); manifest=json.loads((out/"manifest.json").read_text())
    metrics=[json.loads(p.read_text()) for p in sorted(out.glob("metrics_epoch_*.json"))]
    if not metrics: raise RuntimeError("No completed formal validation yet")
    rng=np.random.default_rng(20260907); comparisons={}
    for m in metrics:
        rows=json.loads((out/f"predictions_epoch_{m['epoch']:03d}.json").read_text())
        assert len(rows)==len(manifest["records"]["val"])
        assert len({r["index"] for r in rows})==len(rows)
        comparisons[str(m["epoch"])]=result={}
        for c,name,metric in [(0,"ball","within_16px"),(1,"goal","peak_inside_box")]:
            groups={}
            for row in rows:
                if row["class"]!=c: continue
                video=manifest["records"]["val"][row["index"]]["video_id"]
                groups.setdefault(video,[]).append(float(row["adapt"][metric])-float(row["control"][metric]))
            delta=np.array([np.mean(values) for values in groups.values()])
            ci=np.percentile(delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(1),[2.5,97.5])
            result[name]={"metric":metric,"videos":len(delta),"video_mean_delta_pp":float(delta.mean()*100),
                "conditional_video_bootstrap_95CI_pp":(ci*100).tolist(),"improved_videos":int((delta>0).sum()),
                "per_video_delta_pp":{k:float(np.mean(v)*100) for k,v in groups.items()}}
    latest=metrics[-1]; val=latest["val"]
    eligible=[m for m in metrics if m["feature_preservation_pass"]]
    selected=max(eligible,key=lambda m:m["selection_score"]) if eligible else None
    summary={"complete":(out/"COMPLETE.json").exists(),"latest_epoch":latest["epoch"],
        "selected_epoch":None if selected is None else selected["epoch"],"selected_is_adapted":bool(selected and selected["adapt"]),
        "comparisons":comparisons,"scope":"automatic positive-label localization agreement; not full detection accuracy or independent blind test",
        "stage2_started":False}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    lines=["# Stage 1 定位实验结果", "", "状态："+("训练已完成。" if summary["complete"] else "训练仍在进行，以下仅含已完成验证。"), "",
        "初始化为用户更正的 720P fromlast 实验 best（epoch 3）；输入 720×1280。验证为同源自动正标注，不是人工 GT 检测精度，也未测试无目标帧误报。", "",
        "|轮次|阶段|足球 ≤16px：冻结 / 微调|球门入框：冻结 / 微调|patch / CLS 余弦|", "|---|---|---:|---:|---:|"]
    for m in metrics:
        v=m["val"]; a=v["adapt"]; b=v["control"]
        lines.append(f"|{m['epoch']}|{'LoRA + KL' if m['adapt'] else '公共头预热'}|{b['ball']['within_16px']*100:.2f}% / {a['ball']['within_16px']*100:.2f}%|{b['goal']['peak_inside_box']*100:.2f}% / {a['goal']['peak_inside_box']*100:.2f}%|{v['patch_cosine']:.6f} / {v['cls_cosine']:.6f}|")
    lines += ["",f"按预先配置的保真门槛和定位平均分，best 对应 epoch {summary['selected_epoch']}。若来自预热，不能称为成功的特征微调。", "",
        f"最新验证 patch KL={val['patch_kl']:.6g}，CLS KL={val['cls_kl']:.6g}。KL 与余弦约束输出特征，不能保证全部语义能力不变。", "",
        "逐视频配对差值（最新一轮；正值为微调优于冻结对照）："]
    for name in ("ball","goal"):
        r=comparisons[str(latest['epoch'])][name];lo,hi=r['conditional_video_bootstrap_95CI_pp']
        lines.append(f"- {name}：{r['video_mean_delta_pp']:+.2f} 个百分点，条件 95% 区间 [{lo:+.2f}, {hi:+.2f}]；改善视频 {r['improved_videos']}/{r['videos']}。")
    lines += ["", "区间以本次拟合模型为条件，不覆盖检查点选择、多轮开发和伪标签误差；不能把它称作独立测试显著收益。球门指标仅为一个峰是否落入任一已知门框，不是完整框 AP。", "",
        "Stage 2 保持关闭。先结合本轮与冻结对照的差值确认是否值得保留特征适配，再核查同源真实目标、遮挡和无目标背景上的可靠性。所有参数、数据来源、限制见 PROTOCOL.md。"]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},ensure_ascii=False))


if __name__=="__main__": main()
