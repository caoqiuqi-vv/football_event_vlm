# 足球长视频独立评测接口

该接口不接入训练 epoch，也不会占用训练流程。它对指定 checkpoint 独立执行：

1. calibration 视频 dense 推理并缓存窗口、clip score、response/frame curve；
2. 在 calibration split 上选择满足目标 Recall 的阈值；
3. 阈值不变地应用到 test split；
4. 用严格 1:1 window-overlap 匹配输出整体及 shot/save/set_piece 的 Precision、Recall；
5. 将所有类别和重叠候选对应的 UI 观看区间取并集，输出真实去重人工参与时长与比例。

## 推荐调用

```bash
python scripts/run_long_video_full_pipeline_eval.py \
  --checkpoint /absolute/path/to/best.pt \
  --cal-video-id-file configs/football/splits/thirdparty18_test_long15_val_no_pn_train/internal_val_video_ids.txt \
  --video-id-file configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt \
  --gt-dir /mnt/data_16t/football/football_events_human_repair \
  --gpu-groups '4;5;6;7' \
  --batch-size 8 \
  --num-workers 2 \
  --prediction-postprocess window_overlap \
  --match-tolerance-sec 5 \
  --review-mode cap10_peak \
  --recall-targets 0.85,0.88,0.90,0.92,0.95 \
  --primary-recall-target 0.85 \
  --recall-constraint per_class \
  --output-dir outputs/football_long_video_api/my_checkpoint_eval
```

`--output-dir` 是可复用缓存目录。checkpoint 的文件大小或修改时间改变后会生成新的 fingerprint，避免更新后的 `last.pt` 误用旧缓存。只有明确需要重算时才传 `--force`。

## 主输出

- `primary_operating_point.json/csv`：主要交付物，无逐视频明细。
- `full_pipeline_summary.json/csv`：所有 score source、Recall 目标和参与度预算的汇总。
- `reports/threshold_report_*.json`：完整审计产物，保留逐视频数据但不会打印到控制台。
- `dense_runs/`：可复用的逐视频 dense 中间结果。
- `logs/`：每个 GPU 分片日志。

主报告包含：

- overall Precision / Recall / F1 / TP / FP / FN；
- shot、save、set_piece 各自 Precision / Recall / F1 / TP / FP / FN；
- `test_total_video_minutes`：原始视频总时长；
- `test_dedup_review_minutes`：跨类别、跨窗口合并后的实际观看时长；
- `test_participation_pct`：去重观看时长 / 原始视频时长；
- `test_raw_unmerged_minutes`：未去重候选时长，仅用于检查重复率。

## 数据隔离

默认 `--recall-constraint per_class`，因此 0.85 表示 shot、save、set_piece 在 calibration 上分别都达到 Recall ≥ 0.85，而不是只要求 micro Recall 达标。可用 `--recall-constraint micro` 复现旧口径。

推荐 score source 和阈值只根据 calibration split 选择。test split 只应用固化阈值并报告结果，不参与模型、score source 或阈值选择。若不传 `--cal-video-id-file`，接口会在测试视频本身调阈值，并明确标记 `calibration_is_test=true`；这种结果只能用于诊断，不能作为无泄漏的最终指标。
