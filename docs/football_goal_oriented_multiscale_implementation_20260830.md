# 足球多尺度目标导向系统：可落地实现说明（2026-08-30）

## 本轮交付边界

本轮先交付一条能直接生成“模型候选 + 有界复核片段 + 人工修正回写”的闭环，
并以 calibration18 冻结策略、test18 外部评测。它遵循
`football_goal_oriented_multiscale_system_20260830.md` 的 stop-gate 顺序：先验证完整
clip teacher 是否在真实 dense candidate 分布上增加条件排序信息；没有增益就不强行
上线第二个 DINO。

当前可运行链路为：

```text
长视频
  -> Stage-1 安全候选（5 秒 stride，事件实例不做 NMS）
  -> 完整 event-centred clip teacher（DINO + 已训练时序头）
  -> set-piece 哨声活动补充（review-only）
  -> calibration18 固定的按类排序与双阈值
  -> auto-accept / human-review / reject
  -> <=20 秒复核片段 + 标注模板
  -> 修正事件 patch + contrast set（uncertain 样本 mask loss）
```

`shot/save` 的显示区间可以共享。显示区间分组只减少重复播放，不改变候选实例，
不参与一对一事件匹配，也不是 temporal NMS。

## 正确的验收口径

- shot：calibration 目标 recall >=93%，bootstrap video recall P05 >=90%；最终 test
  recall >=90%。
- save/set-piece：calibration 目标 recall >=88%，bootstrap P05 >=85%；最终 test
  recall >=85%。
- 主人工成本 KPI：`人工复核区间并集时长 / 原始视频总时长 <35%`。
- 片段数、candidate 数只作诊断，不作为 35% KPI 的分母。
- 同类一对一、`+-3s`，不同类别独立；不合并相邻 GT 或预测。

## 已实现模块

| 模块 | 文件 | 作用 |
|---|---|---|
| 完整 clip teacher 提取 | `scripts/extract_candidate_clip_teacher_scores.py` | 对 exact Stage-1 candidates 运行完整时序模型，不只读取 DINO 特征 |
| 固定公式 stop-gate | `scripts/evaluate_candidate_clip_teacher_fusion.py` | calibration 选融合，external 原样应用 |
| 基础双阈值策略 | `scripts/goal_oriented_review_policy.py` | 稳健 recall 阈值、auto/review/reject、复核包 |
| 音频特征 | `scripts/build_audio_feature_index.py` | 5 Hz whistle/crowd 旁路特征 |
| 哨声活动候选 | `scripts/extract_whistle_activity_candidates.py` | 连续声学活动分组；不修改视觉事件 slots |
| 多模态策略 | `scripts/goal_oriented_multimodal_policy.py` | 视觉 + review-only whistle 联合 calibration |
| 复核视频渲染 | `scripts/render_goal_review_clips.py` | 并行生成最长 20 秒的 MP4 复核片段 |
| 人工回写 | `scripts/ingest_goal_review_annotations.py` | 生成 per-video event patch 和带 confounder 的 contrast set |
| 自动接续 | `scripts/watch_goal_oriented_multiscale_pipeline.py` | calibration -> test dense -> 必要时 test teacher -> 固定策略评测 |

## 当前正式实验

- Stage-1 checkpoint：停止训练后的 epoch-2 best。
- teacher：`vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828/best.pt`，
  512x896、训练到 20 epoch 的完整 clip 模型。
- calibration teacher：6 GPU、中心对齐、12,709 个 exact dense candidates，已完成。
- calibration audio：18/18 视频，已完成。
- test18：用户指定的 18 个视频；不会在该集合重选公式或阈值。

### Stage-1-only 稳健基线

calibration18、无 NMS、`+-3s`：

| 类别 | Precision | Recall | Bootstrap P05 | 备注 |
|---|---:|---:|---:|---|
| shot | 6.34% | 93.59% | 90.29% | 稳健 recall gate 通过 |
| save | 2.38% | 88.96% | 85.04% | 稳健 recall gate 通过 |
| set_piece | 2.11% | 85.62% | 82.42% | candidate ceiling 不足 |

三类复核区间并集占原视频 97.25%，不具备可落地性。set-piece 的视觉阈值几乎降到
全视频仍无法通过稳健门槛，因此本轮明确由哨声支路补候选，而不是继续降低视觉阈值。

## 运行与产物

Watchdog：

```bash
python scripts/watch_goal_oriented_multiscale_pipeline.py
```

状态：

```text
outputs/football_goal_pipeline/watchdog_20260830.state.json
outputs/football_goal_pipeline/watchdog_20260830.log
```

最终结果：

```text
outputs/football_goal_pipeline/final_result_20260830.json
outputs/football_goal_pipeline/multimodal_cal18_epoch2_20260830/
outputs/football_goal_pipeline/multimodal_test18_epoch2_20260830/
```

每个 split 包含：

```text
policy.json                    # 仅 calibration 目录；固定 score/threshold/audio policy
report.json                    # 一对一指标、逐视频结果、人工时长 KPI
predictions_selected.csv       # 独立事件 candidates，不做 NMS
review_segments.csv            # 仅 UI 播放区间
review_annotations.csv         # 人工填写模板
```

渲染复核片段：

```bash
python scripts/render_goal_review_clips.py \
  --review-segments outputs/football_goal_pipeline/multimodal_test18_epoch2_20260830/review_segments.csv \
  --video-root /mnt/data_16t/football/raw_video_720P \
  --output-dir outputs/football_goal_pipeline/multimodal_test18_epoch2_20260830/review_clips \
  --workers 6
```

人工完成 `review_annotations.csv` 后回写：

```bash
python scripts/ingest_goal_review_annotations.py \
  --annotations outputs/football_goal_pipeline/multimodal_test18_epoch2_20260830/review_annotations.csv \
  --output-dir outputs/football_goal_pipeline/multimodal_test18_epoch2_20260830/review_feedback
```

`uncertain` 项会写入 contrast set 但 `loss_mask=0`，不会被错误当作普通负样本。

## 后续提升的触发条件

本轮不会把尚未验证的长上下文 retriever 或 motion examiner 伪装成已完成模块：

- 若完整 clip teacher 在固定 robust recall 下没有降低复核时长，它不会进入 test teacher
  推理；下一步直接用本轮人工 contrast set 训练 appearance+motion examiner。
- 若 teacher 有条件排序增益但复核时长仍 >=35%，保留它作为 examiner teacher，同时构建
  8-12 FPS camera-compensated residual-motion 分支。
- 长上下文 retriever 必须在 OOF train128 上达到 shot candidate recall >=97%，且 20 秒
  region coverage <=35% 才能替换安全候选；否则继续与现有 Stage-1 做候选并集。

这保证当前交付可用、结果可审计，同时不违反文档中“每一层必须通过 stop-gate”的原则。
