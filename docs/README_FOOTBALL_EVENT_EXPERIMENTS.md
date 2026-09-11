# 足球事件检测：技术设计、实验结论与后续计划

> 状态：记录截至 2026-08-04 的当前实现、已完成实验和后续计划。
>
> 本文档是当前足球事件实验的唯一主 README。`docs/` 下其他文档包含历史方案，
> 其中的分辨率、split、标签或 ROI 参数可能已经过时。

## 1. 任务目标与当前结论

当前任务是长视频足球事件多标签检测，标签体系为：

- `shot`：射门
- `save`：扑救
- `set_piece`：角球、任意球、点球合并

线上目标按长视频时序后处理后的事件结果评测，而不只看训练过程中的采样 clip 指标。
期望最终达到：

- precision 70% 以上
- recall 85% 以上
- 输出每个视频、每个类别的 precision/recall
- 以 PointNMS 作为主要线上事件 spotting 口径

当前代码已包含：

- DINOv3 ViT-L/16 逐帧编码
- 10 秒 clip、16 帧基础采样
- full-image 与 detector-aware ROI 双流
- 按类别自适应的 ROI gate
- Gaussian frame-event 监督
- E1-E4 时序实验及对应单流对照
- D0-D7 ROI/fusion/precision-oriented 实验
- 从训练长视频 FP 中挖掘 hard negative
- window-overlap 和 PointNMS 两种长视频评测

现有证据支持以下结论：

> ROI 分支提高了正负窗口的排序能力，也删除了大量误报；但当前固定 10 秒 ROI 和
> convex gate 主要在降低分数，尚未有效补回 full-image 分支漏掉的事件。

当前实验主线已经从“直接加 ROI 分支看是否涨点”收敛到更保守的方向：

```text
full-image E1 作为稳定主事件定位器
ROI 作为细节证据分支
fusion 先学习何时使用 ROI，再学习如何修正 full logits
目标优先提高 precision，同时尽量不牺牲 recall
```

## 2. 主要代码位置

训练与模型：

- `train_football_events.py`：dataset、模型、loss、训练、验证
- `football_detection_aware.py`：robust ROI proposal 与训练增强
- `football_roi_scoring.py`：球、球门、中圈、人群候选评分

ROI 工具：

- `scripts/build_football_roi_indices.py`：检测跟踪 JSON 转紧凑 `.pt` index
- `scripts/football_roi_crop.py`：独立复现和可视化 ROI crop
- `scripts/export_gt_event_roi_crops.py`：按人工事件点导出 ROI
- `scripts/export_val_positive_roi_videos.py`：按 val 正样本导出 full-frame overlay 与 ROI-crop 对照视频
- `docs/football_roi_crop_readme.md`：ROI 模块复用说明

评测与分析：

- `scripts/evaluate_football_model.py`：多视频评测入口
- `scripts/eval_long_video_checkpoint.py`：dense inference 与时序后处理
- `scripts/analyze_roi_branch_from_eval.py`：global/local/fused 分支分析
- `scripts/frame_aware_point_nms_from_eval.py`：frame-event-aware 重打分

实验入口：

- `scripts/run_football_da_16f.sh`
- `scripts/launch_football_detection_aware.sh`

难负例：

- `scripts/build_football_hard_negative_manifest.py`
- `configs/football/dinov3_vitl16_robust_dual_16f_e1_frame_det_hard_neg.yaml`

## 3. 数据、标签与抽帧

### 3.1 长视频数据模式

当前推荐：

```yaml
data:
  mode: long_videos
```

训练直接读取长视频和事件标注。正样本由人工事件时间点作为 anchor；负样本从没有
目标事件的窗口中抽取。必须使用视频级 train/val split，避免同一视频的事件同时进入
训练和验证。

### 3.2 Clip 采样

```yaml
video:
  num_frames: 16
  clip_duration: 10.0
  event_margin: 1.0
  temporal_jitter_sec: 1.5
  hflip_prob: 0.5
```

正样本逻辑：

- 围绕事件 anchor 随机采样 10 秒窗口
- 条件允许时，事件距离 clip 边界至少 `event_margin`
- temporal jitter 后重新计算窗口内标签
- 一个 clip 可以同时包含多个正标签

负样本逻辑：

- jitter 后再次检查窗口是否仍为负样本
- 不把被 jitter 到事件附近的窗口错误当作负样本

基础实验在 10 秒内抽 16 帧。E4 先抽 32 个候选帧，再选择事件帧和上下文帧。

### 3.3 负样本比例

```yaml
data:
  long_video:
    negative_ratio_by_split:
      train: 5.0
      val: 2.0
```

设计目标：

- 训练约 5 个随机负样本/正样本
- 验证保持过去实验的负样本比例 2
- 最终必须在 dense 长视频上评测，因为真实负样本比例远高于采样验证集

### 3.4 视频解码与缓存

```yaml
data:
  video_decode_strategy: single_seek
  video_reader_cache_size: 2
  num_workers: 8
```

`single_seek` 只 seek 到第一个目标帧，然后向前 decode。相比逐帧 seek，它更适合当前
间隔较大的稀疏抽帧。

Frame feature cache 可以减少重复 DINO 计算，但单一确定性 cache 会固定抽帧和图像增强。
除非缓存多个采样版本，否则会削弱动态抽帧和 temporal augmentation，因此主配置暂未启用。

当前训练链路没有 GPU video decoding。只有在实验定义固定以后再进行性能优化，因为即使
解码上 GPU，resize、ROI 生成、augmentation 和调度仍可能占用 CPU。

## 4. Full-image 基线

### 4.1 输入分辨率

当前高分辨率 dual full 分支：

```yaml
spatial_crop:
  global_image_size: [512, 896]
```

dual 模式下，`global_image_size` 就是 full-image 实际送入 DINO 的尺寸。
单流 `lora_r5_f32` 对照使用：

```yaml
video:
  image_size: [512, 896]
spatial_crop:
  mode: none
```

当前 E3/E4 dual 配置仍有 `[384, 672]`。在把 E1-E4 当作时序头单变量实验之前，必须统一。

### 4.2 逐帧特征

每帧独立经过 DINOv3 ViT-L/16，帧特征为：

```text
normalized CLS token + normalized patch tokens mean
```

ViT-L 输出 2048 维帧特征，再通过：

```text
LayerNorm -> Linear(2048, 512) -> GELU -> Dropout
```

### 4.3 基础时序分类

```text
16 frame tokens
-> temporal CLS token + position embedding
-> 4-layer, 8-head Transformer
-> temporal CLS feature
-> MLP multi-label classifier
```

输出 `shot/save/set_piece` 三个独立 logit，使用 `BCEWithLogitsLoss`，不是 softmax。

## 5. Robust ROI Crop

### 5.1 离线 index 流程

检测与跟踪在事件模型训练前离线完成，训练时不会实时运行 detector。

```bash
DETECTOR_ROOT=/path/to/detection_and_track_result \
VIDEO_ROOT=/path/to/raw_video_720P \
bash scripts/launch_football_detection_aware.sh index 0
```

流程：

```text
detection/tracking JSON
-> 2 fps 检测证据
-> 10 fps 球轨迹证据
-> outputs/football_roi_indices/robust_v2/{video_id}.pt
```

action stage 会读取配置中的 split，尝试为缺失视频补建 index，然后生成只包含可用 index
的 train/val 列表。缺失 ID 会写入：

```text
train_missing_roi_index_video_ids.txt
val_missing_roi_index_video_ids.txt
```

`require_index: true` 表示 dual ROI 实验不能在缺失 index 时静默退化训练。

### 5.2 当前 ROI 策略

`RobustClipCropper` 聚合完整 10 秒窗口的检测/跟踪结果，只生成一个稳定 ROI。

候选优先级：

1. `goal_ball`
2. `goal_players`
3. `ball_players`
4. `center_circle_players`
5. invalid ROI，回退 global

核心逻辑：

- 优先使用球跟踪信息，而不是孤立的球检测
- 尽量由球和附近人群决定 crop focus
- 球门可作为边缘 anchor 被包含，但不作为默认 crop 中心
- 球门和球不能同时放入时，优先 fallback 到 `ball_players`
- crop 过大时先删除低优先级人群，保留语义 anchor
- source crop 宽高比和 local 输入一致
- source crop 尽可能是 local target size 的整数倍

当前参数：

```yaml
video:
  image_size: [384, 640]

spatial_crop:
  padding: 0.15
  min_crop_area_ratio: 0.0
  max_crop_area_ratio: 0.35
  min_roi_confidence: 0.40
  goal_conf: 0.40
  person_conf: 0.35
  center_circle_conf: 0.35
  raw_ball_conf: 0.10
  min_goal_frames: 3
  min_ball_points: 3
  max_people: 10
```

local tensor 最终 resize 到 `[384,640]`，但 bbox 位于原视频坐标系。source crop 可能是
`384x640`、`768x1280` 等整数倍，也可能 invalid。invalid ROI 不参与最终融合。

### 5.3 当前最大的空间限制

```text
一个 10 秒 proposal -> 16 帧全部使用同一个 bbox
```

当前 crop 不会逐帧跟随球、人群和镜头运动。这是 controlled LoRA 实验之后最需要验证的
空间策略问题。

### 5.4 ROI 训练增强

```yaml
noise_augmentation:
  invalid_prob: 0.15
  ball_drop_prob: 0.10
  goal_drop_prob: 0.10
  isolated_false_positive_prob: 0.05
  center_jitter: 0.08
  scale_min: 1.0
  scale_max: 1.0
```

full/local 使用完全一致的帧索引和水平翻转。

### 5.5 可视化 ROI

```bash
python scripts/football_roi_crop.py \
  --index-root outputs/football_roi_indices/robust_v2 \
  --video-root /mnt/data_16t/football/raw_video_720P \
  --video-id 2027564888428580866 \
  --start-sec 100 \
  --end-sec 110 \
  --num-frames 16 \
  --target-size 384,640 \
  --save-overlay \
  --save-source-crop \
  --output-dir outputs/football_roi_crop_debug/2027564888428580866_100_110
```

## 6. Dual 模型与自适应 ROI 融合

### 6.1 双流结构

```text
full [B,T,3,512,896]
    -> shared DINO
    -> global frame projection / temporal head / classifier
    -> global logits

ROI [B,T,3,384,640]
    -> shared DINO
    -> local frame projection / temporal head / classifier
    -> local logits
```

DINO backbone 共享；frame projection、时序模块、分类头和 frame-event head 分开。
当单流 checkpoint 初始化 dual 时，local 模块从 global 参数复制。

### 6.2 ROI metadata

Gate 使用 16 维 ROI metadata：

- valid、confidence、area ratio
- goal、ball、center-circle、nearby-person 分数
- proposal mode one-hot
- fallback group one-hot

### 6.3 按类别自适应 gate

```python
gate_input = concat(global_temporal, local_temporal, roi_meta)
learned_gate = sigmoid(roi_gate(gate_input))
alpha = learned_gate * roi_confidence * roi_valid
final_logits = global_logits + alpha * (local_logits - global_logits)
```

特性：

- 每个 clip、每个标签独立 gate
- invalid ROI 时 `alpha=0`，严格回退 global
- gate 最后一层 bias 初始化为 `-3`，初始接近 global-only
- ROI confidence 限制 local 最大贡献

当前只有融合权重可学习，crop 坐标由规则产生，不可微、不可学习。

### 6.4 Dual loss

```text
fused clip BCE
  + local_loss_weight * valid-local clip BCE
```

E1-E4 当前 `local_loss_weight: 0.2`。local auxiliary loss 要求局部 crop 独立完成整个
clip 分类；当 ROI 主动移除全局上下文时，这个约束可能过强。

启用 frame supervision 时：

```text
total loss += frame_det_loss_weight
              * average(global frame loss, valid-local frame loss)
```

### 6.5 Freeze 参数区别

- `freeze_backbone`：构建 backbone 时的初始 DINO/LoRA trainability
- `freeze_loaded_backbone`：加载 init checkpoint 后冻结 DINO/LoRA
- `freeze_global_branch`：冻结 global projection、temporal、classifier；当前实现还会冻结共享 backbone

当前 Exp4：

```yaml
freeze_backbone: true
freeze_loaded_backbone: true
freeze_global_branch: true
backbone_lr: 0.0
```

即训练 local 时序/分类模块和 ROI gate，global 与加载的 DINO/LoRA 保持固定。

当前代码下打开 LoRA 训练需要：

```yaml
freeze_backbone: false
freeze_loaded_backbone: false
freeze_global_branch: false
```

由于 `freeze_global_branch` 会冻结共享 DINO，当前无法做到“global head 冻结但 shared LoRA
训练”的严格消融。若要只研究 LoRA，需要增加 head-only freeze。

## 7. E1：Gaussian Frame Event Detection

### 7.1 Gaussian target

人工标注点不作为单帧硬标签：

```text
target_i,c = exp(-0.5 * ((t_i - t_event) / sigma_c)^2)
```

同类别多个事件在每帧取 max。

```yaml
frame_label_sigma_sec:
  shot: 0.8
  save: 0.8
  set_piece: 1.5

frame_label_ignore_radius_sec:
  shot: 3.0
  save: 3.0
  set_piece: 5.0
```

正样本 clip 中，远离标注但可能仍属于持续动作的帧通过 mask 忽略，不强制为 0；负样本
clip 中对应类别所有有效帧监督为 0。

### 7.2 Frame loss

```text
frame_loss =
    0.5 * masked focal heatmap BCE
  + 0.3 * top-k MIL loss
  + 0.2 * frame ranking loss

total loss += 0.5 * frame_loss
```

输出包括：

- `frame_event_logits`
- dual 模式的 `local_frame_event_logits`
- frame top-k indices
- GT 附近 top-k hit rate

E1 中 frame logits 是 auxiliary。普通 clip inference 不会自动使用它，只有 frame-aware
后处理或 E4 selector 才会把它作用到最终事件输出。

## 8. E1-E4 时序实验

| 实验 | Temporal fusion | 候选帧 | 核心问题 |
| --- | --- | ---: | --- |
| E0/Exp4 | `cls_transformer` | 16 | 当前 dual ROI baseline |
| E1 | CLS + frame loss | 16 | frame eventness 是否改善定位？ |
| E2 | `attn_pool_transformer` | 16 | temporal CLS pooling 是否受限？ |
| E3 | `class_query_transformer` | 16 | 不同标签是否应关注不同时刻？ |
| E4 | `event_topk_transformer` | 32 -> 16 | eventness 能否选出更有效帧？ |

E2 对 Transformer 的 frame tokens 做可学习 attention pooling。

E3 为每个类别设置独立 temporal query，使 shot/save 可以关注短时动作，set-piece 可以关注
更长上下文。

E4：

```yaml
video:
  candidate_num_frames: 32
model:
  event_topk: 12
  context_frames: 4
```

当前 32 帧都会经过 DINO。detached eventness 选 top12，再加入 4 个均匀上下文帧，去重并
按时间排序。clip classification loss 不穿过 hard top-k index，eventness 由 frame loss 训练。

## 9. E1-E4 单流对照

| 实验 | Config |
| --- | --- |
| E1 | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e1_frame_det.yaml` |
| E2 | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e2_attn_pool.yaml` |
| E3 | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e3_class_query.yaml` |
| E4 | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e4_event_topk.yaml` |

这些配置：

- `spatial_crop.mode: none`
- `view_fusion: single`
- 默认初始化 `/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_last.pt`
- `freeze_loaded_backbone: false`
- `backbone_lr: 3e-5`
- 不依赖 ROI index

## 10. Hard-negative Mining

当前流程：

1. 在有 index 的训练视频上 dense inference
2. 保存每个 10 秒窗口分数
3. 选择高置信 FP
4. 拒绝人工事件附近候选
5. 按视频/类别去重和限量
6. 写 hard-negative manifest
7. 训练加载时再次对当前标注做安全检查

默认值：

```text
挖掘类别: shot, save
最低分数: shot=0.30, save=0.40
GT 拒绝范围: 5 秒
每视频每类别最多: 40
训练每视频最多: 80
验证集 hard negative: disabled
```

Manifest：

```text
outputs/football_hard_negatives/e1_exp2_best_train_fp_shot_save.json
```

默认不挖 set-piece 是实验选择，不是代码限制。set-piece 存在持续行为和标注边界歧义，
应先人工审查再加入。设置 `E1_HARD_NEG_REJECT_ANY_LABEL_GT=1` 可以拒绝任何目标类别附近
的候选，而不只拒绝当前挖掘类别。

```bash
bash scripts/run_football_da_16f.sh e1_mine_hard_neg 0,1
bash scripts/run_football_da_16f.sh e1_hard_neg 0,1
bash scripts/run_football_da_16f.sh e1_hard_neg_eval 0
```

两张 80G A800：

```bash
E1_HARD_NEG_MINE_BATCH_SIZE=8 \
E1_HARD_NEG_MINE_NUM_WORKERS=8 \
bash scripts/run_football_da_16f.sh e1_mine_hard_neg 0,1
```

## 11. Init Checkpoint 与 Resume

`model.init_checkpoint` 只加载能够匹配的模型参数，允许新/旧 head 差异和 temporal position
resize，不恢复 optimizer。

`train.resume` 恢复 model、optimizer、scheduler 和 scaler。改变 trainable 参数组后恢复旧
optimizer 可能报错：

```text
ValueError: loaded state dict has a different number of parameter groups
```

改变时序头、frame head 或 freeze 策略时，应使用 `model.init_checkpoint`，关闭 optimizer resume。

```bash
ACTION_INIT_CHECKPOINT=/path/to/best.pt \
bash scripts/run_football_da_16f.sh e1 0,1

INIT_CHECKPOINT=/path/to/best.pt \
bash scripts/run_football_da_16f.sh all 0,1

LORA_R5_F32_INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_last.pt \
bash scripts/run_football_da_16f.sh e1_lora_r5_f32 0,1
```

## 12. 评测口径

### 12.1 Sampled validation

训练验证输出：默认阈值 0.5、按 sampled clip F1 搜索的 tuned threshold、mAP、mAUROC、
P/R/F1、per-video 指标，以及启用 frame supervision 时的 top-k hit rate。

Sampled clip tuned threshold 不一定适合 dense 长视频 PointNMS。

### 12.2 Window-overlap

保留每个超过阈值的 sliding window，同一个 GT 可以匹配多个重叠窗口。适合分析 clip classifier、
分支排序和 recall coverage，但比线上 spotting 更宽松。

### 12.3 PointNMS

把每个窗口转成 point candidate，按类别做 score-ordered temporal NMS，然后 prediction/GT
一对一匹配。

```text
clip: 10 秒
stride: 5 秒
NMS radius: 5 秒
match tolerance: 5 秒
```

PointNMS 是主要线上口径。

### 12.4 阈值来源

报告时必须标记：

- `0.5`：固定阈值
- `checkpoint`：sampled validation tuned
- `long-video-val`：独立 dense validation tuned

不能在最终测试视频上调阈值后把结果当作泛化性能。

### 12.5 标准六视频

```text
2027564888428580866
2027572095412195330
2027572406738604033
2042125172076392450
2042520971893485569
2042525152494694401
```

## 13. 当前实验结果

### 13.1 Robust dual Exp4

```text
outputs/football_events/vitl16_robust_dual_16f_exp4/best.pt
```

Best epoch 13，阈值 `shot=0.20`、`save=0.45`、`set_piece=0.45`。

```text
sampled-val mAP=0.6417, mAUROC=0.9087
micro P/R/F1=0.5908/0.7744/0.6702
macro P/R/F1=0.5827/0.7418/0.6489
```

### 13.2 六视频 window-overlap

| 类别 | lora_r5_last P/R/F1 | Dual ROI P/R/F1 |
| --- | --- | --- |
| shot | 0.323 / 0.884 / 0.473 | 0.417 / 0.829 / 0.555 |
| save | 0.165 / 0.783 / 0.272 | 0.235 / 0.667 / 0.348 |
| set_piece | 0.467 / 0.495 / 0.481 | 0.476 / 0.707 / 0.569 |
| micro | 0.285 / 0.739 / 0.411 | 0.384 / 0.755 / 0.509 |

Dual 提高 precision/F1，但 checkpoint 阈值下 shot/save recall 下降。

### 13.3 六视频 PointNMS

| 类别 | lora_r5_last P/R/F1 | Dual ROI P/R/F1 |
| --- | --- | --- |
| shot | 0.251 / 0.781 / 0.379 | 0.345 / 0.760 / 0.474 |
| save | 0.136 / 0.652 / 0.226 | 0.200 / 0.580 / 0.297 |
| set_piece | 0.377 / 0.465 / 0.416 | 0.340 / 0.657 / 0.448 |

### 13.4 ROI 分支行为

| 类别 | 删除负窗口 FP | 丢失正窗口 | 新增正窗口 |
| --- | ---: | ---: | ---: |
| shot | 381 | 35 | 2 |
| save | 315 | 28 | 0 |
| set_piece | 174 | 25 | 0 |

Diagnostic AP/AUROC：

| 类别 | Global AP/AUC | Local AP/AUC | Fused AP/AUC |
| --- | --- | --- | --- |
| shot | 0.365 / 0.896 | 0.405 / 0.860 | 0.415 / 0.906 |
| save | 0.152 / 0.880 | 0.209 / 0.868 | 0.195 / 0.896 |
| set_piece | 0.391 / 0.882 | 0.277 / 0.827 | 0.421 / 0.890 |

结论：local 对 shot/save 排序有帮助但分数偏低；local set-piece 弱于 global；fusion 主要
删除 FP，几乎没有补回 global 漏检。

分析文件：

```text
outputs/football_eval_runs/
  vitl16_robust_dual_16f_exp4_roi_v2_best_6videos_window_overlap_checkpoint_thr/
  roi_branch_window_overlap_thr_ckpt.json
```

### 13.5 E1 frame supervision

```text
outputs/football_events/
  vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr/
  best.pt
```

Best epoch 8：

```text
mAP=0.6680, mAUROC=0.9070
micro P/R/F1=0.5805/0.7823/0.6665
frame top-k hit: shot=0.746, save=0.744, set_piece=0.899
```

Frame eventness 对时间排序有效，但是否提高最终长视频事件输出，需要通过 frame-aware
PointNMS 或 E4 验证。

### 13.6 ROI 分辨率

local 从 `[384,640]` 提高到 `[448,768]`：

```text
mAP:      0.6684 -> 0.6720
macro F1: 0.6617 -> 0.6576
macro R:  0.7174 -> 0.7547
```

Recall 上升但 precision 下降，macro F1 没提高。ROI 分辨率不是下一步最高优先级。

## 14. 已知混杂因素与限制

### 14.1 当前不是严格 ROI 单变量实验

Robust dual 与 `lora_r5_last` 在 split、index 数量、init checkpoint、LoRA trainability、
pos_weight、global freeze、local loss 和部分分辨率上都不同。必须建立 same-split、same-init、
same-optimizer 对照后，才能单独归因 ROI。

### 14.2 当前 E1-E4 配置未对齐

正式比较前统一：split/index coverage、global/local resolution、init、trainable modules、
effective batch、update 次数、`pos_weight` 和阈值来源。

当前 split 已统一：

- 所有正式实验配置统一使用
  `vitl16_lvd1689m_set_piece_seed42_162videos`（train 127 / val 35）
- `detector_aware_smoke` 仅用于 smoke test，不参与正式实验比较

仍然不一致：

- E1/E2 global `[512,896]`
- E3/E4 global `[384,672]`

未修正前不能把结果解释为纯时序头消融。

### 14.3 模型限制

- 固定 ROI 无法跟随 10 秒内动作和镜头运动
- ROI confidence 是 proposal 可靠性，不是事件有效性
- local logits 偏低时 convex fusion 主要执行抑制
- local loss 可能过度惩罚缺少全局上下文的 crop
- set-piece 不一定适合和 shot/save 共用同一种 ROI 使用策略

## 15. 后续实验计划

每轮只改变一个核心因素。

### P0. Controlled dual LoRA adaptation

目标：验证 frozen full-image LoRA 是否导致 local crop 域没有适配。

D0/D1 都从以下 checkpoint 开始：

```text
outputs/football_events/vitl16_robust_dual_16f_exp4/best.pt
```

固定同一 indexed split、ROI、seed、`pos_weight`、update 次数和 effective batch。

D0 frozen continuation：

```yaml
freeze_loaded_backbone: true
freeze_global_branch: true
backbone_lr: 0.0
```

D1 LoRA adaptation：

```yaml
freeze_backbone: false
freeze_loaded_backbone: false
freeze_global_branch: false
backbone_lr: 0.00001
```

两张 80G A800 推荐：

```yaml
train:
  epochs: 6
  batch_size: 16
  grad_accum_steps: 6
  lr: 0.00015
  backbone_lr: 0.00001
```

显存允许可用 batch 24、accumulation 4。

验收信号：

- local shot/save AP 提高
- 正窗口 `local > global` 比例提高
- `global miss -> fused hit` 数量提高
- PointNMS recall 上升，同时不重新引入大部分已删除 FP

注意：当前 D1 也允许 global head 更新。严格 LoRA-only 对照需要新增 head-only freeze。

### P1. 长视频阈值校准

在独立 dense validation 视频上按 PointNMS 调每类阈值，记录 PR curve、recall floor 下阈值、
跨视频稳定性和 per-video 方差。

### P2. Event-time ROI coverage audit

在 GT ±1 秒统计：球覆盖率、附近人群覆盖率、shot/save 的球门/门将证据、crop area/mode、
global/local/fused 分数。建立四类样本：

1. global hit -> fused miss
2. global miss -> local/fused hit
3. global FP -> fused removed
4. 两个分支都 miss

先抽查 10 个 val 正样本：

```bash
ROI_DEBUG_MAX_SAMPLES=10 \
ROI_DEBUG_OUTPUT=outputs/football_roi_debug/val_positive_sample10 \
bash scripts/launch_football_detection_aware.sh val_positive_roi_videos 0
```

只检查 `save` 和 `set_piece`：

```bash
ROI_DEBUG_LABELS=save,set_piece \
ROI_DEBUG_OUTPUT=outputs/football_roi_debug/val_positive_save_set_piece \
bash scripts/launch_football_detection_aware.sh val_positive_roi_videos 0
```

全量导出：

```bash
bash scripts/launch_football_detection_aware.sh val_positive_roi_videos 0
```

默认输出 `videos/review/`：左侧是原图与逐帧 ROI 框，右侧是 resize 后的 ROI 输入。
`--frame-mode model` 是默认值，只导出 dataset 真正送入模型的 16 帧；`manifest.csv` 保存
每帧 proposal、valid/confidence、中心轨迹长度以及人工审核列。检查动态版本必须显式指定配置：

```bash
ROI_DEBUG_CONFIG=configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_d1.yaml \
ROI_DEBUG_MAX_SAMPLES=20 \
ROI_DEBUG_OUTPUT=outputs/football_roi_debug/dynamic_roi_d1_sample20 \
bash scripts/launch_football_detection_aware.sh val_positive_roi_videos 0
```

### P3. Dynamic ROI（已实现）

配置 `spatial_crop.temporal_mode: dynamic` 后，10 秒窗口不再共享一个 bbox：

1. 先按模型真实采样的 16 个 frame time，各自使用默认 3 秒局部上下文运行 robust proposal；
2. 在 `(center_x, center_y, scale)` 语义上做 confidence-weighted 时间邻域平均；
3. 参考 `roi_crop_union.py` 使用 Kalman 平滑，但 measurement noise 会按 proposal confidence 调整；
4. 短漏检最多 hold 1.5 秒，并按 1 秒时间常数衰减 confidence；超过 gap 后优先回退到完整
   10 秒窗口计算的固定 ROI，而不是把 local 输入退化为整帧；仅当固定 ROI 也无效时才使用整帧兜底；
5. 大幅无支撑中心跳变会重置 Kalman 状态，避免镜头切换后旧轨迹长时间拖尾；
6. bbox 始终保持分类器目标宽高比和整数尺度，720p 当前通常为原图 `640x384` crop 后输入
   `384x640`，不是先裁很大区域再强下采样。

`temporal_causal: false` 使用窗口内前后帧，适用于当前完整 10 秒 clip 推理；低延迟流式推理需改为
`true`，不能把离线 look-ahead 指标直接当成在线指标。

D1 只改变 ROI 时序策略，仍用旧 `dual_gate`，用于测量动态 crop 本身的收益：

```bash
bash scripts/run_football_da_16f.sh d1_dynamic_roi_debug 0,1
bash scripts/run_football_da_16f.sh d1_dynamic_roi 0,1
bash scripts/run_football_da_16f.sh d1_dynamic_roi_eval 0
```

默认从当前 E1 best 初始化，也可用 `DYNAMIC_ROI_D1_INIT_CHECKPOINT=/path/to/best.pt` 覆盖。

### P4. Feature-quality fusion（已实现）

D2 保留独立 local 分类头，但它不再直接参与最终 logit 插值：

```text
g_t, l_t = global/local projected frame tokens
q_t = sigmoid(FrameQuality(g_t, l_t, |l_t-g_t|, roi_frame_meta_t)) * roi_frame_valid_t
f_t = g_t + q_t * ZeroInitAdapter(l_t)
h_f = SharedTemporal(f_1 ... f_T)
delta_c = ZeroInitResidual(h_global, h_local, h_f, roi_meta)_c
q_c = sigmoid(ClassQuality(h_global, h_local, h_f, roi_meta))_c
final_logit_c = global_logit_c + q_c * mean(q_t) * delta_c
```

关键约束：

- `ZeroInitAdapter` 与 `ZeroInitResidual` 让初始化预测严格等于 global，不先破坏已有能力；
- local clip BCE 继续训练 local 分类头；local Gaussian/MIL/rank frame loss继续训练 ROI 帧响应；
- local frame loss按逐帧 ROI valid mask，而不是只看一个 clip-level valid；
- learned quality 的辅助目标为
  `sigmoid((BCE_global - BCE_feature_candidate) / temperature)`，target stop-gradient；它学习“ROI
  特征是否能降低该类别误差”，不是重复学习事件概率；
- detector/track confidence只是 quality head 的输入特征，不再硬性决定最终 gate；set-piece 可以自行
  学到较低 ROI 权重，shot/save 也能在局部证据可靠时执行 boost；
- local logits保留用于AP、校准和错误分析，但最终公式不会因local分数偏低自动压低global。

这与 EqFace 的“显式质量参与分类贡献”思路相近，但这里的质量是按帧、按事件类别的 ROI usefulness，
不是图像清晰度或统一样本质量分数。

D2 默认从 D1 best 初始化：

```bash
bash scripts/run_football_da_16f.sh d2_feature_quality_debug 0,1
bash scripts/run_football_da_16f.sh d2_feature_quality 0,1
bash scripts/run_football_da_16f.sh d2_feature_quality_eval 0
```

正式 D2 若找不到 D1 best 会直接报错；debug 模式可回退到 E1 init，仅用于验证代码，不作为正式消融结果。

### P5. 时序头

配置对齐后按顺序：

1. E1 frame auxiliary baseline
2. E2 attention pooling
3. E2 未解决 pooling 时再做 E3 class query
4. E4 从最强 E1 checkpoint 初始化

### P6. Hard negative

架构和阈值稳定后，再比较随机负样本、shot/save hard negative、人工审查后的 all-label hard
negative。验证集始终关闭 hard negative。

## 16. 运行命令

### 16.1 Exp4 baseline

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_best.pt \
bash scripts/run_football_da_16f.sh debug 0

INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_best.pt \
bash scripts/run_football_da_16f.sh all 0,1

bash scripts/run_football_da_16f.sh train 0,1
bash scripts/run_football_da_16f.sh eval 0
```

### 16.2 Dual E1-E4

```bash
ACTION_INIT_CHECKPOINT=/path/to/init.pt \
bash scripts/run_football_da_16f.sh e1 0,1

ACTION_INIT_CHECKPOINT=/path/to/e1_best.pt \
bash scripts/run_football_da_16f.sh e2 0,1

ACTION_INIT_CHECKPOINT=/path/to/e1_best.pt \
bash scripts/run_football_da_16f.sh e3 0,1

ACTION_INIT_CHECKPOINT=/path/to/e1_best.pt \
bash scripts/run_football_da_16f.sh e4 0,1
```

Smoke test 使用 `e1_debug` 到 `e4_debug`；PointNMS 评测使用 `e1_eval` 到 `e4_eval`。

### 16.3 Single-view E1-E4

```bash
LORA_R5_F32_INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_last.pt \
bash scripts/run_football_da_16f.sh e1_lora_r5_f32 0,1

bash scripts/run_football_da_16f.sh e2_lora_r5_f32 0,1
bash scripts/run_football_da_16f.sh e3_lora_r5_f32 0,1
bash scripts/run_football_da_16f.sh e4_lora_r5_f32 0,1
```

模式后添加 `_debug` 或 `_eval` 可进行 smoke test 或评测。

### 16.4 ROI branch 分析

```bash
python scripts/analyze_roi_branch_from_eval.py \
  --run-dir outputs/football_eval_runs/RUN_NAME \
  --output-prefix roi_branch_window_overlap \
  --postprocess window_overlap \
  --thresholds shot=0.2,save=0.45,set_piece=0.45 \
  --match-tolerance-sec 5
```

输入 `window_predictions.csv` 必须包含 `global_prob_*`、`local_prob_*`、`prob_*` 和
`roi_gate_*`。

## 17. 资源与训练注意事项

新实验优先使用单卡参数，训练程序根据 `gpu_ids` 数量解析全局参数：

```yaml
data:
  num_workers_per_gpu: 4
train:
  per_gpu_batch_size: 2
  grad_accum_steps: 4
  lr_per_gpu: 0.00005
  backbone_lr_per_gpu: 0.00001
eval:
  per_gpu_batch_size: 1
```

```text
world_size = len(gpu_ids)  # CPU 或未配置多卡时为 1
global_batch = per_gpu_batch_size * world_size
global_lr = lr_per_gpu * world_size
global_backbone_lr = backbone_lr_per_gpu * world_size
effective_batch = global_batch * grad_accum_steps
```

启动时会输出 `runtime_topology`，保存到 checkpoint/config 的也是解析后的全局值和
`runtime_topology`。历史配置仍可只写 `train.batch_size/lr/backbone_lr`，此时继续按全局值处理。
同一字段组不要混用；存在 `per_gpu_*` 时它优先并覆盖对应全局字段。

降低 accumulation 会改变更新频率和 effective batch；它本身不能证明或解决欠拟合。
如果 `resume.load_optimizer=true`，checkpoint optimizer 中的 LR 可能覆盖动态 LR；改变 GPU 数量的新实验应使用
init checkpoint 且 `resume=false`。

Dual 大约执行两次 DINO forward。LoRA 训练还要保存 trainable path activation，因此冻结 dual
的 batch 设置不能直接用于 LoRA。

遇到 `CUBLAS_STATUS_ALLOC_FAILED`：

- 检查剩余显存和残留进程
- 降低 global batch
- 用 accumulation 保持 effective batch
- 避免训练 GPU 同时运行评测

`data_time` 和 `step_time` 是 running average。性能分析应看 worker 和 video cache 预热后的
steady-state step。

## 18. 实验可复现清单

比较任何两个实验前记录：

- checkpoint 路径和 epoch
- 精确 train/val video IDs
- index 可用和缺失数量
- annotation root
- global/local 分辨率
- clip 时长、帧数、jitter
- random/hard-negative 比例
- init 还是 resume
- 冻结和可训练参数组
- head LR 与 backbone/LoRA LR
- global batch 与 accumulation
- threshold 来源
- post-processing 模式
- NMS radius 与 match tolerance
- per-video 指标，而不只 aggregate

同时改变多个维度、又没有显式 control 的实验，不能作为因果消融结论。

## 19. 当前实验矩阵

本节用于汇总当前真正需要比较的实验。所有正式实验默认使用：

```text
split: configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos
train/val: 127 / 35 videos
labels: shot, save, set_piece
clip: 10s
negative ratio: train=5, val=2
main eval: PointNMS + window-overlap, both with checkpoint tuned thresholds
standard long-video subset: 6 videos listed in section 12.5
```

### 19.1 Single-view 主线

| 名称 | Config / script | 输入 | 时序头 | 训练点 | 目的 |
| --- | --- | --- | --- | --- | --- |
| `lora_r5_last` | historical checkpoint | full 512x896, 16f | CLS Transformer | LoRA r5/f32 | 最早强基线 |
| `E1 single HR` | `configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det.yaml` | full 640x1120 or configured HR, 16f | CLS Transformer + frame head | LoRA | 当前更强 single baseline，frame detection 有效 |
| `E2 single` | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e2_attn_pool.yaml` | full, 16f | attention pooling | LoRA | 检查 CLS pooling 是否限制 precision |
| `E3 single` | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e3_class_query.yaml` | full, 16f | per-class query | LoRA | 检查不同类别是否应独立关注时间 |
| `E4 single` | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e4_event_topk.yaml` | full, 32 candidate -> 16 selected | event top-k Transformer | LoRA + frame head | 验证 frame eventness 是否能改善最终分类 |
| `E5 event-anchor` | `configs/football/dinov3_vitl16_lora_r5_f32_16f_e5_event_anchor.yaml` | full, anchor neighborhood | event-anchor Transformer | LoRA + frame head | 比 hard top-k 更保守地利用 eventness |

关键结论：single-view E1 的基础模型和 frame head 已经证明有价值。后续 ROI 实验应尽量从这个
checkpoint 初始化，避免和旧 `lora_r5_last` 的 split、分辨率、frame supervision 混杂。

### 19.2 Dual ROI 主线

| 名称 | Config / script | ROI | Fusion | Backbone/LoRA | 目的 | 当前判断 |
| --- | --- | --- | --- | --- | --- | --- |
| `Exp4 dual` | `configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml` | fixed 10s ROI | convex `dual_gate` | global frozen | 早期 robust ROI baseline | precision 有增益，但 shot/save recall 下降 |
| `D0 fixed ROI` | `configs/football/dinov3_vitl16_robust_dual_16f_e1_fixed_roi_d0.yaml` | fixed 10s ROI | `dual_gate` | frozen/reference | 固化 fixed ROI control | 只作为 control，不作为最终方向 |
| `D1 dynamic ROI` | `configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_d1.yaml` | per-frame dynamic ROI | `dual_gate` | LoRA adaptation | 验证动态 crop 是否优于固定 crop | loss/收益不稳定，需依赖可视化检查 ROI 覆盖 |
| `D2 feature quality` | `configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2.yaml` | dynamic ROI | feature-quality residual | controlled/frozen variant | 学习 ROI 是否有用，而不是硬用 confidence | 已实现，适合继续作 ablation |
| `D3 dynamic ROI LoRA` | `configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3.yaml` | dynamic ROI | `dual_gate` | LoRA | 直接双流 LoRA 微调 | loss 基本不下降，已停止本地实验 |
| `D4 decoupled LoRA` | `configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4.yaml` | dynamic ROI | `dual_gate` | separate local backbone | 降低 full/ROI 共享 LoRA 冲突 | 显存小但收益不明显，远端重跑需谨慎 |
| `D5 veto rank` | `scripts/run_football_roi_veto_rank_d5.sh` | dynamic ROI | residual-veto `dual_verifier` | local/fusion oriented | 用 hard FP 训练 ROI 只抑制 shot/save FP | 待补充分支分析，不建议单独押注 |
| `D6A quality joint` | `scripts/run_football_roi_quality_joint_d6a.sh` | dynamic ROI | `dual_feature_quality` | global/local LoRA joint | 让 full 与 ROI 都可参与学习，ROI 通过 quality/residual 补充 | 正在/已尝试，首 epoch loss 慢不一定异常 |
| `D6B quality guarded` | 待进行，见 20.1 | dynamic ROI | guarded `dual_feature_quality` | freeze global reference, train local/fusion | 保留 E1 global，不让 ROI 破坏 recall | 下一步推荐 control |
| `D7 dual-token fusion` | `scripts/run_football_dual_token_fusion_d7.sh` | dynamic ROI | mixed tokens + class query residual | separate local backbone | 让 Transformer 自主融合 full/ROI token | 待进行，必须先 aligned 抽帧控制变量 |

### 19.3 Hard negative / ranking 主线

| 名称 | Config / script | 技术点 | 当前结论 |
| --- | --- | --- | --- |
| `hardneg v1` | `scripts/run_football_weekend_a800.sh hardneg` | precision60 FP manifest，shot/save 为主 | hard sample 数量偏少时 precision 提升有限 |
| `hardneg v2 temporal` | `scripts/run_football_hardneg_v2_temporal.sh` | reviewed mid-score FP，包含 set-piece 审查版本 | 需要和 no-hardneg control 同 init 同步比较 |
| `D5 rank` | `scripts/run_football_roi_veto_rank_d5.sh rank` | hard FP + pairwise rank margin | ranking 只能拉开已有可分性，不能替代更强特征 |

Hard negative 的合理性要求：候选必须远离任意真实事件，不能只远离同类事件；同时保留“视觉像事件但不是事件”的困难样本，不能因为阈值或 rejection 过松把真正困难 FP 漏掉。

## 20. 待进行实验

### 20.1 D6B：global-preserving ROI quality residual

目的：验证 ROI 是否可以作为“细节证据补充分支”，同时严格保护当前最强 E1 global 输出。
D6A 让 global/local 都参与训练，可能仍会出现 full 与 ROI 特征学习互相牵制；D6B 则把 global
作为 reference，主要训练 local ROI backbone/adapter/quality/residual。

建议定义：

```text
init: /mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt
base config: configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4.yaml
view_fusion: dual_feature_quality
separate_local_backbone: true
global branch: frozen reference
local branch: trainable LoRA
fusion head: trainable
final: global_logits + quality * bounded residual
hard negative: disabled in first run
positive retention: enabled, branch=fused
frame_det_loss: enabled for global/local, but local frame loss should respect ROI valid mask
```

推荐命令模板：

```bash
OUTPUT_DIR=outputs/football_events/vitl16_e1_dynamic_roi_quality_guarded_d6b \
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=10 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
TARGET_HEAD_LR=0.0001 \
TARGET_GLOBAL_BACKBONE_LR=0.0 \
TARGET_LOCAL_BACKBONE_LR=0.00002 \
EPOCHS=5 \
bash scripts/run_football_roi_quality_joint_d6a.sh 0,1,2,3
```

如果现有 D6A 脚本没有显式覆盖 global freeze，可在命令中确认日志里的 optimizer groups：
`global_backbone_lr` 应为 `0.0`，global trainable 参数应为 0 或只保留非 backbone head。若代码当前
无法严格冻结 global 同时训练 local，需要新增一个 D6B launcher，而不是把 D6B 结果解释为受控实验。

D6B 验收指标：

- long-video window-overlap 下 `shot/save` precision 上升，recall 下降不超过 1-2pp；
- PointNMS 下 FP 数下降，尤其是门前推进、门将移动但无 save、禁区附近传球误报；
- `global hit -> fused miss` 数量不能明显增加；
- `roi_quality_prob` 在 ROI 失效或偏离动作区域时明显降低；
- `roi_residual_logits` 在 hard FP 上主要为负，在 true positive 上接近 0 或小幅正向。

### 20.2 D7-A：aligned dual-token class-query fusion

目的：不是复刻 E3，而是让 full/ROI token 在同一个时序融合器里交互，再由 class query 输出
受限 residual。D7 的初始 fused logits 严格等于 E1 global，避免训练一开始破坏已有模型。

技术点：

```text
16 global tokens + 16 ROI tokens, timestamps aligned
view embedding: 区分 global / ROI
time MLP: 使用真实 frame timestamps，不依赖离散 slot
invalid ROI mask: ROI 无效时不会参与 attention
ROI frame quality: 控制 local token 注入强度
class query: 每类独立从 mixed temporal tokens 聚合证据
bounded residual: positive <= 0.5, negative <= 2.0
final: global_logits + quality * bounded_residual
```

D7 与 E3 的区别：E3 是单流 class query 直接输出最终 logits；D7 是双流 mixed-token class query，
并且只输出 E1 global 之上的受限 residual。因此 E3 表现一般不能直接否定 D7。

推荐 4x A800 命令：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=10 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
NUM_WORKERS_PER_GPU=1 \
PREFETCH_FACTOR=1 \
bash scripts/run_football_dual_token_fusion_d7.sh 0,1,2,3
```

如果远端 `/dev/shm` 小，改为：

```bash
NUM_WORKERS_PER_GPU=0 \
bash scripts/run_football_dual_token_fusion_d7.sh 0,1,2,3
```

D7-A 必须保持 `video.dual_sampling=aligned`，full/ROI 使用同一组 16 帧。错开抽帧是后续 D8，
不能混入 D7，否则无法判断收益来自 fusion 结构还是额外时间信息。

### 20.3 D7-B：full-to-ROI cross-attention early fusion

目的：让 full-image token 主动读取 ROI token 的细节，而不是让 ROI 分支单独学习一套事件分类。
这个实验保留 E1 global 作为主路径，ROI 只通过 cross-attention 注入到 full token，再用同一个
frame/event head 输出最终结果。

技术点：

```text
view_fusion: dual_cross_attention
query: 16 full-image temporal tokens
key/value: 16 ROI temporal tokens
timestamps: full/ROI aligned，使用真实 frame time MLP
invalid ROI mask: 无效 ROI token 不参与 attention
residual gate: zero-init，初始 fused token 严格等于 global token
frame loss: 只监督 fused frame_event_logits，不再额外监督 local_frame_event_logits
clip loss: fused logits
disabled: local_loss_weight=0.0, roi_quality_loss_weight=0.0
```

推荐命令：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
bash scripts/run_football_dual_cross_attention_d7b.sh 2,3,4,6
```

D7-B 的预期收益：如果 ROI 的细节确实能补充 full-image，它应该比 D6A/D6B 更容易提升
`shot/save` precision，同时 recall 不明显下降。因为它不要求 ROI 自己独立判断事件，只要求
ROI 帮 full token 改善特征。

### 20.4 D7-C：class-query mixed-token early fusion

目的：把 full/ROI token 放在同一个 mixed temporal token 序列中，每个类别用独立 query
读取证据，直接学习类别相关的 ROI/full 互补关系。它比 D7-B 更强，但也更容易过拟合或学习不稳定。

技术点：

```text
view_fusion: dual_class_query_fusion
tokens: 16 global tokens + 16 ROI tokens
class query: shot/save/set_piece 各自独立聚合 mixed tokens
bounded residual: 在 E1 global logits 上学习受限修正
roi alpha: 使用 ROI availability/validity，不使用 learned roi_quality gate
frame loss: 只监督恢复到原 full-image 时间顺序的 fused global-view frame_event_logits
disabled: local_loss_weight=0.0, roi_quality_loss_weight=0.0
```

推荐命令：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
bash scripts/run_football_dual_class_query_fusion_d7c.sh 2,3,4,6
```

D7-C 和 E3 的区别：E3 是单流 class query 直接分类；D7-C 是 full/ROI mixed-token 早融合，
并且只在 E1 global 输出上做 residual 修正。因此它验证的是 ROI 细节和 full 全局语义是否能互补，
不是单纯验证 class-query 时序头是否更好。

### 20.8 Single-View Precision Focal Post-Eval

`neg8 + focal` 单流实验完成 checkpoint 后，用统一脚本评测 6 个长视频，并同时输出
`window_overlap` / `point_nms`、frame_detection oracle 分析、FP 距离分析和 vs E1 recall guard 对比。

```bash
# 评测 best.pt
SELECT_CHECKPOINT=best \
GPU_LIST=3,4,6,7 \
bash scripts/run_football_single_precision_focal_post_eval.sh eval

# 如果只想先看 epoch_1.pt
SELECT_CHECKPOINT=epoch_1 \
GPU_LIST=3 \
bash scripts/run_football_single_precision_focal_post_eval.sh eval

# 自动等待 best.pt 出现后再评测
SELECT_CHECKPOINT=best \
GPU_LIST=3,4,6,7 \
bash scripts/run_football_single_precision_focal_post_eval.sh wait_eval
```

默认输出目录：

```text
outputs/football_eval_runs/<train_output>_<checkpoint>_6videos_dense_checkpoint_thr/
  protocol_comparison_checkpoint_thr.json
  vs_e1_recall_guard_1pp.json
  frame_detection_precision/
  window_overlap_fp_distance.json
  summary/summary.md
```

## 21. 当前优先级

近期目标是提高 long-video precision，同时保持 recall。建议顺序：

1. 优先跑 D7-B full-to-ROI cross-attention early fusion，验证 ROI 细节能否作为 full token 的补充证据。
2. D7-B 完成后跑 D7-C class-query mixed-token early fusion，判断更强的类别独立聚合是否有额外收益。
3. 对 D7-B/D7-C 都用同一套 E1 init、aligned 抽帧、关闭 local_loss/roi_quality_loss，避免把收益归因搞混。
4. 如果 D7-B/D7-C 在 window-overlap 上 precision 有提升且 recall 不掉，再扩展到 hard negative rank 或错开抽帧 D8。
5. 暂缓单纯提高 ROI 分辨率；已有结果显示 recall 上升但 precision 下降，收益不够干净。
6. 暂缓复杂候选重打分校准器；等基础模型的 long-video P/R 更稳定后，再作为后处理增益使用。

## 22. Precision/Recall 平台期反思与下一阶段路线

### 22.1 当前证据

6 个长视频 window-overlap 口径下，当前关键结果大致为：

| run | micro precision | micro recall | micro f1 | 主要现象 |
|---|---:|---:|---:|---|
| E1 high-resolution frame-det | 0.354 | 0.803 | 0.491 | recall 高，但 FP 多，尤其 save |
| exp4 ROI dual-view | 0.397 | 0.736 | 0.516 | precision 最高，但 recall 明显下降 |
| D1 dynamic ROI | 0.342 | 0.809 | 0.481 | recall 保住，precision 未提升 |
| D6A ROI quality residual | 0.345 | 0.783 | 0.479 | ROI veto 有作用但不够强 |
| event-anchor sym20 | 0.366 | 0.720 | 0.485 | precision 小升，recall 被吃掉 |
| topk / OHEM / duration-rank | 0.26-0.30 | 0.71-0.75 | 0.39-0.42 | 没有解决线上 FP，部分还伤主模型 |

per-class 瓶颈：

```text
E1_hr:    shot P=0.399 R=0.884, save P=0.214 R=0.841, set_piece P=0.536 R=0.657
exp4_roi: shot P=0.422 R=0.856, save P=0.224 R=0.696, set_piece P=0.622 R=0.586
D1_roi:   shot P=0.407 R=0.863, save P=0.205 R=0.884, set_piece P=0.513 R=0.677
D6A:      shot P=0.396 R=0.863, save P=0.214 R=0.783, set_piece P=0.473 R=0.667
anchor:   shot P=0.423 R=0.808, save P=0.249 R=0.826, set_piece P=0.476 R=0.515
```

结论：当前最影响可用性的类别是 `save`。它保持高 recall 时 precision 只有 0.20-0.25，
说明模型把大量门将移动、禁区混战、门前推进误判成 save。`shot` 还有提升空间，`set_piece`
更依赖上下文和标注完整性，不应和 shot/save 使用同一套过滤策略。

### 22.2 这是不是模型进入平台期？

不是简单的模型容量平台期，更像是 **训练目标和线上目标不一致导致的应用平台期**。

证据：

- full-image/E1 已经能做到高 recall，说明 backbone 对粗粒度事件召回仍有效；
- frame_event/topk 对事件附近帧有定位能力，但简单乘法融合和 hard topk 没有稳定转化成 precision；
- ROI 提高分辨率、动态 ROI、质量 gate 都没有稳定改善，说明 ROI 信息存在但噪声和融合方式会抵消收益；
- hard negative 直接加入训练没有明显提升，说明当前 hard negative 的定义/权重/使用方式还没有打到真正的线上 FP 分布；
- D7-B 这种保守 cross-attention 没有变强，说明只把 ROI 当弱补充不够，需要更明确地让 ROI 学会“反证/细节验证”。

所以后续不要继续枚举大结构，而要围绕线上目标重构训练与评估：

```text
保持高 recall 主模型 -> 挖真实长视频 hard FP -> 学 precision-oriented verifier/veto/reranker -> 在 recall floor 下验收
```

### 22.3 下一阶段主线：类别条件化的 verifier/veto

目标不是替换 E1/D7 主模型，而是在高 recall 候选上减少 FP。

主模型输出：

```text
clip logits: shot/save/set_piece
frame_event logits: max/topk/peak shape
full temporal feature
ROI temporal feature
ROI validity/quality/attention
full-vs-ROI residual/disagreement
```

新 head 输出：

```text
veto_logits[label] 或 calibrated_score[label]
final_score[label] = clip_score[label] + bounded_residual[label]
```

训练目标：

- 正样本 retention：在 GT 附近窗口，final_score 不允许明显低于 global_score；
- hard FP suppression：对长视频挖出的 FP，final_score 应低于 global_score，并低于类别阈值；
- ranking：同一类别内，GT matched windows 的 score 必须高于 hard FP；
- 类别分治：shot/save 优先启用 ROI/frame 细节验证，set_piece 初期更保守，只做轻量校准。

验收标准必须用长视频：

```text
window-overlap: recall 不低于 E1/D7 baseline 1pp 以内，precision 提升 >= 3pp
PointNMS: FP/event 减少，recall 不低于 baseline 2pp 以内
save: precision 必须单独提升，不接受只靠 set_piece 改善 macro
shot: precision 提升同时 recall >= baseline - 1pp
set_piece: 不追求 aggressive filtering，优先不伤 recall
```

### 22.4 推荐实验优先级

P0：固定高 recall teacher / candidate generator

- 使用当前 E1 high-resolution frame-det 或原始 D7 best/epoch6 作为 candidate generator；
- 先不要再大幅改 backbone；
- 为每个长视频保存 dense candidates、frame_event statistics、ROI statistics、matched/unmatched 标记。

P1：离线 verifier/reranker 小模型

- 从长视频候选构建训练集：TP、near-miss、hard FP、easy negative；
- 输入使用已有模型输出和轻量特征，不重新解码视频；
- 先训练 logistic/MLP calibrator，快速验证是否能在 recall floor 下提高 precision；
- 如果有效，再把 verifier head 合并回训练脚本端到端微调。

P2：ROI residual/veto head，而不是 ROI 独立分类

- full-image 继续给主判断；
- ROI 负责细节验证和反证：有没有射门动作、守门员扑救动作、球门前真实对抗；
- ROI 无效或偏离时，默认回退 global，不允许 ROI 随机创造高分。

P3：hard FP mining 重做

- mining 不能只按普通 FP 保存，要按类别和 FP 类型分桶；
- save 的 hard FP 单独建桶：门将普通移动、禁区拥挤、球门附近传中、球出界后门将动作；
- shot 的 hard FP 单独建桶：禁区推进、传球、长传、门前混战但无射门；
- set_piece 单独处理，避免未标注/标注偏移导致误挖。

P4：结构实验降级

- D7-C 可以继续看，但不作为唯一希望；
- D7-B 已证明弱 cross-attention 不够；
- event_topk/event_anchor 只有在不伤 recall 时才继续，否则暂停；
- 单纯提高分辨率、继续加 epoch、盲目 hard negative 都不是当前最高 ROI 的方向。

### 22.5 最近两天建议执行的实验

1. 先完成 D7-C 标准版评估，作为 early fusion 是否有希望的最后一次结构验证。
2. 同时开发 candidate export：把 E1/D7 在训练/val 长视频上的候选、TP/FP、frame_event、ROI stats 全部导出成表。
3. 训练 `V1 verifier`：只用候选级统计特征，按类别输出校准分数；目标是在 recall floor 下提升 precision。
4. 如果 V1 有效，再做 `V2 feature verifier`：加入 full/ROI temporal pooled feature 和 class-query attention feature。
5. 最后才考虑端到端，把 V2 head 接回模型，用 fixed teacher score 做 retention。

这条路线的关键是：先不要再赌大模型结构自然学会减少 FP，而是把线上 FP 显式变成训练对象。

### 22.6 候选级 verifier 数据导出

为了后续在不重跑 DINO 推理的情况下验证候选重打分/二阶段过滤，先把长视频评测结果整理成
`(video_id, window_index, label)` 级别的数据集：

```bash
python scripts/build_football_candidate_verifier_dataset.py \
  --eval-run-dir outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr \
  --output-csv outputs/football_verifier_datasets/e1_hr_6videos_candidates.csv \
  --output-summary outputs/football_verifier_datasets/e1_hr_6videos_candidates.summary.json \
  --candidate-thresholds shot=0.10,save=0.10,set_piece=0.10 \
  --match-tolerance-sec 2.0
```

导出字段包括：

- clip-level: `clip_prob`, `prob_shot/save/set_piece`, `margin_to_max_other`;
- frame-level: `global_frame_max_prob`, `frame_prob_top2_mean`, `frame_prob_top4_mean`, `frame_peak_relative_time_sec`;
- ROI-level: `roi_valid`, `roi_confidence`, `roi_frame_quality_mean`, `crop_area_ratio`, `crop_goal_count/ball_count/person_count`;
- dual-view 若存在：`global_prob_*`, `local_prob_*`, `roi_gate_*`, `roi_quality_prob_*`, `local_minus_global_prob_*`;
- target: `is_gt_window`, `is_candidate`, `is_tp_candidate`, `is_hard_fp_candidate`, `nearest_gt_distance_sec`。

这里的 `is_gt_window` 使用 window-overlap 口径：只要 GT 时间点落在 `window +/- tolerance` 内，就认为该
window-label 是正例。这样不会把一个真实事件附近多个覆盖窗口误当 hard negative。候选阈值只控制导出的
低分背景数量，正样本窗口会始终保留。

后续 V1 verifier 可以先从这个 CSV 上训练一个很小的 per-class 校准器，验收方式固定为：

```text
在 recall 不低于 E1 high-recall baseline 的前提下，比较 long-video precision 是否提升。
```

### 22.7 V1 候选 verifier 诊断

已新增脚本：

```bash
python scripts/fit_football_candidate_verifier_from_csv.py \
  --candidate-csv outputs/football_verifier_datasets/e1_hr_6videos_candidates.csv \
  --output-dir outputs/football_candidate_verifiers/e1_hr_6videos_csv_oof \
  --labels shot,save,set_piece \
  --recall-drop-tolerance 0.01 \
  --folds 3 \
  --regularization-c 0.1
```

注意：自动特征选择会排除任何包含 `target` 的列，例如 `global_frame_target`，否则会发生
GT-derived target leakage，得到虚假的高 precision。

E1 high-resolution frame-det 的 6 视频 grouped OOF 诊断结果：

| metric | precision | recall | tp | fp | fn |
|---|---:|---:|---:|---:|---:|
| baseline candidate threshold 0.10 | 0.2166 | 0.6795 | 583 | 2108 | 275 |
| V1 verifier OOF | 0.4378 | 0.6772 | 581 | 746 | 277 |

per-class：

```text
shot:      baseline P/R=0.3378/0.6354 -> verifier P/R=0.6163/0.6304
save:      baseline P/R=0.1673/0.6684 -> verifier P/R=0.3553/0.6631
set_piece: baseline P/R=0.1724/0.7500 -> verifier P/R=0.3624/0.7536
```

这个结果不是最终上线结论，因为它仍然是 6 个 val 长视频上的候选级 OOF 诊断，不是独立 holdout。
但它证明了一点：模型并不是完全进入平台期，已有的 clip/frame/ROI 统计特征里存在可以分离 hard FP 的信号。
后续应该把这个 verifier 思路扩展到 29-video calibration / 6-video holdout，并把 ROI/full temporal feature
接入 V2 verifier，而不是继续只看训练 loss 或单纯堆结构。

### 22.8 主模型 precision-focused 训练实验

二阶段 verifier 不替代主模型优化。为了直接提升 DINO 主模型的 precision，新增一个受控实验：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_last.pt \
PER_GPU_BATCH_SIZE=8 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
bash scripts/run_football_single_precision_focal.sh 0,1
```

对应配置：

```text
configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det_precision_focal.yaml
```

相对 E1 high-resolution frame-det，只改以下变量：

```text
train negative_ratio: 5 -> 8
val negative_ratio: keep 2
clip_loss_type: bce -> focal_bce
clip_focal_gamma: 2.0
clip_focal_alpha: shot=0.35, save=0.40, set_piece=0.45
frame_detection: unchanged
view_fusion: single, no ROI branch
```

这个实验的目标不是单纯让训练 loss 下降，而是让 clip-level logits 对 hard FP 更敏感。验收必须看
long-video window-overlap 和 PointNMS：

```text
recall >= E1 baseline - 1pp
precision 尤其 shot/save 上升
如果 recall 明显下降，说明负样本/focal 太强，需要降低 negative_ratio 或 gamma
```

#### 22.8.1 PointNMS clean clip-ranking hard sample

`negative_ratio=8 + focal` 的第一个 epoch 长视频结果没有达到目标：在 6 个测试视频上，
PointNMS 与 window-overlap 都没有带来 precision 增益，并且 recall 明显下降。因此如果后续
epoch 仍不能通过 recall guard，不建议继续单纯加 epoch。

下一步更干净的 hard-sample 实验改为：

```text
E1 high-res frame-det checkpoint
-> 在训练长视频上按最终 PointNMS 输出挖 FP
-> 丢弃已匹配 GT 和 GT safety margin 附近预测
-> 只对 shot/save 的 hard FP 加 clip-level pairwise ranking
-> hard negative BCE 不额外加权，避免整体压低分数伤 recall
-> safety margin=10s，保护人工标注附近的正样本/近正样本
-> 关闭 frame_rank / online hard negative
-> LoRA/backbone adaptation 打开，而不是只训练时序头
```

新增入口：

```text
scripts/build_pointnms_hard_negatives.py
scripts/run_football_clip_rank_clean.sh
scripts/run_football_clip_rank_lora_last8.sh
configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det_clip_rank_clean.yaml
configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det_clip_rank_lora_last8.yaml
```

本机 5090 两卡参考命令：

```bash
RUN_MINING=1 \
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
OUTPUT_DIR=outputs/football_events/vitl16_e1_clip_rank_clean_lora \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
TARGET_HEAD_LR=0.0003 \
TARGET_BACKBONE_LR=0.00003 \
bash scripts/run_football_clip_rank_clean.sh all 2,3
```

A800 四卡参考命令：

```bash
RUN_MINING=1 \
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
OUTPUT_DIR=outputs/football_events/vitl16_e1_clip_rank_clean_lora \
PER_GPU_BATCH_SIZE=8 \
TARGET_EFFECTIVE_BATCH_SIZE=160 \
TARGET_HEAD_LR=0.0004 \
TARGET_BACKBONE_LR=0.00004 \
bash scripts/run_football_clip_rank_clean.sh all 0,1,2,3
```

验收标准：

```text
PointNMS/window-overlap 都看；
recall 不低于 E1 baseline 1pp 以上；
shot/save precision 至少提升 2-3pp；
set_piece 不作为第一版 hard-rank 目标，避免引入定位定义更长的噪声。
```

#### 22.8.2 LoRA last8 capacity check

如果 clean clip-ranking last4 仍然没有明显提升，下一步不是立刻全量解冻 DINO，而是先扩大
LoRA 覆盖范围：

```text
base: E1 high-res frame-det
input: single full-image 640x1120, 16f
LoRA: rank=8, target_last_blocks=8, qkv/proj, train_norm=true
loss: BCE clip loss + frame heatmap/MIL + shot/save hard-FP clip ranking
disabled: focal loss, frame rank, online hard negative
hard negative: clean PointNMS FP, safety margin=10s
```

这个实验回答的问题是：当前 precision 平台期是否来自 DINO 后层可学习容量不足。如果 last8
相比 last4 clean-rank 能提升 long-video precision 且 recall 不掉，说明继续扩大 LoRA 覆盖或加入
少量 MLP LoRA 有价值；如果 last8 仍无收益，就不要继续盲目放开更多 DINO 参数，应转向二阶段
verifier/reranker。

本机 5090 参考命令：

```bash
RUN_MINING=0 \
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
TARGET_HEAD_LR=0.0003 \
TARGET_BACKBONE_LR=0.00003 \
bash scripts/run_football_clip_rank_lora_last8.sh all 2,3
```

A800 四卡参考命令：

```bash
RUN_MINING=0 \
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=8 \
TARGET_EFFECTIVE_BATCH_SIZE=160 \
TARGET_HEAD_LR=0.0004 \
TARGET_BACKBONE_LR=0.00004 \
bash scripts/run_football_clip_rank_lora_last8.sh all 0,1,2,3
```

注意：`RUN_MINING=0` 的前提是
`outputs/football_hard_negatives/pointnms_fp_clean_shot_save.json` 已经生成。否则先执行：

```bash
MINE_FORCE=1 \
MINING_BATCH_SIZE=6 \
MINING_NUM_WORKERS=4 \
HARDNEG_SAFETY_MARGIN_SEC=10 \
bash scripts/run_football_clip_rank_clean.sh mine_build 2
```

### 22.9 ROI 价值为什么还没有挖出来

当前 ROI 实验没有稳定收益，不代表 ROI 没价值，更可能是使用方式有问题：

- ROI 分支如果被迫独立分类，会在 ROI 偏移/失效时学习困难；
- full-image 和 ROI 共用或同时更新 LoRA 时，可能出现梯度目标冲突；
- convex gate/直接加权会让 ROI 变成“另一个分类器”，而不是细节证据；
- ROI 对 shot/save 应该提供动作细节和反证，对 set_piece 则更多是上下文补充，两者不该同策略。

后续 ROI 应按这个方向继续：

```text
full-image: 保持高 recall 主判断
ROI: 提供细节 token / 局部动作证据 / hard FP 反证
fusion: 早融合或 verifier residual，不再要求 ROI 单独承担最终分类
training: 加正样本 retention + hard FP suppression/ranking，避免 ROI 伤 recall
```

也就是说，ROI 下一步不应该再做“单独 local_loss 越大越好”的实验，而应做：

1. full/ROI token early fusion，最终一个分类头；
2. ROI residual/veto 只在有充分证据时修正 full logits；
3. 用长视频 hard FP 明确监督 ROI 学会反证，而不是只靠 clip BCE 自己摸索。

ROI early-fusion + precision focal 对照命令：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
OUTPUT_DIR=outputs/football_events/vitl16_e1_dual_class_query_fusion_d7c_precision_focal \
NEGATIVE_RATIO_TRAIN=8.0 \
NEGATIVE_RATIO_VAL=2.0 \
CLIP_LOSS_TYPE=focal_bce \
CLIP_FOCAL_GAMMA=2.0 \
CLIP_FOCAL_ALPHA=shot=0.35,save=0.40,set_piece=0.45 \
VIEW_FUSION_POSITIVE_DELTA_PER_CLASS=shot=0.10,save=0.50,set_piece=0.50 \
VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS=shot=0.50,save=2.00,set_piece=1.50 \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
bash scripts/run_football_dual_class_query_fusion_d7c.sh 0,1,2,3
```

如果要更保守地只让 full tokens 读取 ROI 细节，而不是 mixed-token class query：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
OUTPUT_DIR=outputs/football_events/vitl16_e1_dual_cross_attention_d7b_precision_focal \
NEGATIVE_RATIO_TRAIN=8.0 \
CLIP_LOSS_TYPE=focal_bce \
VIEW_FUSION_POSITIVE_DELTA_PER_CLASS=shot=0.10,save=0.50,set_piece=0.50 \
VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS=shot=0.50,save=2.00,set_piece=1.50 \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
bash scripts/run_football_dual_cross_attention_d7b.sh 0,1,2,3
```

解释：先跑 single precision focal 是为了验证主模型本身是否能学到更强负例边界；再跑 ROI early-fusion
precision focal，是为了验证 ROI 在同样 loss/采样目标下是否提供额外信息。只有当 ROI 版本相比 single
版本在 long-video precision 上有增益且 recall 不明显下降，才能说明 ROI 价值被真正挖出来。

### 22.10 ROI/Frame 特征组消融诊断

新增脚本：

```bash
python scripts/analyze_football_candidate_verifier_features.py \
  --candidate-csv outputs/football_verifier_datasets/e1_hr_6videos_candidates.csv \
  --output outputs/football_candidate_verifiers/e1_hr_6videos_feature_ablation.json \
  --labels shot,save,set_piece \
  --groups clip,clip_frame,clip_roi,clip_frame_roi

python scripts/analyze_football_candidate_verifier_features.py \
  --candidate-csv outputs/football_verifier_datasets/exp4_roi_dual_6videos_candidates.csv \
  --output outputs/football_candidate_verifiers/exp4_roi_dual_6videos_feature_ablation.json \
  --labels shot,save,set_piece \
  --groups clip,clip_frame,clip_roi,clip_frame_roi
```

E1 single 候选表的 6-video OOF：

| features | micro precision | micro recall | 结论 |
|---|---:|---:|---|
| clip | 0.4829 | 0.6737 | clip 分数本身已有强可校准信号 |
| clip+frame | 0.4378 | 0.6772 | 简单 frame max/topk 统计反而加噪 |
| clip+ROI | 0.4829 | 0.6737 | E1 无 ROI，等价于 clip |
| clip+frame+ROI | 0.4378 | 0.6772 | 同 clip+frame |

ROI dual 候选表的 6-video OOF：

| features | micro precision | micro recall | 结论 |
|---|---:|---:|---|
| clip | 0.4804 | 0.6270 | dual 模型自己的 clip score 已有可校准性 |
| clip+ROI/dual | 0.5100 | 0.6270 | ROI/dual 字段有边际增益 |

per-class ROI/dual 增益：

```text
shot:      clip P/R=0.5244/0.6810 -> clip+ROI P/R=0.4906/0.6633  # 下降
save:      clip P/R=0.2875/0.6257 -> clip+ROI P/R=0.3587/0.6043  # precision 明显提升
set_piece: clip P/R=0.7600/0.5507 -> clip+ROI P/R=0.7913/0.5906  # precision/recall 都提升
```

这个诊断说明：ROI 不是没有价值，但它不是对所有类别都同向有益。当前 ROI 信息更适合作为
`save/set_piece` 的 verifier/context，而 `shot` 的 ROI 融合需要更谨慎，可能要降低 ROI residual scale，
或者只在 ROI 质量高、球/人群/球门关系明确时启用。

因此下一步 ROI 训练不要再使用统一 gate。推荐改成类别条件化策略：

```text
shot:      full-image 主导，ROI 只在高质量时做小幅正/负 residual
save:      ROI 重点参与，学习守门员动作/门前反证，允许更强 negative residual
set_piece: ROI/goal/context 可参与，但要防止未标注导致过筛
```

代码已支持：

```text
model.view_fusion_positive_delta_per_class
model.view_fusion_negative_delta_per_class
```

推荐初值：`positive=shot=0.10,save=0.50,set_piece=0.50`，
`negative=shot=0.50,save=2.00,set_piece=1.50`。含义是限制 ROI 对 shot 的改写，
但允许 ROI 对 save hard FP 做更强负向 residual。

同时，frame detection 的简单 max/topk 统计不宜直接作为最终融合公式。它仍可用于定位解释、top-k 抽帧、
候选窗口构造，但如果要提高 precision，需要用更丰富的 temporal feature 或 verifier 学习，而不是简单拼
`frame_max_prob`。

### D7D direct class-query early fusion

目的：验证 ROI 特征是否能真正参与最终事件分类，而不是只对 global logits 做保守 residual 修正。

和 D7C 的关键区别：

- D7C: `full tokens + ROI tokens -> class-query features -> bounded residual`, 最终 `logits = global_logits + alpha * residual`。
- D7D anchored: `full tokens + ROI tokens -> class-query features -> bounded delta`, 最终 `logits = global_logits + small_learned_gate * delta`。

D7D 最初版本直接用随机初始化的 fused head 覆盖 `global_logits`，第一个 log 里出现了较大的 `roi_veto_delta` 和 `positive_retention_loss`，说明启动阶段会破坏 E1 global baseline。现在改为 anchored residual：direct head 最后一层零初始化，`view_fusion_direct_delta_gate_init=0.05`，训练开始时输出等价于 global baseline，ROI/fusion 只从小幅修正开始学习。

默认控制变量：

- `model.view_fusion=dual_direct_class_query_fusion`
- `model.freeze_global_branch=true`
- `model.separate_local_backbone=true`
- `video.dual_sampling=aligned`
- `train.local_loss_weight=0.0`
- `train.roi_quality_loss_weight=0.0`
- `model.view_fusion_direct_delta_gate_init=0.05`
- `model.view_fusion_positive_delta=0.5`
- `model.view_fusion_negative_delta=2.0`
- `train.positive_retention_loss_weight=0.1`
- `train.frame_det_loss_weight=0.3`
- `spatial_crop.global_image_size=[640,1120]`

启动命令：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
  bash scripts/run_football_dual_direct_class_query_fusion_d7d.sh 2,3,4,6
```

观察重点：

- `roi_veto_delta = global_logits - logits` 初始应接近 0，而不是一次跳到绝对值 3 左右。
- `roi_gate` 初始应约等于 `roi_frame_quality_mean * 0.05`，如果后续稳定增大，说明 fusion 在主动使用 ROI。
- `positive_retention_loss` 应显著低于直接覆盖版，避免正样本 recall 启动即被破坏。
- tuned PointNMS 下 precision 是否提升，同时 shot/save recall 不明显下降。



## 24. Feature-First Experiment Line: A/B/C

当前 hard-negative、ranking、时序头实验说明：模型可以被训练得更保守，但还没有稳定学到区分真事件和相似负样本的视觉证据。因此新增三组 feature-first 实验，先验证 backbone adaptation、时间采样密度和 ROI 局部证据是否能增强特征本身。

### A. Strong Single-View LoRA 16f HR

目的：在不引入 ROI、不引入 hard-negative/ranking 的情况下，验证更强 DINO LoRA adapter 是否能拉开 TP/FP 特征。

关键设置：

```text
config: configs/football/dinov3_vitl16_strong_lora12_mlp_16f_hr_e1.yaml
input: full-image 640x1120, 16 frames
init: /mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt
LoRA: rank=8, last12 blocks, attn.qkv, attn.proj, mlp.fc1, mlp.fc2, train_norm=true
hard negative/ranking: disabled
frame detection: heatmap + MIL enabled, frame rank disabled
```

本机 5090 推荐命令：

```bash
bash scripts/run_football_strong_lora12_mlp_16f.sh 3,4,6,7
```

### B. Strong Single-View LoRA 24f 512x896

目的：控制空间分辨率成本，验证更多采样帧是否比更高分辨率更有效，尤其针对 1-2 秒内发生的 shot/save。

关键设置：

```text
config: configs/football/dinov3_vitl16_strong_lora12_mlp_24f_512x896_e1.yaml
input: full-image 512x896, 24 frames
LoRA: same as A
hard negative/ranking: disabled
```

远端 A800 推荐命令：

```bash
PER_GPU_BATCH_SIZE=8 TARGET_EFFECTIVE_BATCH_SIZE=128   bash scripts/run_football_strong_lora12_mlp_24f.sh 0,1,2,3
```

如果显存吃紧，把 `PER_GPU_BATCH_SIZE` 降到 6 或 4。

### C/D8. ROI Evidence Fusion

目的：ROI 不再作为独立分类分支，而是作为局部细节证据注入 full-image 表征；global branch 冻住保护 baseline recall，local ROI LoRA + fusion 学习何时、如何修正 global。

关键设置：

```text
config: configs/football/dinov3_vitl16_d8_roi_evidence_fusion_from_strong_lora.yaml
fusion: dual_cross_attention
input: global 640x1120 + ROI 384x640, aligned 16 frames
global branch: frozen reference
local ROI branch: trainable LoRA
local_loss_weight: 0
positive_retention_loss_weight: 0.5
hard negative/ranking: disabled
```

推荐等 A 产出 best 后启动；脚本默认优先使用 A 的 best，不存在时回退到 E1 last：

```bash
INIT_CHECKPOINT=outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt PER_GPU_BATCH_SIZE=4 TARGET_EFFECTIVE_BATCH_SIZE=96   bash scripts/run_football_roi_evidence_fusion_d8.sh 0,1,2,3
```

验收不只看 val F1，必须看 6 长视频 `window_overlap` 和 `point_nms`：shot/save precision 提升，recall 相对 E1 HR baseline 不下降超过 1-2pp，且 set_piece 不被明显误伤。

## Spatial Probe P1-P3 (2026-08-10)

This pipeline isolates whether frozen DINO patch tokens contain useful local event evidence before allowing that branch to modify the global classifier.

### P1: standalone spatial probe

- Config: `configs/football/dinov3_vitl16_spatial_probe_p1_16f_hr.yaml`
- Global reference: `vitl16_strong_lora12_mlp_16f_hr_e1/best.pt`
- Global DINO, LoRA, temporal head, and classifier are frozen.
- Global frame readout remains `last_cls_patch_mean`.
- Layers `[-8, -4, -1]` are used only by the zero-initialized residual multi-layer patch adapter.
- The final prediction is the independent `spatial_clip_logits` classifier.
- Clip BCE is balanced per class between positive and negative samples; frame MIL remains at weight `0.2`.
- Entropy, overlap, diversity, retention, and fusion losses are disabled.
- Checkpoint selection uses standalone spatial mAP.

```bash
bash scripts/run_football_spatial_probe_pipeline.sh p1 3,4,6,7
```

Each epoch records `metrics.spatial_branches.global`, `.spatial`, and `.fused`. P1 should proceed only if standalone shot/save AUROC reaches at least about `0.65` and both positive and negative losses improve.

### P2: frozen residual fusion

- Config: `configs/football/dinov3_vitl16_spatial_probe_p2_fusion_16f_hr.yaml`
- Initializes from P1 best.
- Freezes the global reference and the complete P1 spatial probe.
- Trains a per-class adaptive residual fusion head from global/spatial logits, frame evidence, attention entropy, and class embedding.
- Delta is zero initialized and gate starts at `0.15`, so the initial fused output exactly equals global output.

```bash
bash scripts/run_football_spatial_probe_pipeline.sh p2 3,4,6,7
```

### P3: low-LR joint fine-tuning

- Config: `configs/football/dinov3_vitl16_spatial_probe_p3_joint_16f_hr.yaml`
- Initializes from P2 best.
- Unfreezes the spatial branch and the existing last-12-block LoRA path.
- Uses global head LR `5e-5`, LoRA LR `2e-6`, standalone spatial retention loss `0.2`, frame MIL `0.1`, and weak attention regularization.

```bash
bash scripts/run_football_spatial_probe_pipeline.sh p3 3,4,6,7
```

Run all stages only after P1 has passed the separability check:

```bash
bash scripts/run_football_spatial_probe_pipeline.sh all 3,4,6,7
```

## Global-conditioned Spatial Residual / Raw Labels (2026-08-10)

This experiment starts from `vitl16_spatial_attn_residual_multilayer_16f_hr/epoch_3.pt` and tests whether the existing spatial representation can correct the frozen global classifier rather than classify events independently.

Configuration: `configs/football/dinov3_vitl16_global_conditioned_spatial_residual_raw_16f_hr.yaml`

- Annotations: `/home/new_users/qiuqi/code/football_events_raw`; all 127 train and 35 validation split videos have a raw JSON.
- Frozen reference: DINO/LoRA, multi-layer global readout, global temporal/head, and the complete epoch-3 spatial reader.
- Trainable path: a class-conditioned correction head that combines frozen global temporal features, class-specific spatial features, global confidence, loaded spatial residual/frame evidence, entropy, gate, and query-overlap quality.
- Initialization: the bounded correction is exactly zero, so initial fused logits equal epoch-3 global logits. Maximum ungated correction is `2.0` logits.
- Error definition uses the epoch-3 tuned thresholds: shot `0.2038660`, save `0.2679825`, set_piece `0.7164171` (recalibrated on raw validation).
- FP slots are pushed below threshold minus a logit margin; FN slots are pushed above threshold plus the margin. TP/TN guards and a small stability penalty protect correct reference decisions.
- The primary BCE has weight `0.25`; the targeted correction objective has weight `1.0`.

Run:

```bash
setsid bash scripts/run_football_global_conditioned_spatial_residual.sh 3,4,6,7 \
  >> outputs/football_events/vitl16_global_conditioned_spatial_residual_raw_16f_hr/launcher.log \
  2>&1 < /dev/null &
```

R1-A trains only the correction head. Continue with low-LR spatial-reader fine-tuning only if R1-A raises tuned precision while meeting the configured recall floors.

## Top-band Content-aware Crop / Sky Removal (Proposed, 2026-08-11)

### Motivation

For a `640x1120` ViT-L/16 input, the frame contains `40x70=2800` patch tokens. If the top `10%`, `15%`, or `20%` contains only sky, stands, or other irrelevant background, roughly `280`, `420`, or `560` tokens are spent on weak evidence. Cropping this band and resizing back to the configured input size increases the vertical scale of players and the ball by approximately `11%`, `18%`, or `25%`, without increasing backbone FLOPs.

This may improve distant-action representation and recall, but it is not expected to solve shot-like pass/back-pass false positives by itself. Those errors require stronger temporal and contextual discrimination.

### Recommended crop policy

Do not crop independently per frame and do not use the goal top as a hard boundary. Frame-level boundaries can jitter and create artificial motion, while goal misses or partial detections can remove the ball or players.

1. Estimate one stable crop ratio per video. Use a per-10-second estimate with temporal smoothing only when camera zoom changes substantially.
2. Build a robust on-pitch player set before estimating the crowd boundary. Keep stable tracks with valid confidence, box size/aspect ratio, and feet positions consistent with the pitch; reject spectators, coaches, isolated false detections, and short tracks.
3. For each sampled frame, estimate the upper crowd envelope with a confidence/track-stability weighted low quantile rather than the highest single box:

   `people_top_t = weighted_quantile(player_y1, q=0.10)`

   Aggregate `people_top_t` over time with another low quantile. Define `crop_y_people = people_top - margin`, where `margin = max(0.06 * image_height, 1.0-1.5 * median_player_height)`. This makes the margin scale with both resolution and camera zoom instead of using a fixed pixel count.
4. Aggregate trusted goal detections and compute a goal-relative candidate:

   `crop_y_goal = goal_y1 - k * goal_height`, where initial `k` is `1.0-1.5`.

5. Build an object-preservation guard from trusted ball, player, and goal boxes. The final boundary must stay above the low-percentile object top minus a `3%-5%` image-height margin.
6. When both goal and player evidence are available, use the conservative boundary `crop_y = min(crop_y_goal, crop_y_people, crop_y_object_guard)` and cap cropping at `18%`.
7. When the goal is not visible but the player set is reliable, fall back to `min(crop_y_people, crop_y_object_guard)` and use a stricter maximum crop of `12%`. If player evidence is also insufficient, do not crop.
8. Quantize the crop ratio to stable bins such as `[0%, 8%, 12%, 16%]` to prevent small detection fluctuations from changing the input distribution.

The crop is resized directly to `global_image_size`; letterboxing would preserve aspect ratio but would spend patch tokens on padding and undermine the purpose of this experiment.

### Risks and safeguards

- Aerial balls and high crosses can appear above the crossbar, so goal-only cropping is unsafe.
- Aggressive crops above `18%-20%` can remove global geometry and harm `set_piece` recognition.
- Per-frame crop changes create artificial vertical motion and must be prohibited.
- Crop statistics can become a goal-presence shortcut. The same policy must apply to positive and negative clips.
- Evaluate retained-object rates before training: retain at least `99.5%` of trusted ball boxes and `99%` of trusted player/goal boxes.

### Experiment sequence

1. **Crop audit:** render original/cropped pairs for `20-30` videos and at least `200` positive clips. Report crop-ratio distribution, retained-object rates, and boundary stability.
2. **S0:** current no-crop baseline.
3. **S1:** fixed top crop of `10%`, followed by normal fine-tuning. This isolates whether removing the top band alone helps.
4. **S2:** goal-relative per-video adaptive crop with player/ball preservation guard and an `18%` cap.
5. **S3:** goal-plus-crowd adaptive crop. Use the crowd-envelope fallback with a `12%` cap when the goal is absent.
6. Compare validation AP/mAP and six-video WindowOverlap/PointNMS metrics. Do not evaluate only by applying the crop to an old checkpoint because the resize changes the input distribution.

Continue only when shot/save AP improves by at least `0.5pp`, or long-video recall improves by at least `1pp` without reducing precision, while set-piece changes by no worse than `-1pp`.

### Checkpoint and controlled training configuration

The default initialization is `outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt`. It already contains football-domain adaptation and last-12-block LoRA weights, while retaining the same single-view architecture used by the crop experiment. Do not initialize the first crop ablation from a dual-ROI, multi-layer readout, or spatial-attention checkpoint.

Use `model.init_checkpoint` with non-strict model loading. Do not resume the old optimizer, scheduler, scaler, epoch counter, or thresholds because the crop changes the input distribution. Keep `frame_proj`, temporal Transformer, clip head, and frame-event head weights; their tensor shapes and semantic roles are unchanged.

If `vitl16_clean_negative_margin_raw_16f_hr/best.pt` later beats A-best on both the raw-label validation protocol and the six-video long-video protocol while satisfying the recall floors, it may replace A-best as the common initialization. In that case, fork two runs from exactly the same checkpoint:

- `S0-control`: no crop, continuation under the selected training objective.
- `S1/S3-crop`: identical data, losses, optimizer settings, and schedule, changing only the crop policy.

If clean-negative does not improve the baseline, use A-best and the standard strong-LoRA12 E1 objective, with `clean_negative_rank_loss_weight=0`. The first crop experiment remains single-view, `16` frames, `640x1120`, `last_cls_patch_mean`, `cls_transformer`, frame detection enabled, no ROI branch, no multi-layer readout, no focal replacement, and no hard-negative mining.

For the modest `8%-12%` crop, initialize all matching model weights rather than training from DINO pretraining alone. Use a fresh optimizer and a conservative adaptation schedule: head LR `5e-5`, LoRA LR `2e-6` to `3e-6`, BF16, and `3-4` epochs. If the first epoch is unstable, use a one-epoch head-only warm-up before enabling LoRA; do not reset the trained temporal/head weights.

## Multi-ROI Complementary Evidence (Proposed)

### Motivation

A single ROI can increase the effective pixel density of distant players and the ball, but one noisy crop can omit the decisive action. The first multi-ROI experiment should use exactly two complementary ROI tubes, not three arbitrary top-scoring boxes. The expected gain comes from higher `any-ROI` action coverage; correlated or duplicated crops only add compute and noise.

### Candidate construction

Generate several clip-level tube candidates and select two with a diversity constraint:

- `ROI-A`: ball track or interpolated ball trajectory plus the nearest interacting player group. If ball evidence is absent, use the most temporally stable on-pitch player-interaction cluster.
- `ROI-B`: goalkeeper/goal-adjacent interaction region when supported; otherwise use a second stable player cluster that has low overlap with ROI-A.
- Require temporal stability and `IoU(ROI-A, ROI-B) <= 0.35`. Score candidates by trusted-object coverage, track continuity, person compactness, goal adjacency, and area efficiency rather than detector confidence alone.
- Preserve a stable tube over the clip with smoothing. Do not independently reorder ROI slots per frame.
- Missing or low-quality candidates use an invalid mask and a learned null token. They must not fall back to a duplicate full image.

Ball misses are handled by track interpolation and player-cluster fallback. Goal misses are handled by the second player cluster. Both positive and negative clips use the identical candidate generator to prevent crop-selection shortcuts.

### Fusion architecture

Use global features as the baseline and ROI features as optional residual evidence:

1. Encode the full frame with the A-best global adapter.
2. Re-encode both crops at local resolution so the experiment genuinely recovers spatial detail; ROIAlign on the already-downsampled global patch map is not sufficient for this hypothesis.
3. Add ROI type, normalized box coordinates, validity, detector/track quality, and time embeddings.
4. Perform permutation-aware set attention over `[global, ROI-A, ROI-B, null]` for each sampled time, producing `fused_t = global_t + zero_init_residual_t`.
5. Feed the 16 fused temporal tokens into the existing CLS Transformer and existing clip/frame heads. Do not change the temporal head in the first ablation.

The null path and zero-initialized residual guarantee that the initial fused output equals the global A-best output. ROI dropout (`p=0.25` per ROI, small probability of dropping both) prevents dependence on one noisy crop.

Use separate LoRA adapters on one shared frozen DINO base:

- Global LoRA starts from A-best.
- ROI-A and ROI-B share one ROI-specific LoRA adapter initialized from the global adapter.
- Phase 1 freezes global LoRA and trains ROI LoRA plus fusion only.
- Phase 2 enables global LoRA at a lower LR only after Phase 1 demonstrates gain.

This avoids the earlier gradient conflict where full-image and crop distributions updated the same LoRA parameters. Do not add an independent ROI classification loss in the first version; noisy ROI should contribute only through the final fused clip/frame objective.

### Controlled experiments

- `M0`: A-best global-only reference.
- `M1`: global plus the current single ROI, using the new null/residual fusion.
- `M2`: global plus two complementary ROI tubes; this is the primary test.
- `M2-duplicate`: duplicate ROI-A into both slots as a compute/control check. M2 must beat this control to demonstrate spatial complementarity.

Start with `384x640` local crops. If memory is limiting, process 8 local frames per ROI while retaining all 16 global frames; two 8-frame local paths add approximately 34% linear token work over a `608x1200` 16-frame global path. Move to 16 local frames only after M2 shows value.

Before training, report `any-ROI` coverage for ball, nearest interacting players, goal/keeper, tube validity, ROI overlap, and visual examples for TP/FP/FN clips. Continue only if two ROIs materially raise action-element coverage over one ROI and the six-video evaluation improves shot/save precision by at least `2pp` without more than `1.5pp` recall loss.

## 24. Dual-ROI Temporal-Memory Fusion

This experiment tests two separate hypotheses that the aligned dual-ROI runs could not isolate:

1. Two ROI paths can complement global temporal coverage instead of repeatedly encoding the same 16 timestamps.
2. ROI should act as detail memory for the full-image representation, not as an independent clip classifier.

### 24.1 Three complementary timelines

`video.dual_sampling=multi_staggered` keeps 16 stratified global frames and samples one early and one late ROI timestamp from every temporal bin. For a normal 10-second clip this gives approximately 48 distinct source timestamps:

```text
global: 16 stratified anchors
ROI-A:  16 early-bin detail frames
ROI-B:  16 late-bin detail frames
```

The number of DINO forwards is still 48 frames, equal to the previous aligned `global16 + ROI-A16 + ROI-B16` experiment. The change increases temporal coverage without increasing backbone compute.

### 24.2 ROI memory attached to the global temporal head

The new fusion is `model.view_fusion=dual_multi_roi_memory`:

```text
global frame tokens [B,16,512]
        |
        | queries
        v
masked cross-attention <--- ROI-A/ROI-B memory [B,32,512]
        |
quality-gated, small residual
        |
original global Temporal Transformer
        |
fused frame eventness + one fused clip classifier
```

The two ROI paths share the same DINO/LoRA and local projection. ROI metadata, validity, real timestamps and ROI slot embeddings condition the memory. Invalid ROI tokens are masked. A learnable residual gate starts at `0.02`, so initialization stays close to the loaded checkpoint while cross-attention receives gradients immediately.

There is no independent ROI classification loss and no final convex global/local logit interpolation. Clip BCE and Gaussian frame eventness directly supervise the fused global path. Frame ranking is disabled because point annotations may be temporally biased.

### 24.3 Files and command

- Config: `configs/football/dinov3_vitl16_dual_roi_temporal_memory_safe_neg_16f_hr.yaml`
- Runner: `scripts/run_football_dual_roi_temporal_memory.sh`

```bash
INIT_CHECKPOINT=outputs/football_events/vitl16_dual_roi_shared_16f_hr/best.pt \
bash scripts/run_football_dual_roi_temporal_memory.sh 3,4,6,7
```

The primary comparison is against `vitl16_dual_roi_shared_16f_hr` on the same validation set and six-video window-overlap protocol. Continue only when shot/save precision improves while recall stays within 1.5 percentage points. Diagnostic logs include `roi_memory_residual_gate`, `roi_memory_quality_a`, `roi_memory_quality_b`, and ROI-B validity.


## 25. K4 Adaptive Spatial Tokens (2026-08-17)

### 25.1 Hypothesis

The previous frame readout compresses the final DINO layer to `CLS + patch_mean`. A small action region can therefore be diluted by grass, sky, stands, and inactive players before temporal modeling. This experiment retains all final-layer patch tokens and learns four adaptive spatial queries per frame:

```text
DINO patches [B,T,N,1024]
  -> 4 global-conditioned patch queries
  -> 4 distinct local tokens [B,T,4,512]
  -> [global, local-1, local-2, local-3, local-4] per frame
  -> existing CLS Temporal Transformer
  -> existing clip classifier
```

The four slots are not assigned fixed ball/goal/player/crowd labels. Independent base queries, persistent slot embeddings, global-frame conditioning, slot dropout, and weak diversity/overlap regularization let useful roles emerge without depending on noisy detections.

### 25.2 Supervision and checkpoint compatibility

The final clip BCE remains the primary objective and backpropagates through hard-free spatial pooling. A low-weight local MIL objective uses all `T x 4` slot logits: a positive clip requires at least one high local response, while trusted negatives suppress every slot. This does not use the possibly biased event timestamp. Query diversity and attention-overlap penalties only prevent slot collapse.

The existing 16-frame temporal position embedding is retained. All five tokens from one frame share its temporal position and receive distinct slot-type embeddings, so A-best temporal/head weights load without resizing. The controlled first run freezes loaded DINO/LoRA and trains the new spatial readout plus existing temporal/frame/clip heads.

### 25.3 Files and launch

- Config: `configs/football/dinov3_vitl16_spatial_token_pool_k4_16f_hr.yaml`
- Runner: `scripts/run_football_spatial_token_pool_k4.sh`
- Transition watchdog (archived): `archive/2026-09-03/retired_watchdogs/scripts/watch_temporal_difference_then_spatial_token_k4.sh`
- Test: `tests/test_spatial_token_pooling.py`

Manual launch:

```bash
INIT_CHECKPOINT=outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt \
PER_GPU_BATCH_SIZE=4 TARGET_EFFECTIVE_BATCH_SIZE=64 \
bash scripts/run_football_spatial_token_pool_k4.sh 4,5,6,7
```

The installed watchdog waits for temporal-difference epoch 4, checks two consecutive GPU/CPU/RAM samples, then launches the command above. It exits instead of launching if the prerequisite run terminates without complete checkpoint and metrics artifacts.

### 25.4 Diagnostics and decision rule

Training logs include `spatial_token_mil_pos_loss`, `spatial_token_mil_neg_loss`, `spatial_token_attention_entropy`, `spatial_token_attention_overlap`, `spatial_token_gate_0..3`, and `spatial_token_context_scale`. Healthy behavior is nonzero query gradients, gradually decreasing positive/negative MIL losses, separated slot attention, and gates that move without immediately saturating.

Evaluate with the same validation split, tuned recall floors, and six-video window-overlap/PointNMS protocols as A-best. Continue or unfreeze LoRA only if shot/save precision improves by at least `2pp` with recall loss no greater than `1.5pp`, or validation mAP improves by at least `0.5pp`. If K4 does not improve frozen-readout results, adding detection semantics or more query slots is not justified; the next test should change backbone/domain adaptation rather than make the pooling head larger.

## 26. other_action Hard Negative (protocol_v2): confirmed negative (2026-08-20)

对照: `vitl16_weekend_event_anchor_512x896_sym20_st_control_4ep_protocol_v2` (control) vs
`vitl16_weekend_event_anchor_512x896_sym20_st_other_action_hn_4ep_protocol_v2` (HN)。

配置: HN manifest = ST checkpoint 重打分后的 other_action(拦截/解围/抢断/盘带等, `other_action_reviewed_train_shot_save_score_filtered_p005_st_rescored.json`),
clip 采样约 2%, BCE `loss_weight=4` + `hard_negative_rank_loss`(0.15, margin 0.6)。
已确认分支生效(fraction 稳定 ~2%、loss 有响应),这是第二次独立确认的 HN 负结果——本次不是配置 bug。

### 26.1 protocol_v2 tuned 口径

| metric | control ep2 | HN ep2 | Δ | control ep4 | HN ep4 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| mAP | 0.7184 | 0.7031 | -1.53pp | 0.7155 | 0.7012 | -1.43pp |
| micro P | 0.6434 | 0.6031 | -4.03pp | 0.6147 | 0.6093 | -0.54pp |
| micro R | 0.7807 | 0.7903 | +0.96pp | 0.8086 | 0.7683 | -4.03pp |
| micro F1 | 0.7054 | 0.6841 | -2.13pp | 0.6984 | 0.6796 | -1.88pp |

### 26.2 confidence separation (ep4, 正/负 score mean)

| class | control pos | HN pos | Δpos | control neg | HN neg | Δneg |
|---|---:|---:|---:|---:|---:|---:|
| shot | 0.534 | 0.455 | -8.0pp | 0.080 | 0.072 | -0.8pp |
| save | 0.540 | 0.477 | -6.3pp | 0.117 | 0.104 | -1.3pp |
| set_piece | 0.648 | 0.650 | +0.2pp | 0.050 | 0.055 | +0.5pp |

### 26.3 结论

- 负结果,性质是"生效后的错误方向学习",不是配置未生效:
  - **set_piece 完全不动**是干净的内部对照——HN 只作用于 shot/save 且确实参与训练;
  - shot/save 正样本被压 6-8pp,负样本只被压 0.8-1.3pp → 模型整体变保守,伤的是正样本排序(AP/F1 双降)。
- 机制判断: other_action 与 shot/save 语义同域(射门前/进攻链路上的拦截、解围、抢断),硬 BCE 负标签要求
  模型在**共享证据子空间**上同时压低"前奏"与"事件",模型只能去权重化共享证据 → 正样本先受伤。
  HN 只告诉模型"不是 shot/save",没告诉"为什么不是",学不到细粒度判别边界。
- tail_gap 细节: shot tail_gap 微升(-0.204 → -0.181)只来自负样本 p90 下降(0.317→0.266),
  同时正样本 p10 也下降(0.113→0.085)——边界没有变干净,只是整体更安静,operating point 更差。

**决定**: 训练态 broad other_action HN 不继续加训,不升级为显式 other_action 类;HN 思路转移到
verifier/reranker 阶段(22.3 主线):用"高分假阳性桶 + retention"而非粗暴负标签。
control 为当前可靠分支;等 35-val PointNMS tol5 正式结果做最终确认。
如再试训练态 HN:降 loss weight、去掉/弱化 rank loss、只挖真正确认的高分 FP 且加时间距离保护、人工目审。

## 27. 瓶颈定位双轨道(2026-08-20)

背景: HN 负结果后,需要回答"瓶颈在特征提取还是时序头、小头是否更好"。诊断结论:
冻结 patch 探针 AUC 0.36-0.57(近随机)、头结构消融(E1-E5 全 ≤ E1)、control 训练
`frame_mil_pos_clip_loss=1.31`(稳定压不下去)三者共同指向**监督信号/训练目标**,而非容量。

### 轨道 1:centered anchor 监督修复(主 bet,GPU1 单卡)

- 脚本: `train_football_events_v1.py` — 仅改 `_sample_window` 训练分支:正样本窗口由
  "anchor 在窗口最后 1s" 改为 `centered_window`(anchor 居中 + 1.5s jitter),与 eval 对称
  (原 train/eval 不对称:训练 anchor 在尾部、评估在正中)。
- 配置: `configs/football/dinov3_vitl16_weekend_event_anchor_512x896_sym20_st_centered_anchor_4ep_protocol_v2.yaml`
  与 control 严格等价: 等效 batch 96(单卡 batch2 × accum48)、总 LR 4e-5、init 同 checkpoint、4ep。
- 启动: `tmux` session `football_centered_anchor_v1`, `python train_football_events_v1.py --config ...`
- 早期信号(ep1 step20): `frame_mil_pos_clip_loss = 0.598` vs control ep1 全程 1.19-1.46 —— 减半。
- 注意: 单卡 wall time ~9.4h/epoch,4ep 约 37h;信号 1ep 内可见,可按需提前评估。

### 轨道 2:小头消融(验证"头容量无关",等 35-val 评估结束接 GPU2+3)

- 配置: `configs/football/dinov3_vitl16_weekend_event_anchor_512x896_sym20_st_smallhead_4ep_protocol_v2.yaml`
  control 等价 + `hidden_dim 512→256, temporal_layers 4→2`(其余全等;init 非 strict 兼容 resize)。
- 已归档启动 watchdog: `archive/2026-09-03/retired_watchdogs/scripts/watch_start_smallhead_after_35val.sh`(tmux `football_smallhead_watch`)等 HN ep1 +
  v2ep3 两个 35-val 评估 summary 出现后自动 `python train_football_events.py --config ...`(GPU2+3, accum 12)。
- 预期: 持平或微降(E1-E5 证据);若明显变差则"头容量无关"+ "当前头大小非瓶颈" 双重确认。

### 其他变更

- 已终止并归档 `archive/2026-09-03/retired_watchdogs/scripts/watch_eval_control_hn_epochs_35val.sh` 循环 watchdog(它会占 GPU1/2 做 control/HN
  ep2-4 的 35-val 评估十几小时);HN ep1 评估是独立进程不受影响,继续跑完。control/HN ep2-4
  的 35-val 评估如需补做,可从归档位置恢复后执行(已完成的部分会自动跳过)。
- 35-val window-overlap 已可离线重算: `scripts/recompute_football_eval_protocols.py --run-dir <35val run>`
  从缓存 `window_predictions.csv` 同时算 point_nms + window_overlap,m0_st/m0_e1/control 的结果见各 run 目录
  `window_overlap_recompute.json`(m0_st micro F1=0.476 / m0_e1 0.468 / control 0.432)。

## 28. v4 shot_hpm_save_cond(2026-08-25)

### 28.1 目的

v3(independent_preprojection + save/set_piece 监督修复)两次启动均卡死——decode retry 循环无墙钟
保护,cap.grab() 可无限阻塞(1985527741064613890 的 720P 文件 ffmpeg 可解但 cv2 挂死,166/166 审计
通过佐证)。v4 在 v3 监督栈之上做三个改动:

1. **Shot Hard Positive Tail Margin loss**: EMA floor(每 class,进程内状态,不入 checkpoint,
   resume 后从 margin 重热)+ softplus margin,拉高低于 floor 的正样本尾(阈值塌缩的靶向修复)。
2. **Save-conditioned-on-Shot**: save head 输入加零初始化残差,来自 shot 自身(detach)frame tokens
   的窗口 max 分数 + 距峰值时间。窗口级近似(shot 在 save 后同窗会泄漏,causal cummax 是 v2)。
3. **解码墙钟超时保护**: 每次 decode attempt 60s deadline(thread-local,worker 进程内),超时
   raise VideoDecodeError → 走既有 retry/mask 路径。拦不住单次无限阻塞的 cap.grab(),FFmpeg
   10s 构造参数仍是兜底。

Loss 结构其余保持 v3 原样(未精简)。

### 28.2 配置与启动

- 配置: `configs/football/dinov3_vitl16_independent_preprojection_evidence_ddp_lora_last8_r8_save_setpiece_supervision_v4_shot_hpm_save_cond.yaml`
- HPM: `hard_positive_tail_margin_loss_weight=0.2`,labels=[shot],quantile=0.2,ema_decay=0.95,floor 初值 0.25
- 条件化: `save_shot_conditioning=true`,hidden_dim=256,zero-init 残差;参数 +132,612 → class_evidence_local 2,845,840
- 超时: `data.decode_attempt_timeout_sec=60`(0=禁用)
- init: progressive best.pt strict=false(matched=681 missing=12);12 epochs;recall floors 0.85/0.8/0.8
- 启动: `CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc-per-node=4 ...`(2026-08-25 14:5x 启动)
- smoke 已验证: HPM log 键存在(floor_shot 初始 0.25)、MLP 参数精确 +132,612、checkpoint missing=12 无 raise、
  2 step 训练 + 反向 + pre_eval_recovery 保存均正常

### 28.3 判据

阈值、pos_p10、floor precision(同前);HPM 生效看 `hard_positive_tail_floor_shot` 从 0.25 向 shot
底部 0.2 分位 logit 收敛、violation_fraction 下降;save 提升看 save 的 pos_p10 与 save 尾部分布
(条件化应抬高 save 尾部,同时时间特征给 save 序列先验)。
