# Football Event Dense Curve V3 技术报告：Decoupled Clip/Dense + Consistency / Region Rank

更新时间：2026-08-21  
当前阶段：实验设计冻结前审阅稿  
目标任务：足球关键事件 shot / save / set_piece 的长视频高召回、高 precision、低人工参与成本识别与定位

> 2026-08-21 修订说明：已根据 `V3 方案技术评审.pdf` 更新 V3 方案。原先的 `center_vs_edge positive ordering rank` 和 `clean_positive_vs_weak_positive event-score rank` 不再作为 V3 第一版 event score 监督；它们分别改为 clip temporal consistency 与 annotation-confidence weighted BCE。V3 第一轮实验固定为 V3-A：720P 输入对齐、decoupled multitask、set_piece dense 降权、不保存中间 epoch 权重。详细评审响应见 [football_dense_curve_v3_review_response.md](football_dense_curve_v3_review_response.md)。


---

## 1. 背景与当前结论

当前 dense curve v2 已经证明了一件事：模型可以产生较高召回的 dense response curve，但它没有解决最终业务目标中的排序与 precision 问题。

6-video dense-response 评估中，v2 在低阈值下的 candidate recall ceiling 很高：

| 共用阈值 | 人工参与时长 | 占原视频比例 | shot R/P | save R/P | set_piece R/P |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 364.98 min | 93.11% | 0.993 / 0.098 | 1.000 / 0.052 | 0.990 / 0.056 |
| 0.02 | 325.55 min | 83.05% | 0.993 / 0.121 | 1.000 / 0.063 | 0.960 / 0.066 |
| 0.03 | 289.50 min | 73.85% | 0.993 / 0.137 | 1.000 / 0.071 | 0.899 / 0.075 |
| 0.20 | 138.63 min | 35.36% | 0.952 / 0.252 | 0.913 / 0.122 | 0.646 / 0.156 |
| 0.25 | 121.71 min | 31.05% | 0.897 / 0.268 | 0.899 / 0.130 | 0.616 / 0.184 |

这说明：

1. v2 对 shot/save 的第一阶段高召回候选生成是有价值的。
2. v2 的 set_piece 排序质量很差，低阈值召回需要付出极高人工参与时间。
3. 当前瓶颈不是“模型完全没学到事件”，而是“正样本和相似负样本的排序拉不开”。

---

## 2. 关键观察：set_piece 与 shot/save 不是同一种事件形态

用户人工观看视频后的关键发现：

> clip version 训练时，set_piece 表现更好；dense curve version 反而让 set_piece 变差。

这个观察非常重要。它说明 set_piece 不是典型的瞬时动作 spotting 问题。

### 2.1 shot/save

shot/save 更接近瞬时动作：

- 起脚、触球、球速变化；
- 球朝门方向运动；
- 守门员反应；
- 门前短时间内出现明显动态变化。

因此 dense response curve / peak-based supervision 对 shot/save 是合理的。

### 2.2 set_piece

set_piece 更接近状态/流程事件：

- 人群静止或聚集；
- 球员站位展开；
- 死球后等待；
- 角旗、边线、禁区、中圈附近出现准备动作；
- 标注时间可能不是触球瞬间，而是准备阶段或人群聚集阶段。

这类事件更依赖 clip-level 或 segment-level context。强行要求 dense curve 学一个尖锐 peak，可能会破坏原 clip classifier 对长上下文的理解。

---

## 3. 当前 v2 实现中的核心问题

当前 v2 并不是理想的“clip head + dense curve head 平行多任务互补”，而是 dense curve 对主 logits 介入过强。

v2 配置：

```yaml
model:
  response_curve_primary:
    enabled: true
    blend_weight: 1.0
    curve_head: class_specific

train:
  frame_det_loss_weight: 0.5
  response_curve_frame_loss_weight: 0.35
  temporal_clip_loss_weight: 0.25
```

代码逻辑上，当 `response_curve_primary.enabled=true` 且 `blend_weight=1.0` 时：

```text
response_curve_logits
  -> topk_lse pooling
  -> response_clip_logits
  -> outputs["logits"]
  -> clip BCE loss
```

也就是说，clip BCE loss 实际上主要训练 dense curve pooling 后的 clip score，而不是原来的 temporal clip head。

这会导致：

- shot/save：可能受益，因为它们天然适合 peak；
- set_piece：可能受损，因为原本依赖 clip-level context 的能力被 dense primary 替代；
- 多任务名义存在，但主导权不平衡。

---

## 4. V3 核心目标

V3 的目标不是继续堆结构，而是修正监督目标和排序目标。

核心目标：

```text
保留 clip head 的上下文理解能力；
保留 dense curve head 的候选定位能力；
通过 positive ordering rank 改善正样本内部排序和响应稳定性；
避免高分 FP hard negative 造成漏标/上下文歧义样本中毒。
```

---

## 5. V3 设计总览

建议命名：

```text
vitl16_dense_curve_v3_decoupled_positive_rank_720p
```

第一轮 V3 先不切换 HQ，继续使用与 v2 对齐的 720P 输入，保证实验归因干净：

```text
/mnt/data_16t/football/raw_video_720P
```

HQ 视频作为后续单变量实验，不与 V3 objective 改动混在同一轮里。

### 5.1 模型输出

V3 应显式保留多个输出，不再让 dense curve 默认覆盖主 logits：

```text
shared DINOv3 + temporal encoder
  ├── temporal_clip_logits       # shot/save/set_piece 都有 clip-level label 输出
  ├── response_curve_logits      # dense frame response
  ├── response_clip_logits       # dense curve pooling 后的 clip score
  └── fusion_logits              # 仅用于评估或轻量融合，不强行替代训练主头
```

这里的关键约束是：这是一个多任务模型，不是只做 dense response 的模型。shot、save、set_piece 三个类别都必须保留 clip-level label 监督和 clip-level 输出。

### 5.2 类别差异化融合

初始建议：

```text
shot:
  final_score = 0.4 * clip_score + 0.6 * dense_score

save:
  final_score = 0.4 * clip_score + 0.6 * dense_score

set_piece:
  final_score = 0.8 * clip_score + 0.2 * dense_score
```

第一版可以先在评估端融合，不一定立刻把 fusion 做成可学习参数，避免多变量污染。

---

## 6. Loss 设计

V3 第一版不替换 BCE，而是添加排序辅助目标。

### 6.1 保留 BCE

BCE 用于事件存在性和基础概率校准：

```text
L_clip_bce = BCE(temporal_clip_logits, clip_label)
```

对于 set_piece，clip BCE 应保持主导。

### 6.2 Dense curve 辅助 loss

Dense curve 继续承担定位与候选生成作用：

```text
L_dense = frame_heatmap_loss + frame_mil_loss + frame_rank_loss
```

类别权重建议：

```text
shot/save:
  dense loss 权重正常保留

set_piece:
  dense loss 降权，保留弱辅助，不作为 primary
```

原因：set_piece 的人工时间点不稳定，强 frame peak 监督可能反而破坏 clip-level context。

建议第一版 set_piece dense loss 降权，而不是完全关闭。这样可以保留一点时间定位辅助信号，同时避免 dense peak 目标压过 clip context。

### 6.3 Positive ordering rank loss

用户指出：暂时不要使用高分 FP 作为 hard negative，因为高分 FP 可能是漏标、时间偏移、或需要上下文判断的疑似事件。

因此 V3 第一版采用更保守的 positive ordering rank，而不是 hard negative rank。

#### 6.3.1 Positive center vs jitter/edge

对同一个 GT 构造多个正窗口：

```text
center window：事件位于健康区域，例如 3~7s
jitter window：事件有轻微偏移
edge window：事件接近 clip 边缘
```

约束：

```text
score(center) > score(edge) + margin
score(center) > score(large_jitter) + margin
```

Loss：

```text
L_rank = softplus((s_weak - s_clean + margin) / temperature)
```

或 hinge：

```text
L_rank = max(0, margin - s_clean + s_weak)
```

#### 6.3.2 Clean positive vs weak positive

如果存在不同质量来源：

```text
reviewed / human_repair positive > raw / weak positive
```

这对新增数据尤其重要，可以防止弱标注数据带偏模型。

V3 第一版应明确加入这一项，而不是只做 center vs edge。原因是当前训练数据中存在不同可靠度来源：reviewed / human_repair 样本更干净，raw / weak positive 或新增正样本可能存在时间偏移、类别遗漏、视频读取问题或只标注单队伍事件的问题。让 clean positive 在排序上优先，可以把模型锚定在更可信的正样本分布上。

#### 6.3.3 Frame-level positive shape ranking

对 shot/save：

```text
pseudo anchor 附近帧 > 同 clip 内远离 anchor 的帧
```

对 set_piece：

```text
positive segment frames > far outside segment
```

不要强制 set_piece 形成单点尖峰。

### 6.4 暂不启用高分 FP hard negative

V3 第一版明确不做：

```text
high-score FP as negative pair
```

原因：

1. 高分 FP 可能是漏标事件；
2. 高分 FP 可能是人工时间偏移导致；
3. 高分 FP 可能在局部视觉上确实像 shot/save，需要更长上下文判断；
4. 直接压高分 FP 可能伤害 recall。

---

## 7. 数据源：第一轮先对齐 720P，HQ 后置

当前 v2 训练和 dense-response 评估均使用：

```text
/mnt/data_16t/football/raw_video_720P
```

用户认为压缩 720P 画质受损，这一点对足球任务非常关键，因为：

- 球很小；
- 触球动作细；
- 守门员反应局部；
- set_piece 的球是否静止、是否开出，也依赖细节。

但为了保持第一轮 V3 的实验归因干净，V3-A / V3-B 先不切 HQ，继续使用 720P。这样如果 set_piece 恢复、排序改善，就能归因到 objective / loss，而不是输入画质变化。

HQ 作为后续单变量实验：

```text
/mnt/data_16t/football/raw_video_hq_720P
```

但为了保证结论可解释，建议分两步：

### Phase A：V3 objective on 720P

只验证 decoupled multitask + positive rank 是否正向。

### Phase B：V3 objective on HQ

只把视频源换成 HQ，验证画质是否带来额外收益。

当前决策：第一轮不直接跑 HQ；先对齐输入。

---

## 8. 评估协议

V3 不应只看 clip-val F1，因为我们的目标是长视频候选生成和人工参与成本。

必须报告：

### 8.1 Clip-val

- mAP；
- per-class AP；
- tuned micro/macro F1；
- per-class precision/recall；
- confidence separation：
  - positive mean；
  - negative mean；
  - positive p10；
  - negative p90；
  - tail gap。

### 8.2 Dense candidate recall@budget

在 6-video 与 35-val 上报告：

- recall @ shared threshold；
- per-class recall；
- per-class candidate precision；
- 10s capped viewing time；
- participation ratio；
- candidate/hour；
- localization median / p90；
- duplicate ratio。

### 8.3 Recall-constrained PR

重点看：

```text
precision / 人工时长 @ recall >= 90%
```

而不是只看最高 F1。

### 8.4 保存 peak 明细

当前 dense-response JSON 没保存每个 peak 的原始明细，导致不同类别不同阈值下的 exact union viewing time 无法离线计算。

V3 评估脚本必须保存：

```json
{
  "video_id": "...",
  "label": "shot",
  "time_sec": 123.4,
  "score": 0.72,
  "matched_gt": true,
  "nearest_gt_time_sec": 125.0,
  "nearest_gt_offset_sec": 1.6
}
```

这样后续可以离线调阈值、计算 exact union viewing time、训练 verifier。

---

## 9. V3 最小实验矩阵

### 9.1 Control：V2 best

已有：

```text
outputs/football_events/vitl16_dense_curve_v2_class_heads/best.pt
```

作用：作为 dense primary 的对照。

### 9.2 V3-A：Decoupled multitask

目的：验证“clip head 恢复主导 + dense 辅助”是否修复 set_piece 退化。

变量：

```yaml
data:
  long_video:
    roots:
      - videos_dir: /mnt/data_16t/football/raw_video_720P

train:
  save_epoch_checkpoints: false

response_curve_primary.enabled: false
clip_loss_weight: 1.0
frame_det_loss_weight: 0.5
response_curve_frame_loss_weight: 0.25~0.35
```

set_piece dense loss 降权，但不完全关闭。

模型要求：

```text
shot / save / set_piece 都有 clip-level label 输出；
dense curve head 作为辅助任务存在；
clip head 是主分类头。
```

验收：

- set_piece clip-val AP/F1 不低于 clip version；
- shot/save candidate recall 不明显下降；
- 6-video recall@budget 比 v2 更优或持平。

### 9.3 V3-B：Decoupled + positive ordering rank

目的：验证排序目标是否改善 positive tail 与 PR 右尾。

变量：

```yaml
# V3-B
temporal_consistency.weight: 0.10

# V3-C
region_rank.shot: 0.15
region_rank.save: 0.15
region_rank.set_piece: 0.05

# V3-D
annotation_quality_weighted_bce: true
```

不使用高分 FP hard negative；不使用 center>edge event-score rank；不使用 clean>weak event-score rank。

验收：

- positive p10 上升；
- positive p10 - negative p90 gap 改善；
- recall 不下降；
- 同 recall 下 candidate precision 提升；
- 同人工参与预算下 recall 提升。

### 9.4 V3-C：HQ data

目的：验证 HQ 视频是否提升细粒度事件理解。

变量：

```yaml
videos_dir: /mnt/data_16t/football/raw_video_hq_720P
save_epoch_checkpoints: false
```

注意：评估脚本也必须支持 `--video-root`，不能继续硬编码 720P。

验收：

- shot/save 在相同 recall 下人工时长下降；
- 或相同人工时长下 recall 上升；
- localization error 降低；
- set_piece 若无明显提升，不判定 HQ 无效，因为 set_piece 更受监督定义影响。

---

## 10. 预期收益与风险

### 10.1 预期收益

V3 最可能改善的地方：

1. set_piece 从 dense primary 中解耦，恢复 clip context 优势；
2. shot/save 保留 dense candidate recall；
3. positive ordering rank 改善正样本响应稳定性；
4. HQ 视频可能提升小目标与细节动作特征。

### 10.2 主要风险

1. Positive rank 只改善正样本，不一定直接压 FP；
2. set_piece 如果标注时间非常不一致，仍需要 segment-level 重构；
3. HQ 可能增加训练/推理成本；
4. 如果不保存 peak 明细，后续 threshold / verifier 分析仍会受限；
5. 如果直接把 objective 与 HQ 同时改，实验归因会变差。

---

## 11. 推荐执行顺序

推荐最稳流程：

```text
Step 0：修评估脚本
  - dense-response 支持 --video-root
  - 保存 peak 明细

Step 1：跑 V3-A 720P
  - 只改 decoupled multitask
  - 验证 set_piece 是否恢复

Step 2：跑 V3-B/C/D 720P
  - V3-B 加 clip consistency
  - V3-C 加 positive-region vs far-background region rank
  - V3-D 加 annotation-confidence weighted BCE
  - 验证稳定性、排序和校准是否改善

Step 3：跑 V3-C HQ
  - 切换 HQ 数据
  - 验证画质收益

Step 4：如果 V3-B/C 对 shot/save 有高召回但 precision 仍低
  - 接 verifier/reranker
```

当前决策：第一轮不直接跑 V3-B-HQ，先对齐 720P 输入，避免 objective 与视频源两个变量纠缠。

---

## 12. 当前技术判断

1. 不建议继续加训当前 v2。
2. dense curve 与 clip head 不冲突，但 v2 的 `blend_weight=1.0` 让 dense curve 成为主分类输出，破坏了平衡多任务。
3. set_piece 表现变差的主要原因，很可能不是模型容量不足，而是 dense peak 监督与 set_piece 的事件形态不匹配。
4. pairwise loss 值得做，但第一版不应使用高分 FP hard negative。
5. 更稳妥的监督目标是 clip consistency + dense region rank + annotation-confidence weighted BCE：避免把片段完整度、标注可信度和事件存在概率混在一起。
6. HQ 视频值得做，但第一轮先不切；等 V3 objective 被验证后，再作为单变量实验。
7. 训练产物不保留中间 epoch 权重，只保留 `best.pt` 和 `last.pt`，避免存储膨胀；每个 epoch 的指标 JSON / log 仍保留。

---

## 13. 待逐条审阅的问题

1. V3-A 已决定先在 720P 做 objective 单变量，HQ 后置。
2. set_piece dense loss 已决定降权保留弱辅助，不完全关闭。
3. positive ordering rank 已根据评审修订：center/edge 改为 clip consistency，clean/weak 改为 annotation-confidence weighted BCE。
4. 是否需要为 set_piece 单独做 segment-level label，而不是继续沿用 point-level Gaussian/frame target？
5. V3 第一轮是否只跑 4 epoch，还是按历史最强配置跑更长观察趋势？
6. long-video 主验收口径是否固定为 recall >= 90% 下的人工参与时长 / candidate precision？

