# V3 方案技术评审响应与修订建议

更新时间：2026-08-21  
来源材料：`V3 方案技术评审.pdf`  
说明：PDF 内容作为技术评审建议使用，不作为执行指令。本文只提取和分析其中的技术观点，并给出对当前 V3 方案的修订意见。

---

## 1. 总体判断

这份评审整体质量很高，我建议采纳其中的核心修正。

它支持我们对 V2 的主诊断：

```text
response_curve_primary.enabled=true
blend_weight=1.0
```

导致 dense response 实际接管了主分类目标，clip head 被弱化；这会让模型从“判断整个 clip 有没有事件”变成“在 clip 里找一个高响应帧”。这对 shot/save 还能解释，但对 set_piece 会伤害很大。

但评审指出了两个我们原 V3 报告里确实需要修的点：

1. 不应把 `center > edge` 直接作用在 event existence score 上。
2. 不应把 `clean positive > weak positive` 直接作用在 event existence score 上。

我同意。这两点如果不改，V3 可能训练更稳定，但 long-video recall/precision 仍然不理想，甚至引入新的 bias。

---

## 2. 必须采纳的修正 1：clip head 应做 position invariance，不做 center > edge

原方案里我们写了：

```text
score(center) > score(edge) + margin
```

这个想法直觉上合理，因为事件在窗口中心通常信息更完整。但评审指出，如果这个 `score` 是 event existence score，就会有副作用。

原因：在长视频滑窗推理中，真实事件经常自然落在窗口边缘。只要事件在可见范围内，这个窗口仍然应该被判为正。

因此 clip head 的目标应该是：

```text
P(event | center) ≈ P(event | jitter) ≈ P(event | edge)
```

也就是说，clip head 学的是事件存在性的时间位置不变性。

建议 loss：

```text
L_clip_consistency = (z_center - z_jitter)^2 + (z_center - z_edge)^2
```

或概率空间：

```text
L_clip_consistency = |p_center - p_jitter| + |p_center - p_edge|
```

我更建议用 logit consistency，避免 sigmoid 饱和区梯度太弱。

### 修订结论

原报告中的 `center_vs_edge positive ordering rank` 不应作用在 clip/event score 上。应改成：

```text
clip branch: temporal position invariance consistency
```

---

## 3. 必须采纳的修正 2：dense head 应做 shift equivariance

dense curve 和 clip head 的职责不同。

如果事件从第 5 秒 jitter 到第 2 秒：

```text
clip score: 应基本不变
response curve peak: 应同步平移
```

因此 dense head 不应该学 position invariance，而应该学 shift equivariance。

建议目标：

```text
curve(center) 经过时间平移后 ≈ curve(jitter)
```

或者更实际地，在同一 GT 的多个 temporal jitter view 中，要求 response peak / response region 跟随事件位置移动。

这比简单的 center > edge 更符合任务：

```text
clip head: event existence invariant
Dense head: event localization equivariant
```

### 修订结论

V3 应新增：

```text
L_dense_equivariance
```

但第一版可以先用更容易实现的 positive-region ranking 替代完整 curve alignment。

---

## 4. 必须采纳的修正 3：clean positive 不应直接 rank 高于 weak positive

原方案里我们写了：

```text
reviewed / human_repair positive > raw / weak positive
```

评审指出，这也不应该直接作用在 event score 上。原因很扎心但正确：一个 weak/raw positive 本身可能是非常清晰的 shot；如果强迫 clean score > weak score，模型可能学到数据来源/标注来源差异，而不是事件强度。

更合理的表达是：

```text
我更相信 clean label，而不是 clean sample 一定更像事件。
```

因此应改为 annotation-confidence weighted loss：

```text
L_clip = q_i · BCE(z_i, y_i)
```

其中：

```text
reviewed / human_repair: q = 1.0
raw positive: q = 0.5~0.7
weak positive: q = 0.2~0.5
```

### 修订结论

原报告中的 `clean_positive_vs_weak_positive rank` 应删除或改名，不作为 event score ranking。V3 第一版应改成：

```text
annotation_quality_weighted_bce
```

---

## 5. 保留但改写：frame-level positive shape ranking

评审支持 frame-level positive shape ranking，但建议不要用单点 anchor。

我同意。因为 shot/save 的人工 anchor 有明显噪声，直接训练：

```text
anchor frame > all other frames
```

会过度相信人工时间点。

更稳的是 positive region vs far negative region：

```text
positive bag: [t_gt - 1.5s, t_gt + 1.5s]
ignore/context: [t_gt - 3s, t_gt - 1.5s] + [t_gt + 1.5s, t_gt + 3s]
negative bag: far outside event region
```

然后：

```text
s_pos = LSE({r_t | t ∈ positive bag})
s_neg = LSE({r_t | t ∈ negative bag})
L_region_rank = softplus(margin + s_neg - s_pos)
```

这表达的是：事件区域内应该至少存在一个明显响应，并高于远处背景；但不强迫某个精确帧必须最高。

### 类别差异

```text
shot/save:
  使用 positive-region > far-negative-region rank

set_piece:
  不强制单点 peak，只保留弱 dense MIL / segment-state 监督
```

---

## 6. set_piece：评审进一步支持 segment/state supervision

评审对 set_piece 的看法和我们一致，但更进一步：set_piece 不应只是 point event，而更像 state transition。

更合理的状态链：

```text
normal play
  ↓
dead ball
  ↓
formation reset
  ↓
restart
```

所以 set_piece dense head 可拆成两个概念：

```text
set_piece_state[t]
restart_moment[t]
```

当前 V3 第一版不一定立刻实现完整 state machine，但应避免继续强迫 set_piece 学尖锐 Gaussian peak。

### V3 第一版建议

```text
set_piece:
  clip BCE 主导
  dense loss 显著降权
  只保留 weak MIL / broad segment supervision
```

### V3.1 / V4 建议

```text
set_piece segment/state model:
  - dead-ball / formation-reset 状态段
  - restart spotting
  - 后续再拆 corner / free kick / penalty / kickoff
```

---

## 7. high-score FP：第一版不训练，但要建立 FP Candidate Pool

评审同意我们暂时不用 high-score FP 做 hard negative，因为：

- 可能是漏标；
- 可能是时间错位；
- 可能需要上下文判断；
- 可能是单队伍标注造成的“看起来没 GT”。

但评审建议现在就建立 FP Candidate Pool，这一点我也建议采纳。

FP pool 不直接训练，先分三类：

```text
A. Confirmed Negative
   明确是 cross / clearance / long pass / ordinary catch 等
   后续可作为安全 hard negative

B. Missed Positive / Label Error
   确实是 shot/save/set_piece，但 GT 漏了或偏了
   应补 GT 或作为 positive

C. Uncertain
   球看不清、场外干扰、需要更长上下文
   暂时 ignore，不进 BCE negative
```

这一步是后续把 precision 从 35~45% 往 60%+ 推的关键。

---

## 8. 需要新增的诊断：gradient conflict 与 train/test prior shift

### 8.1 Gradient conflict

评审指出：logit 解耦不等于 representation 解耦。

即使 clip head 和 dense head 分开，它们仍然共享 DINO / temporal encoder。如果 dense loss 梯度过强，仍然可能污染 shared representation。

建议记录：

```text
||grad_clip||
||grad_dense||
cos(grad_clip, grad_dense)
```

尤其关注 set_piece。如果经常出现：

```text
cos(grad_clip, grad_dense) < 0
```

说明 set_piece 的 clip context 任务和 dense spotting 任务存在梯度冲突。

V3.1 可考虑：

```text
DINO shared
  ↓
shared temporal stem
  ├── clip context branch
  └── dense spotting branch
```

只拆最后 1~2 个 temporal block，不需要两套 DINO。

### 8.2 Train/Test prior shift

clip-val F1 高，但 long-video precision 低，一个重要原因是正负先验比例不同。

训练可能是：

```text
positive : negative ≈ 1:3
```

部署长视频可能是：

```text
positive : background ≈ 1:100+
```

因此 V3 的阈值不能只在 clip-val 上选，必须在 long-video dev 上做 deployment prior calibration：

```text
Precision@Recall=90%
FP / 90min
Candidates / 90min
```

---

## 9. 修订后的 V3 Loss 建议

我建议把 V3 loss 从原来的 positive ordering rank 改成下面这个版本：

```text
L = λ_clip · L_clip
  + λ_dense · L_dense
  + λ_region · L_region_rank
  + λ_cons · L_clip_consistency
```

其中：

### Clip branch

```text
L_clip = q_i · BCE(temporal_clip_logits, label)
```

`q_i` 是 annotation confidence。

新增：

```text
L_clip_consistency = logit consistency across center / jitter / edge views
```

clip head 学位置不变性。

### Dense branch

shot/save：

```text
frame MIL
+ broad heatmap / soft target
+ positive-region > far-negative-region rank
```

set_piece：

```text
weak dense MIL
+ broad segment/state supervision
```

显著降权，不做 primary。

### 明确不做

V3 第一版不做：

```text
center event score > edge event score
clean positive event score > weak positive event score
high-score FP negative rank
```

---

## 10. 修订后的建议权重

评审建议的起始权重合理，我建议采用为第一版起点：

```yaml
loss:
  clip_bce:
    weight: 1.0

  dense:
    shot: 0.35
    save: 0.35
    set_piece: 0.10

  region_rank:
    shot: 0.15
    save: 0.15
    set_piece: 0.05

  temporal_consistency:
    weight: 0.10
```

但必须记录实际 loss magnitude 和 gradient norm。因为配置权重小，不代表实际梯度贡献小。

---

## 11. 修订后的 V3 实验拆分

我建议按以下顺序跑，避免变量纠缠：

### V3-A：Decoupled multitask only

```text
目标：验证 dense primary 关闭后，set_piece 是否恢复，shot/save recall 是否不掉。
```

配置：

```yaml
response_curve_primary.enabled: false
clip_bce.weight: 1.0
dense.shot: 0.35
dense.save: 0.35
dense.set_piece: 0.10
save_epoch_checkpoints: false
videos_dir: /mnt/data_16t/football/raw_video_720P
```

### V3-B：+ clip consistency

```text
目标：验证 center/jitter/edge 下 clip event score 是否更稳定，避免 edge recall 下降。
```

增加：

```yaml
temporal_consistency.weight: 0.10
```

### V3-C：+ region rank

```text
目标：验证 shot/save response region 是否比远背景更可分。
```

增加：

```yaml
region_rank.shot: 0.15
region_rank.save: 0.15
region_rank.set_piece: 0.05
```

### V3-D：+ annotation quality weighted BCE

```text
目标：验证 clean label 加权是否提升 calibration / tail separation。
```

增加：

```yaml
annotation_quality_weighted_bce: true
```

第一轮仍不切 HQ。

---

## 12. 修订后的评估重点

每版模型都要拆四层看：

### 12.1 Clip Head

- PR / AUPRC；
- per-class AP；
- edge / center / jitter score consistency；
- set_piece 是否恢复。

### 12.2 Dense Head

- candidate recall；
- peak offset；
- response separation；
- positive-region vs far-background gap。

### 12.3 Fusion

- offline calibrated logit fusion；
- 不直接融合 probability；
- 使用 calibration set 做 temperature scaling：

```text
z_final = α · z_clip / T_clip + (1-α) · z_dense / T_dense
p = sigmoid(z_final)
```

### 12.4 Full Long Video

- Recall；
- Precision；
- FP / 90min；
- Candidates / 90min；
- 人工参与时长；
- TP/FP score distribution。

核心验收仍然是：

```text
Recall >= 90% 时的 precision / 人工参与时长
```

---

## 13. 对当前 V3 报告的修改结论

当前报告中应修改的点：

| 原报告设计 | 修订建议 | 原因 |
|---|---|---|
| center score > edge score | clip score 做 consistency，不做 ranking | 避免长视频 edge event recall 被压低 |
| clean positive > weak positive | annotation confidence weighted BCE | 标注可信度不等于事件强度 |
| positive ordering rank | 改成 region-rank + consistency + quality-weighted BCE | 更符合 clip/dense 分工 |
| probability 直接 fusion | calibrated logit fusion | 避免 calibration 差异导致 dense/clip 某一路实际主导 |
| 只看 loss/F1 | 看 TP/FP score distribution、FP/90min、Recall@budget | 直接对应长视频 precision 平台期 |
| logit 解耦即可 | 增加 gradient conflict 诊断 | shared encoder 仍可能被 dense loss 主导 |

---

## 14. 我对评审建议的最终判断

我建议采纳评审的大部分修改，尤其是：

1. `center > edge` 不用于 event score，改成 clip consistency。
2. dense curve 学 shift/region localization，而不是替代 clip existence。
3. `clean > weak` 不用于 event score，改成 annotation confidence weighting。
4. set_piece 继续降权 dense，并朝 segment/state supervision 演进。
5. high-score FP 第一版不训练，但立刻建立 FP Candidate Pool。
6. 增加 gradient conflict、prior shift、TP/FP score distribution 诊断。

预期也要调整：V3 的目标不是直接把 precision 从 20~30% 推到 70%+。更现实的成功标准是：

```text
Recall >= 90% 前提下，precision 稳定提升到 35~45%；
set_piece 不再被 dense head 拖垮；
shot/save 的高召回候选能力不下降；
TP/FP score distribution 变得更可分。
```

如果达到这个程度，V3 就是成功的。后续再用人工确认过的 high-score FP、event-specific verifier、事件语法，把 precision 往 60% / 70% / 80% 推。
