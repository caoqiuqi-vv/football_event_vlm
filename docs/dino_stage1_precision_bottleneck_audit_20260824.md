# DINO Stage1 Precision 瓶颈审计（2026-08-24）

## 结论

当前 18-video 高召回、低 precision 不是单一的 calibration 问题，也不能归因于“时序头太小”。证据支持以下优先级：

1. **LoRA 配置生效但训练无效，是必须先修的实现/运行缺陷。** V1.1 best 和 Stage1 last 中 12 个 `lora_b` 均严格为零，故所有 LoRA 有效增量 `alpha/r * B @ A` 都为零。两个 checkpoint 的 Adam `exp_avg`、`exp_avg_sq` 对 24 个 LoRA 参数也全部为零。DINO 的足球视觉表征实际上从未被 LoRA 更新。
2. **主排序只有平均意义上的可分性，低尾正例和高尾负例严重重叠。** Stage1 epoch5 online 的 mean gap 为 shot/save/set-piece `0.420/0.238/0.350`，但 tail gap 为 `-0.185/-0.260/-0.251`。在长视频低基率背景中，负例高尾会淹没正例低尾。
3. **Stage1 从一个已训练 V1.1 checkpoint 做“简单任务 warmup”实质上是倒退课程。** 490/490 参数完整加载，没有随机新头；但 localization/rank/positive-margin 权重被同时降小，正例 p10 和 frame top-k 均退化。Stage1 并未建立新的语义能力，反而遗忘了已有定位与分数尺度。
4. **共享 patch pooling + 共享 temporal token 是第二层、可验证的表征瓶颈。** 当前一个 `LayerNorm(1024)+Linear(1024,1)` 同时服务三个层、三个类别和全部时刻；三层 `CLS + pooled patch`（6144维）直接压缩到384维；三个事件再共享同一个 temporal clip token。它能学习通用“足球活跃度”，但没有显式机制学习 shot/save/set-piece 各自的局部证据。
5. **partial labels 是次要因素，不是总根因。** 当前 train unknown 比例为 shot `0%`、save `8.77%`、set_piece `2.56%`。它会恶化 save precision，但解释不了 shot 和所有类别的长视频 FP。
6. **训练/长视频推理的基率和窗口分布不一致放大了 FP。** 训练为 5330 positive + 15990 negative（1:3），正窗事件固定落在3–7秒；长视频扫描包含大量远背景、边缘事件和重复重叠窗口。18-video strict window P 仅10.85%，合并 segment 后升到21.76%，说明约一半问题来自重复/窗口粒度，剩余仍是 hard-context 排序不足。

## 指标证据

### V1.1 init 与 Stage1

| 模型 | mAP | tuned micro P | tuned micro R | frame top-k shot/save/set |
|---|---:|---:|---:|---:|
| V1.1 best (init) | 0.577 | 0.378 | 0.881 | 0.901 / 0.881 / 0.960 |
| Stage1 online best mAP (e3) | 0.586 | 0.360 | 0.904 | 0.894 / 0.878 / 0.953 |
| Stage1 online e5 | 0.561 | 0.357 | 0.908 | 0.840 / 0.820 / 0.925 |
| Stage1 EMA e5 | 0.573 | 0.357 | 0.898 | 0.859 / 0.847 / 0.937 |

V1.1 init 的 positive p10 为 `0.299/0.188/0.123`；Stage1 online e5 降为 `0.025/0.0036/0.045`。这不是单纯 sigmoid 校准：AP 与 frame top-k 也下降，只是 score-scale 坍塌最显眼。

### 模块真实变化（V1.1 init -> Stage1 e5 EMA）

| 模块 | 参数量 | relative L2 delta |
|---|---:|---:|
| LoRA last6 | 294,912 | 0.10%（仅A的weight decay；B始终为0） |
| shared patch attention | 3,073 | 0.39% |
| 6144→384 frame projection | 2,371,968 | 2.62% |
| temporal stem + 2-layer transformer | 6,225,792 | 6.79% |
| clip head | 149,763 | 4.45% |
| frame head | 1,923 | 5.75% |

时序头真实学习了；视觉骨干没有学习。继续放大或加深时序头不能补偿没有事件条件化局部证据、LoRA 为零和监督低尾重叠。

### LoRA 断点范围

- CPU 单帧完整 `logits -> BCE -> backward` 单元测试中，所有 `lora_b.grad_norm` 均非零（约0.04–0.48），说明 `LoRALinear.forward`、DINO attention 和单模型 activation checkpoint 路径本身可导。
- 实际多卡 checkpoint 中，LoRA Adam step 分别达到 V1.1 `2368`、Stage1 `1480`，但所有一/二阶动量严格为零。
- 因此断点位于真实多卡训练路径，而非 loss 数学本身；当前最高概率是 `DataParallel + backbone activation checkpoint` 的 replica/反向组合，仍需用真实多卡单 batch grad assertion 最终定位。

## 最小变量实验顺序

### E0：LoRA 梯度闸门（必须先做，不算效果实验）

在第一个 optimizer step 前后记录并断言：

- `lora_a_grad_norm`、`lora_b_grad_norm`；
- `lora_b_abs_max_before/after_step`；
- 每个 epoch 的 `||alpha/r * B@A|| / ||W_base||`。

验收：首步 `lora_b_grad_norm > 0` 且 step 后 `lora_b_abs_max > 0`。未通过就立即停止，不再浪费完整 epoch。

定位矩阵只需各跑一个真实 batch：

1. 单卡 + checkpoint on；
2. DataParallel 2卡 + checkpoint off；
3. DataParallel 2卡 + checkpoint on。

若只有第3项失败，改 DDP 或暂时关闭 backbone checkpoint；不要用增加 LR 掩盖零梯度。

### E1：有效 LoRA vs frozen backbone 单变量对照

保持 init、数据顺序、时序头、loss、采样、LR schedule 完全一致，仅比较：

- Control：明确冻结 backbone；
- Experiment：修复并通过 E0 的 last6 LoRA。

至少观察2–3 epoch。验收不是 tuned threshold，而是：

- val15 per-class AP、mAP；
- positive p10、negative p90、tail gap；
- frame top-k 不下降超过1pp；
- cached long-video `precision @ recall>=80%` 与真实去重参与度。

有效判定建议：mAP至少 +1.5pp，shot/save AP至少各 +2pp 或 tail gap 稳定改善，且长视频 P@R80 提升超过视频级 bootstrap CI。

### E2：class-conditioned prototype/evidence（E1后才做）

这条线比继续增加 logit rank 更合适，但必须避免多标签冲突：

- 从 temporal/frame token 生成每类独立 `z_c = normalize(W_c(token))`，不要让 shot/save/set_piece 共用一个对比 embedding；
- prototype 按类别维护，并优先跨视频采正对，防止记住场地；
- clean positive 和 weak positive 都向同类 prototype 靠近，只是 clean 权重大、weak 权重小；**不做 `clean score > weak score` 排序**；
- save+shot 共现不是负对；未知标签不参与负 prototype；负 margin 只用 label-complete clean negatives；
- 总权重从0.02–0.05起，防止压过 BCE。

为什么不继续只加 logit rank：Stage1 clean-negative rank 的平均 logit gap 已约6.2、violation仅6.7%，说明当前 pair 大多已是 easy pairs，loss基本失活；它无法改善真正决定长视频 precision 的 negative p90 和 positive p10。

## 产品层验收

18-video 当前 full-segment 去重人工参与度63.64%、human-visible recall96.67%。要达到参与度低于30%且 recall 接近90%，候选覆盖时长需至少再减半，同时最多损失约6.7pp可见 recall。

DINO 表征实验的主 KPI 应固定为 15-val 定阈值后的：

- `precision @ recall>=80%/85%/90%`；
- deduplicated human participation；
- human-visible recall；
- per-video bootstrap CI。

18 external test 只使用 val15 固化阈值评一次，不能在18-test自身重新定阈。

## 产物

- 机器可读审计：`outputs/football_events/vitl16_stage1_peak_spotting_v1_3_curriculum_small384_raw720_no_pn_24f/stage1_bottleneck_audit.json`
- 审计脚本：`scripts/audit_dino_stage1_bottleneck.py`
