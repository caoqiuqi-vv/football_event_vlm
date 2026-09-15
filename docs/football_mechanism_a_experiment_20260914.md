# 足球事件检测辅助机制 A：实现与验证协议

日期：2026-09-14
代码基线：`2d55c81d43d88c4cb4820d546d6b87a32c34355b`
开发分支：`codex/mechanism-a`

## 1. 本轮只回答一个问题

检测定位监督通过改变 DINO 共享表征，能否提升事件识别与定位？

机制 A 不回答“检测证据在推理时如何路由”。在机制 A 获得明确、可复现的事件收益前，不加入 ROI、框坐标、轨迹、检测置信度、object residual、cross-attention 或 motion residual。

## 2. 为什么历史 object auxiliary 实验不能回答这个问题

旧实现同时存在两个因果混杂：

1. `controlled_online_train_scope: object_spatial_aux` 冻结了 DINO/LoRA 和事件头，定位损失不能改变事件头实际消费的共享表征。
2. `ObjectSpatialAuxHead.residual` 又直接加到事件 logits。一旦出现收益，也无法区分是“表征学习”还是“检测证据融合”。

机制 A 将两者拆开：

```text
视频帧 ──> DINO + shared LoRA ──> 帧特征 ──> 时序/事件头 ──> L_event
                    │
                    └──────────> patch heatmap head ──> L_ball

反向：L_event -> 事件头 + shared LoRA
      L_ball  -> heatmap head + shared LoRA
前向：heatmap/box/track/confidence 不进入事件 logits
推理：不需要检测器或离线轨迹
```

形式上：

`L = L_event + λ_ball * L_ball`

其中共享参数集合被硬限制为 DINO backbone 内名称以 `.lora_a`/`.lora_b` 结尾的参数；定位关系 GRU 和 residual 分支冻结。

## 3. 已实现的因果约束

- `representation_only=true` 时，即使 object residual 被人工设置为非零，事件 `logits` 仍与 `global_logits` 逐元素完全相等。
- `controlled_online_train_scope=mechanism_a` 只允许训练：
  - DINO shared LoRA；
  - `frame_proj`、`temporal`、`head`、`frame_event_head`；
  - `object_spatial_aux.heatmap_head`。
- 配置校验器拒绝其它 evidence/fusion 模块、缓存特征、水平翻转和额外训练损失。
- 对照组保持相同模型结构，但 `λ_ball=0`；数据层不会误加载 object target。
- 教师使用 v9 repaired ball-track NPZ。只有 heatmap-usable 行参与监督；缺失帧与 goal 通道保持 unknown，而不是伪造为负样本。
- 教师 `quality_weight` 作为 mask 权重；置信度保留为 provenance，不再重复压低正目标。
- 验证输出增加 `object_localization`，包含 held-out ball valid fraction、positive-frame fraction、top-1 hit 和 heatmap loss。
- 单 GPU probe 可直接测量共享 LoRA 上两项损失的梯度范数、余弦和加权梯度比；正式 DDP 不启用额外 `autograd.grad` 诊断。

## 4. 固定数据与协议

- 训练：canonical train，实际可解码 128 个视频。
- checkpoint/阈值选择：预声明 `calibration7`。
- 开发报告：预声明 `development8`，使用 checkpoint 中由 calibration7 得到的冻结阈值。
- 历史 test18 已参与过多轮开发，只能作为同口径回归集，不能再承担干净泛化证明。
- 真正决定是否进入检测证据路由前，应冻结新的 8–10 场 holdout，并且只评一次。
- 事件指标：10 秒窗、5 秒 stride、point NMS 半径 5 秒、1:1 匹配容差 5 秒。
- 定阈 recall floor：shot 0.90、save 0.85、set_piece 0.85；阈值目标为 floor 约束下最大 precision。
- 必报：逐类 P/R/F1/TP/FP/FN、FP/90min、macro precision、审核时长并集比例、峰值时间误差，以及 object localization 健康指标。

## 5. 实验矩阵

| ID | 共享 LoRA | 球热图损失 | 教师对齐 | 目的 |
|---|---:|---:|---:|---|
| A0 | 是 | `λ_probe` | 对齐 | 验证两项损失确实同时到达 shared LoRA，并确定 λ |
| A1 | 是 | 0 | 不加载 | 严格训练预算对照 |
| A2 | 是 | 固定 `λ*` | 对齐 | 机制 A 主实验 |
| A2-S | 是 | 固定 `λ*` | 视频内 +30 秒错位 | A2 通过后运行；排除普通正则化/额外计算带来的伪收益 |

除表中变量外，初始化 checkpoint、采样顺序、seed、epoch、optimizer step 数、LR、EMA、增强和评测协议必须相同。

### A0：梯度与 λ probe

先用 seed42、单 GPU、20 micro-steps。记录：

- `mechanism_a_event_grad_norm > 0`；
- `mechanism_a_object_grad_norm > 0`；
- `mechanism_a_shared_gradient_tensors > 0`；
- `object_ball_teacher_valid_fraction > 0`；
- 梯度余弦及 `weighted_object_to_event_grad_ratio`。

λ 的预声明选择规则：取有效测量的 raw object/event 梯度比中位数 `r`，令 `λ*=clip(0.20/r, 0.02, 1.0)`，使辅助梯度初始处于事件梯度的约 10%–30%。确定后，所有 seed 和 A2-S 都冻结同一个 λ。若中位梯度余弦低于 -0.20，先降低 λ；本轮不引入 PCGrad 等新变量。

### A1/A2：第一轮决策

1. seed42 各训练 2 epoch。
2. 只用 calibration7 选择 checkpoint 和阈值。
3. 在 development8 以冻结阈值报告。
4. 若 A2 没有方向一致的收益，停止，不追加 seed，不进入 evidence routing。
5. 若 A2 有候选收益，再补 seed43、seed44；两组都补，不能只补更好的组。
6. 如果两组 calibration 学习曲线在 epoch2 仍同步上升，只能将 A1/A2 一起扩为 4 epoch。

### A2-S：语义负对照

只有 A2 达到事件收益门槛后运行。它保留教师覆盖、质量分布和 loss 计算，但用同视频 +30 秒的球位置监督当前帧。若 A2-S 获得与 A2 接近的收益，说明收益更可能来自正则化或训练预算变化，而不是正确的检测定位语义。

## 6. 进入检测证据路由的门槛

同时满足才判定机制 A 正向：

1. **路径成立**：A0 两项梯度都到达 shared LoRA；单测证明 object residual 无法改变 logits。
2. **定位确实学到**：calibration7 的 ball heatmap loss 相对 epoch0 下降，ball top-1 hit 至少提升 5 个百分点，且教师有效覆盖没有异常塌缩。
3. **事件收益**：development8 冻结阈值下，3 seeds 的 macro precision 平均提升至少 1.5 个百分点；每类 recall 相对 A1 不下降超过 2 个百分点，且至少 2/3 seeds 同方向。
4. **统计证据**：以视频为重采样单位、seed 为外层的层级 paired bootstrap，主指标差值 95% CI 下界大于 0。不能把 clip/window 当独立样本。
5. **语义特异性**：A2-S 的收益不超过 A2 收益的 25%，或 A2 至少比 A2-S 高 1 个百分点。
6. **最终泛化**：新冻结 holdout 上方向一致；否则结论只能写成开发集候选收益。

如果仅 localization 变好而事件不变，结论是“检测监督能塑造球表征，但球定位不是当前事件瓶颈”；这不是机制 A 成功，也不应继续做证据路由。

## 7. 执行入口

```bash
# 已执行：静态检查、13 项定向测试、A1/A2 数据 dry-run
scripts/run_football_mechanism_a.sh preflight

# 单 GPU 梯度/λ probe
GPU_LIST=0 OBJECT_LOSS_WEIGHT=0.25 scripts/run_football_mechanism_a.sh probe

# 同一拓扑顺序跑严格对照与主实验
GPU_LIST=0,1,2,3 OBJECT_LOSS_WEIGHT=<A0确定的λ> \
  scripts/run_football_mechanism_a.sh pair

# 对每个 best.pt 用 calibration7 阈值冻结迁移到 development8
GPU_LIST=0 CHECKPOINT=<run>/best.pt \
  scripts/run_football_mechanism_a.sh development

# 仅在 A2 过门槛后
GPU_LIST=0,1,2,3 OBJECT_LOSS_WEIGHT=<冻结的λ> \
  scripts/run_football_mechanism_a.sh shifted
```

默认 4 GPU 时，两组统一为有效 batch 48、event/head global LR `8e-5`、shared LoRA global LR `1e-5`。launcher 会按 GPU 数重算 per-GPU LR 和 gradient accumulation，保持全局实验量不变。
