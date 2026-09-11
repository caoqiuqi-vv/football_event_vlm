# 足球关键事件 E1.3 全流程优化与实验执行方案

更新时间：2026-08-26

## 1. 目标、固定锚点与本轮边界

### 1.1 产品目标

主任务是在完整长视频中发现 `shot / save / set_piece`，DINOv3 是事件理解主模型。优化目标不是 sampled clip-val F1，而是：

- 提高严格 PointNMS、1:1 matching、`tolerance=5s` 下的事件 Precision/Recall；
- 保留足够的候选召回上限，不能用少报事件换取虚高 Precision；
- 降低跨类别、跨窗口时间并集去重后的真实人工观看时长；
- 改善跨视频泛化、低尾正样本和窗口峰值稳定性。

### 1.2 固定 Epoch0 control

固定初始化：

`outputs/football_events/vitl16_independent_preprojection_evidence_ddp_lora_v1_accelerated_resume_e1/best.pt`

历史同口径的 Epoch0 结果（Val15 sampled 调阈值，External18 exact online）：

| 指标 | 结果 |
|---|---:|
| External18 总时长 | 1029.45 min |
| micro Precision / Recall | 0.1622 / 0.7933 |
| shot Precision / Recall | 0.3139 / 0.8082 |
| save Precision / Recall | 0.0723 / 0.9617 |
| set_piece Precision / Recall | 0.1970 / 0.6396 |
| TP / FP / FN | 833 / 4304 / 217 |
| 去重人工参与 | 544.82 min / 52.92% |

该历史 operating point 的 Online-Val15 固化阈值为：`shot=0.0288709`、`save=0.00263167`、`set_piece=0.0600867`。它来自旧的 `shot R>=0.85 / save R>=0.80 / set_piece R>=0.80` 约束，只作为 Epoch0 可复现锚点；E1.3 正式判读必须复用同一 NPZ 缓存，按新混合目标离线重选阈值：shot 在 `R>=0.85` 下最大 Precision，save/set_piece 各自最大 exact F1。

评测数据层级固定为：`sampled Val15（逐 epoch 训练诊断） → Online-Val15（阈值选择与 checkpoint gate） → untouched External18（仅 Epoch0/control 和预注册候选最终审计）`。External18 不参与调阈值，也不在训练期逐 epoch 自动执行。

同口径 E1 epoch1 为 `P/R=0.1479/0.7981`、人工参与 `56.97%`。因此 E1 微调已经被证实退化，E1.3 必须从 Epoch0 control 重新开始，不继承 E1/E1.1/E1.2 权重。

### 1.3 本轮不同时引入的变量

E1.3 不加入 detector pseudo label、ball/goal auxiliary、save-conditioned-on-shot、新 LoRA depth/rank、未人审 dense FP 或 verifier。它只修复已经有直接证据的四个问题：

1. train-online 窗口语义不一致；
2. `no_evidence` 内容无关出口擦除局部信息；
3. 正窗内帧级 core/context 计数失衡；
4. 重叠窗口缺少绝对时间一致性。

## 2. 数据与 Online Sampler

### 2.1 训练数据可信度语义

事件和上下文必须分开保存：

- `human accepted GT`：可信事件类别与 reviewed anchor，产生 clip 正标签；
- `raw valid context span`：表示动作/站位上下文，只用于保护区和辅助 context，不替代 reviewed anchor；
- `raw rejected span`：不是正标签，但其区间及安全边界必须设为 ignore，禁止作为负监督；
- 未审核或类别标注不完整的视频：正标签可信，缺失类别不能自动当负标签；
- 未人工确认的 dense 高分 FP：不进入本轮 hard negative。

### 2.2 E1.3 的最小训练单元

每个 per-rank micro-batch 固定为四条记录：

```text
1. central positive：同一事件位于10s窗口的3–7s
2. edge positive：同一事件位于0.5–2s或8–9.5s
3. linked near-context：来自同一事件附近，但与所有accepted/rejected support span不重叠
4. cross-video clean：来自另一视频，且远离所有可信/拒绝支持区间
```

central 与 edge 共享唯一 `pair_id`，必须在同一 local batch；near-context 用 `online_chunk_id` 关联，但不伪装成 consistency pair；clean 必须来自不同视频，防止模型只比较场地/相机风格。

当前真实 manifest 审计：

- 4800 central/edge pairs；
- 4793 个真实 near-context，7 个找不到安全 near 的 pair 显式回退为 cross-video clean；
- 4800 个 global clean；
- 共 19200 rows；
- 4 卡每 rank 1200 micro-batches，`grad_accum=4`，每 epoch 300 optimizer updates；
- effective batch 为 `4 × 4 GPU × 4 accum = 64`。

### 2.3 Near-event 的安全定义

不能只按 anchor 点距离判负。窗口 `[w0,w1]` 必须与每个事件支持区间 `[s0,s1]` 比较：

- accepted 事件优先使用 raw valid context span；没有有效 span 时回退 reviewed anchor；
- rejected set-piece 使用 raw span，并向两侧扩展 5s ignore margin；
- near-context 优先位于目标事件前后 5–15s，但只要与任何 accepted/rejected support 相交就丢弃；
- 对目标类为负不代表对其他类为负，始终使用 per-class label mask；
- `shot-only` 可作为 save 条件负样本，但 shot 槽仍是正标签；
- save 前的 shot 上下文不得被粗暴标为三类全负。

### 2.4 当前 sampler 的定位

E1.3 是“成对窗口在线语义训练”，不是已经实现了共享 DINO 编码的 40s chunk 模型。它解决监督语义和窗口一致性，但不会宣称已经验证共享解码/共享特征。真正 40s chunk 一次解码、96帧只编码一次属于后续工程提速 E1.4，必须在 E1.3 证明目标有效后再做。

## 3. 模型结构

### 3.1 固定主干

- DINOv3 ViT-L；
- last 6 blocks LoRA，`r=8, alpha=16`；
- 24 frames / 10s / 512×896；
- class-specific independent preprojection evidence；
- 本轮不切 last8/r8，避免把 LoRA depth 与 sampler/loss 混在一起。

LoRA/backbone 使用保护性低学习率 `2.5e-7/GPU`（4卡 global `1e-6`），避免重复 E1.2 从强 checkpoint 继续高 LR 后跨视频退化。Evidence local/temporal 为 `6.25e-6/GPU`，classifier 为 `9.375e-6/GPU`，让新路由能学习但不快速改坏已学表示。

### 3.2 no_evidence 结构修复

旧结构把每类/每query的可学习全局 scalar 与约 1792 个 patch 一起 softmax，并用 null value 替换 appearance。负样本可以通过调高一个内容无关 scalar，让 appearance 变常数、motion 变零，再依赖 global CLS 完成分类。

E1.3 改为：

1. patch attention 始终只在真实 patches 内条件归一化；
2. appearance 始终由真实 patch values 加权得到，null mass 再高也不能擦除内容；
3. null 只作为内容依赖的 evidence-presence gate/诊断，不参与 appearance 替换；
4. 内容 gate 使用 class/query patch-score 的轻量显著性统计，不使用可自由升高的全局出口；
5. 旧 null parameters 仅为 checkpoint shape 兼容保留并冻结；
6. 新 gate 必须小于 1k 参数，禁止为 6 个 gate 新增百万级随机 MLP。

硬测试：强制 `null_mass > 0.99` 后扰动输入 patch，`patch_appearance` 仍必须变化。

### 3.3 防止 global 分支继续绕过 local

首轮先记录而不强加新的 causal loss：

- appearance 输入敏感度；
- event/background 的 slot-weighted null mass；
- patch attention entropy；
- positive clip 的局部证据消融 logit drop（可离线小批量计算）。

若 E1.3 的局部指标改善但长视频排序不改善，下一轮 E1.4 再加入轻量 counterfactual：保持 global 不变，将 local appearance 替换/打乱，要求 accepted positive 的原始 logit 高于 erased logit；负样本只做稳定性约束。不能在 E1.3 首轮同时加入，以免无法归因。

## 4. Loss 设计

### 4.1 Clip loss

主 clip BCE 保留 class-specific multi-label 形式。E1.3 使用 per-class positive/negative 诊断，不把 save/shot 改成 softmax 互斥。

已知风险：per-rank batch 只有一个主事件类，本地 `per_class_equal_pos_neg` 不是严格的 DDP-global balance。首轮保持现有 clip loss，避免再叠加一项核心变量，但必须记录每 rank / global 的正负 slot 数。若出现 rank-local class starvation，E1.4 单独实现 DDP-global clip balance。

### 4.2 Balanced no-evidence loss

对每类分别构造：

- event core：Gaussian target `>=0.5`，目标 null mass 为0；
- background：target `<=0.1`，目标 null mass 为1；
- 0.1–0.5 Gaussian shoulder：ignore。

只有 DDP 全局同时存在 event/background 的类别才参与；两侧先按全局 slot count 归一，再 1:1 等权。DDP 实现使用 `local_sum / all_reduced_global_count × world_size`，之后 DDP 的梯度平均恰好得到全局均值。

日志不再平均空 batch 的0值。必须累计 numerator 与 valid slots，再计算：

- `null_mass_event_slot_weighted`；
- `null_mass_background_slot_weighted`；
- 每类 numerator/slots；
- gate content-logit std。

### 4.3 Frame core/context balanced heatmap

正 clip/class 内：

- core (`target>=0.5`) 与可信 context/background (`target<=0.1`) 各自归一后 1:1；
- 0.1–0.5 shoulder ignore；
- edge positive 保留较弱直接监督：clip weight 0.35、frame weight 0.5；
- central full weight；
- 全负 clip 保持普通 focal BCE。

这不是继续无上限提高 `frame_pos_weight`，而是修复 24 帧中少数正帧被计数稀释。必须报告 core/context loss 与有效 slot 数。

### 4.4 Central-to-edge consistency

- central 是 stop-gradient teacher；
- edge 是 student；
- 只约束 pair 的可信正类别；
- clip logit 使用 SmoothL1，权重0.05；
- response/frame curve 按绝对 `frame_times` 插值，只在窗口重叠且 anchor±2s内约束，权重0.01；
- 前0.25 epoch warm-in；
- 每 batch 必须发现合法 pair，否则立即报错，禁止静默 loss=0。

该 loss 保证 peak 随窗口平移，而不是要求两个窗口的相对帧下标相同。

## 5. 训练流程与门禁

### 5.1 启动前门禁

必须全部通过：

1. `py_compile`；
2. central/edge/near/clean sampler 单测；
3. anti-erasure、balanced null、frame core/context、consistency 单测；
4. 四 rank 首 batch 组成断言；
5. init checkpoint 审计：除新轻量 gate 参数外，不允许 missing/unexpected/shape mismatch；
6. 20–40 step GPU smoke：无 NaN、无空 loss、四卡利用率正常、显存稳定；
7. 确认 LoRA trainable 参数只在 last6/r8，旧 null scalar/value 不可训练。

### 5.2 首轮 E1.3 gate

- GPU：4,5,6,7；
- 1 epoch，300 optimizer updates；
- effective batch 64；
- init：固定 Epoch0 last6/r8；
- 只保存 `best.pt`、`last.pt`，不保留每个中间 epoch；
- 首轮预计训练约45–55分钟。

训练中在 step40/80/160/epoch-end检查：

- total / clip / frame / consistency loss；
- positive P10、negative P90、tail gap；
- slot-weighted event/background null mass；
- central/edge logit gap和绝对时间 response gap；
- 每类正负 slot 数；
- LoRA、evidence、temporal、head 的 grad norm；
- sampled-val AP/AUROC/frame-topk 仅作诊断。

若在前80–160 step出现以下任一情况立即停：

- positive P10持续下降且 negative 仅被整体压低；
- event/background null mass同时趋近1或同时趋近0；
- consistency pair/slots为0；
- 新 gate content-logit std趋近0；
- LoRA grad为0或大于 evidence/head 一个数量级并伴随 Val 退化。

### 5.3 epoch1 后的决策

- 长视频主指标、tail gap、峰值稳定性同时改善：继续相同配置至3 epoch；
- sampled loss改善、长视频不改善：停止加训，检查阈值迁移与局部因果利用；
- null/局部指标改善、排序不改善：E1.4做 local counterfactual，不再调 sampler；
- shot改善但save不改善：下一轮做 save-conditioned-on-shot / save-only、shot-only cohort；
- set_piece仍差：单独采用长上下文 clip head与 subtype-balanced supervision，不强迫它服从单点 dense peak；
- 所有类都不改善：回到跨视频标签完整性/domain shift，不扩大 LoRA rank或堆新时序头。

## 6. 评测协议与提速

### 6.1 阈值策略（2026-08-26更新）

阈值只能在 Online-Val15 上确定，External18 untouched：

- shot：精确搜索 `max Precision s.t. Recall >= 0.85`；
- save：不设 recall floor，按原始 exact tuned F1 最优点；
- set_piece：不设 recall floor，按原始 exact tuned F1 最优点；
- 禁止把 save/set_piece 的 floor 写成0后继续使用“max precision”目标，这会退化为只报极少候选；
- 同时报告三类完整 PR、candidate recall ceiling、P@R85/P@R90（若可达）。

工作阈值不强求 save/set_piece recall，但实验验收仍报告真实 recall、AP 和 ceiling，防止靠召回坍塌获得虚高 precision。

### 6.2 主事件协议

- 10s window / stride5s；
- candidate time 优先使用 response/frame peak；
- PointNMS radius 5s；
- 1:1 GT matching；
- tolerance 5s；
- window-overlap 只用于 UI coverage 诊断，不作为模型 Precision 主指标；
- 人工参与时间按所有类别、所有重叠候选的 capped10 区间做时间并集，不重复计时。

### 6.3 评测分层，避免每 epoch 浪费35分钟

训练期分三级：

1. **每 epoch轻量诊断**：sampled Val15，只输出汇总 AP/AUROC/tail/frame-topk，不打印逐视频；预计3–5分钟。
2. **Online-Val15校准**：epoch1、epoch3及出现明确 sampled-Val 新高的 checkpoint 才运行；四卡DDP、batch4，原始预测写 NPZ。阈值、NMS、人工预算全部离线重算，不重复视频解码；预计16–17分钟。
3. **External18最终测试**：只评 Epoch0 control 和预注册候选（E1.3 e1、后续最佳），绝不每 epoch运行；四卡DDP，预计18分钟。任何阈值策略变化直接复用缓存。

进一步提速：

- Val/External 使用 DistributedEvalSampler，不允许每rank重复完整数据集；
- 控制台只打印单行汇总，不打印每视频；
- `persistent_workers` 只在多epoch同一loader时启用；独立一次性评测关闭，避免僵尸worker；
- decoder worker保持3/GPU，data time已经显著低于forward，不盲目扩大worker；
- batch只在显存和吞吐实测后从4增大，不能仅看推理显存估计训练batch；
- 缓存以 checkpoint fingerprint + split + window/stride + model score source 命名，防止误复用。

首轮 E1.3 完整 gate 预计：训练45–55min + sampled Val3–5min + Online-Val15约17min + External18约18min，总计约83–95min。后续普通 epoch 不跑长视频时约48–60min。

## 7. E1.3 验收表

相对固定 Epoch0 control：

| 维度 | 通过条件 |
|---|---|
| shot | Online-Val约束点可达R85；External P/R不劣于control且P提高优先 |
| save/set_piece | exact tuned F1与AP改善；真实Recall单独报告，不允许隐藏 |
| micro | Precision至少+2pp，或在相近事件召回下人工参与下降至少10% |
| candidate ceiling | 任一类下降不超过1pp |
| tail | positive P10 - negative P90改善 |
| spatial route | event null mass低于background，appearance输入敏感度通过 |
| peak | central/edge绝对时间误差、duplicate ratio下降 |
| 泛化 | worst-20%视频不继续恶化；最终用视频级bootstrap CI |

单个 sampled-val 数字、总 loss下降或 tuned threshold升高，均不能单独宣告实验成功。

## 8. 后续路线（严格串行，不与E1.3混开）

1. E1.4：local counterfactual causal utilization，验证真实 patch evidence 是否被事件头使用；
2. E1.5：DDP-global clip pos/neg balance，前提是E1.3日志证明rank-local starvation；
3. Save专线：shot history作为条件输入，配合save-only/shot-only cohort，不把shot当save捷径；
4. Set-piece专线：长上下文 clip head + corner/freekick/penalty/kickoff subtype辅助，不强迫单点peak定义；
5. Football evidence：ball/goal pseudo heatmap + token routing，仅在当前局部appearance仍缺乏可见实体证据时启动；
6. Representation retention：若LoRA继续表现为train提升/Val下降，再增加冻结teacher feature retention或进一步降低LoRA LR；
7. Verifier：主模型候选召回达到目标后，用候选级统计/局部特征压剩余FP，不替代DINO主模型。
