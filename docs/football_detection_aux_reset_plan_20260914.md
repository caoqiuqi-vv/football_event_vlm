# 足球事件检测辅助路线重启方案

日期：2026-09-14
状态：方案冻结稿；尚未修改训练代码，尚未启动新实验。

## 1. 决策摘要

下一阶段不继续叠加新的检测/ROI/motion 分支，也不直接延长 object-motion v9。工作拆成三个阶段：

1. **方案与协议冻结**：明确唯一主假设、对照、指标、数据边界和停止条件，并统一最新实验的源码与产物来源。
2. **代码审计、兼容性重构与开发**：先补可观测性和强类型接口，再逐步拆分主训练器；保留旧入口、checkpoint 键和历史实验复现能力。
3. **新实验**：先做“训练期检测监督塑造共享表征”的单变量实验，只有它通过后才做检测引导 token 路由，再决定是否投入交互状态标注和高帧率局部分支。

本轮真正要回答的问题是：

> 在推理时不依赖外部检测器的条件下，训练期对象定位监督是否能让与事件任务共享的 DINO 适配特征获得增量判别能力，并在固定事件召回下减少完整长视频误报？

“检测头本身更准”“窗口 AP 上升”“融合分支学到了非零残差”都不足以回答这个问题。

## 2. 已有证据与当前判断

### 2.1 已验证的事实

- 早期 object-teacher 实验只训练约 316 万个辅助头参数，原时序头和 LoRA 未更新；保存 clip mAP 与对应基线约为 `0.57579` 和 `0.57599`，不能证明检测监督改善了主表示。
- B1 epoch1 在统一高召回重评中，对 set-piece 有内部收益信号，但 shot 退化、save 改善很小；它证明的是自动证据投影可能对个别类别有用，不是检测监督塑造骨干的证据。
- 自动轨迹增强把事件附近球证据可用率从约 37.6% 提高到约 77.1%，但轻量几何读出仍未超过仅分数对照，说明 coverage 提升不等于事件增益。
- ROI/dynamic ROI/quality gate/object motion 多条路线没有稳定正收益；共同问题包括局部证据噪声、任务相关性不足、融合改变 recall、缺少严格同预算 control。
- object-motion v9 的最佳 epoch3 在 3 个 sentinel 视频上仅达到 micro P=5.36%、R=58.53%，没有达到 shot/save/set-piece 90%/85%/85% 的召回目标；本轮没有匹配 control，也未保存 reference/shared/fused 三路独立预测。
- v9 的 relation loss 有下降，但共享主分支变化很小，frame residual 全程为 0；因此“辅助任务学会了”与“主任务得到增益”明显脱节。

### 2.2 当前工程风险

- 当前 `train_football_events.py` 为 20,877 行，`VideoEventClassifier` 同时承载大量互斥分支，训练器也直接组合大量 loss；继续加条件分支会让对照和 checkpoint 语义越来越难核实。
- `football_object_motion/train.py` 通过运行时 monkey patch 替换数据、模型、loader、optimizer、teacher 和 loss；object-motion loss 甚至复用了名为 `object_teacher_heatmap_loss` 的通用调用点。功能可运行，但依赖关系、配置语义和审计边界不透明。
- v9 保存配置中的 `event_context_enabled`、`event_frame_fusion_enabled`、`event_relation_grad_enabled` 在当前工作区源码中没有对应引用。正式审计前不能假设这些开关实际生效。
- 最新 v9 日志显示训练源码来自 `/mnt/data_7t/qiuqi/dinov3-det/.runtime/football_event_vlm/`，其中包含当前工作区缺失的 v8/v9 测试和 dense sampling 实现。当前工作区分支为 `codex/source-only-import-20260911`，不能在未对齐源码前宣称可复现 v9。
- `val15` 命名与实际 3 个 sentinel 视频不一致，容易让实验支持范围被误读。

### 2.3 对“为什么没有收益”的优先解释

按当前证据，优先级从高到低是：

1. **因果变量混杂**：检测监督、采样、融合、可训练范围、hard negative 和时序结构经常同时变化。
2. **辅助目标与事件边界不一致**：球/球门“在哪里”不足以区分射门与传球、扑救与普通接球；交互和状态变化更接近事件语义。
3. **梯度没有作用到需要改善的表示**：部分实验冻结或 detach 了主干，检测任务只能改善读出头或独立分支。
4. **主任务消费路径太弱或不可观察**：残差过小、frame 路径关闭、局部证据没有进入原时序层，都会让辅助任务与最终事件预测脱节。
5. **教师与时序采样质量不足**：漏检、时间错配、身份切换和 patch16 小球分辨率上限会把辅助监督变成噪声。
6. **评测协议不足以支持结论**：窗口 AP、三视频 sentinel、验证集重新定阈和缺少匹配 control 不能回答完整长视频上的净增益。

梯度冲突是待测假设，不预先认定为根因；PCGrad 等方法只在实际测到持续冲突后进入消融。

## 3. 目标架构：把两种机制彻底分开

后续把“检测辅助”拆成两个独立机制，不能在首轮同时开启。

### 3.1 机制 A：训练期表征塑造（首要实验）

外部检测器只产生训练目标；推理时辅助头可删除，事件模型不读取 detector box、轨迹或置信度。

```text
RGB frames
   -> frozen DINO base + trainable shared adapter/LoRA
      -> global/frame tokens -> existing temporal event head -> event logits
      -> object localization head -> ball/goal/person targets (train only)
```

梯度拓扑必须显式固定：

|参数组|事件 loss|对象 loss|说明|
|---|---:|---:|---|
|冻结 DINO base|否|否|保持 anchor 可恢复|
|共享 adapter/LoRA|是|是|这是待验证的核心机制|
|事件时序层/分类头|是|否|不让对象 loss 直接训练分类器|
|对象定位 head|否|是|推理可删除|

首轮不增加 ROI crop、轨迹几何、cross-attention 或事件 residual。相同结构、参数量和训练预算的 RGB-only adapter 是硬对照；两组唯一差异应为对象监督是否反传到共享 adapter。

### 3.2 机制 B：检测引导的证据路由（机制 A 通过后）

由学生自己的定位头产生多候选与 NULL token，从原始 patch 中抽取局部核心和上下文 token，再送入原事件时序层。外部 detector 仍不参与推理。

```text
global frame token ------------------------------+
                                                  -> original temporal head -> logits
student object queries -> K candidates + NULL ---+
                         -> core/context patches -+
```

约束：

- 不使用硬单峰或必须有球的规则；保留候选置信度、可见性、时间和位置。
- 球候选附近必须包含动作球员/门将的上下文，不能只输入球像素。
- 无证据时严格回退全局路径；新增融合采用零初始化或显式 anchor residual。
- shot/save 与 set-piece 分开建模和验收，不共享相同强度的 veto 策略。
- 与普通 learnable query/local token 的同预算对照配对，防止把更多 token 或参数带来的收益归因给检测。

### 3.3 机制 C：交互状态监督（前两步仍不足时）

对象存在性不能稳定提升事件时，才加入可观察交互状态：触球/脱离、门将响应、球速或方向变化、定位球准备到执行。先完成已有 180 个片段的人工可见性、轨迹和交互锚点，保持 unknown，不把自动漏检当负例。

首轮对照保持为：同预算局部 RGB、检测引导局部 RGB、检测引导加交互监督。若只有 RGB 组提升，收益不能记到检测辅助上。

## 4. 阶段一：方案与实验契约冻结

### 4.1 单一主指标

主指标使用完整长视频的一对一事件匹配，不使用窗口 AP 代替：

- calibration 视频选择每类阈值；evaluation 视频只使用冻结阈值。
- 默认延续最新业务召回下限：shot 90%、save 85%、set-piece 85%；同时报告 R80/R85/R90 曲线。若业务目标要调整，只能在首个新实验前调整一次。
- 主结果报告每类 precision、recall、FP/90min、TP/FP/FN、事件时间误差，以及 micro/macro 汇总。
- 以视频/比赛为配对单位 bootstrap；窗口不能当独立样本计算显著性。
- 三视频 sentinel 只作为运行和退化筛查，不作为模型选型集合。

### 4.2 数据边界

- 固定 train/calibration/evaluation video IDs、annotation hash、媒体 hash 或稳定指纹。
- 对标注不完整的类别保留 unknown mask；未标注窗口不能自动转为可信负例。
- hard FP 必须经人工确认并远离任意真实事件，而不只是远离同类事件。
- teacher target 记录 source、时间差、置信度、插值/跟踪状态和 unknown；训练日志必须报告真正有效监督的分母。
- 当前反复使用的 15 视频只能作为开发集；正式确认至少留出按比赛隔离、未参与结构选择的一组视频。

### 4.3 每个 run 的最小 provenance

每次运行必须原子保存：

- source tree/commit/hash、启动命令、完整解析后配置及配置 schema 版本；
- checkpoint 路径与 SHA256、实际加载/缺失/shape mismatch 参数；
- split/annotation/teacher-index 指纹；
- world size、batch、gradient accumulation、optimizer steps、采样覆盖率；
- reference/shared/fused 三路预测及其 frame-level 输出；
- raw/weighted loss、每参数组梯度范数、事件与辅助梯度 cosine；
- 阈值来源、NMS、容差和 score 语义。

配置出现未知键时默认失败，禁止静默忽略。

## 5. 阶段二：审计、重构与开发

不进行一次性大搬迁。采用兼容层包住旧入口，再逐块迁移；每一步都要求数值回归。

### R0：统一源码事实

1. 固化 v9 实际 runtime 源码、测试和 launcher 的只读快照及 SHA256。
2. 对比当前工作区、runtime tree 和 checkpoint provenance，列出缺失/漂移文件。
3. 明确后续唯一开发源；历史运行源码只归档，不原地改写。
4. 为每个已有关键实验建立 ledger：假设、唯一变量、源码、数据、checkpoint、指标、结论强度。

未完成 R0 前，不修改 object-motion 主实现，也不启动 v10。

### R1：先补契约和可观测性

建议新增职责边界：

```text
football_events/
  contracts/
    batch.py            # FootballBatch、三态 targets/masks、时间语义
    outputs.py          # reference/shared/fused/frame/aux 标准输出
    config.py           # schema、未知键拒绝、跨字段校验
  multitask/
    model.py            # anchor + shared adapter + event path
    auxiliary.py        # AuxiliaryTask 接口和 registry
    localization.py     # 训练期对象定位任务
    loss_graph.py       # 显式 loss/参数组路由
    diagnostics.py      # 梯度、分支差值、teacher 覆盖率
  evaluation/
    spotting.py         # 唯一完整视频事件匹配口径
    paired.py           # 视频级配对统计
```

旧 `train_football_events.py` 和旧脚本先调用兼容 adapter，不立刻更改历史公开导入。

标准模型输出至少包含：

```text
reference_logits/reference_frame_logits
shared_logits/shared_frame_logits
fused_logits/fused_frame_logits
features: global/frame/patch/local
auxiliary: task-name -> predictions/quality/validity
```

输出命名必须反映真实语义，禁止再用 object heatmap loss 的调用点承载 object-motion 综合 loss。

### R2：去除运行时 monkey patch

- 用显式 `DataPipeline`、`ModelFactory`、`OptimizerFactory`、`AuxiliaryTask` 注册代替 `install_hooks()`。
- `forward_model_batch` 不再靠临时写入模型 `__dict__` 传 motion 输入；统一由结构化 batch 传参。
- loss composer 从 trainer 中拆出，逐项声明 inputs、weight、active masks 和接收梯度的参数组。
- 配置 schema 对未消费字段报错，并输出最终生效配置；curriculum override 同样校验。

### R3：拆分主模型但保持 checkpoint 兼容

按以下顺序拆 `VideoEventClassifier`：

1. backbone feature extractor；
2. frame projection 与 frame head；
3. temporal encoder 与 clip head；
4. view/local evidence fusion；
5. auxiliary heads。

优先保持原 state_dict key；必须改名时提供显式映射和双向验证，不用 `strict=False` 掩盖关键缺失。

### R4：训练器拆分

- dataset/sampler、loss graph、optimizer/scheduler、evaluation、checkpoint/resume 分开。
- 标签 schema 不再依赖全局可变 `LABELS`。
- checkpoint selection 只消费标准化完整视频指标；window AP 留作诊断。
- 每个 optimizer step 记录实际采样类型和数据覆盖，避免“4 epochs”被误解为 4 次全量遍历。

### R5：必须通过的工程验收

- 关闭所有新模块时，旧 checkpoint 的 logits、frame logits 和最终事件结果数值一致，容差预先固定。
- 同 seed 下首批及跨 epoch 的 frame indices、targets、label masks、teacher masks 一致。
- 单卡与 DDP 的有效样本、gradient accumulation、optimizer step 和尾部 batch 语义一致。
- 旧 checkpoint load、resume、best/last 选择和中断恢复回归通过。
- reference/shared/fused 三路输出可独立评测；空证据精确回退 reference。
- 每个启用的 loss 都有非零有效支持；每个预期可训练参数组都能测到梯度；禁止死配置键。
- 辅助任务关闭后无额外推理依赖；训练期 detector 不进入部署图。
- 小规模真实视频 smoke 与合成测试都通过；仅合成测试不作为完成依据。

## 6. 阶段三：最小实验矩阵

### E0：锚点重评，不训练

- 在同一源码和固定协议下重评当前选定 anchor。
- 对 v9 checkpoint 输出 reference、shared anchor、full fused 三路结果，量化收益/退化来自 BallLoRA 共享分支还是 motion fusion。
- 用现有 prediction cache 分析 NMS 前后 recall、时间误差和候选稳定性。

只有 E0 的 anchor 达到可用召回并且三路语义核实后，才进入训练实验。

### E1：共享 adapter 的 RGB-only 容量对照

- 从同一 anchor 初始化；冻结 DINO base。
- 训练 shared adapter/LoRA、原时序层和分类头，只使用事件与现有 frame-event 监督。
- 参数量、帧数、分辨率、更新步数与 E2 完全一致。

它回答“增加可训练表示与继续事件训练本身是否有收益”。

### E2：训练期对象定位辅助

- 与 E1 唯一差异：对象 loss 反传到 shared adapter 和独立定位 head。
- 事件 head 不读取预测 heatmap、框、轨迹或 detector confidence。
- 先以 ball 为主；goal/person 分开报告有效支持和独立损失，不把关闭项的零指标解释为失败。
- 目标采用 positive/trusted-negative/unknown 三态；按真实有效 mask 归一化。

它回答本轮核心问题：“检测监督是否改变了对事件有用的共享表示”。

### E2-S：错时教师对照

保持同视频、同场景边际分布，把对象监督做受控时间移位。若 E2 不优于 E2-S，模型很可能利用的是场景/来源捷径，而非正确的对象时序信息。该对照只作依赖性诊断，不宣称严格因果证明。

### E3：学生定位引导 token 路由

仅当 E2 同时满足定位和事件增益后运行：

- E3-C：相同数量的普通 learnable local queries，无检测监督；
- E3-D：学生对象 queries + core/context tokens；
- 两组使用相同参数、帧预算、时序层、融合和训练数据。

首轮不加入人工轨迹、hard FP ranking、PCGrad 或更高分辨率 crop，避免再次混变量。

### E4：交互状态与高帧率局部动作

仅当 E3 表明定位选择有价值但仍不足以区分 hard FP 时运行。按已有 C0/C1/C2 方案比较同预算局部 RGB、检测引导局部 RGB、检测引导加交互监督。先做 1–2 epoch pilot，胜出组再跑至少 3 seeds。

## 7. 梯度与优化策略

首轮只保留以下损失：

```text
L = L_event + lambda_frame * L_frame + lambda_obj * L_object + lambda_ret * L_retention
```

- `L_event` 始终是主损失；不在首轮同时加入 relation、coordinate、consistency、dense-rank、gate-budget、saturation 等完整 v9 组合。
- 在固定诊断 batch 上分别对 `L_event` 与 `L_object` backward，记录 shared adapter 的梯度范数与 cosine。
- `lambda_obj` 按共享参数上的梯度比例确定，使对象梯度成为受控辅助量，而不是按 loss 数值拍权重；选定后整轮固定。
- 只有观察到持续负 cosine、且 E2 的定位改善但事件退化时，才增加 PCGrad/gradient projection 的单独配对消融。
- retention 保护 reference 正例尾部，但不能靠过强约束让新分支等同零更新；同时报告 shared-reference 激活差值和 logits 差值。

## 8. 继续与停止条件

### 8.1 表征阶段通过条件

E2 必须同时满足：

1. 在按比赛隔离的人工定位诊断集上，定位指标显著优于冻结特征读出；训练伪标签 top1 hit 不算定位验收。
2. 对 E1 的固定长视频协议，在各类召回达到预定下限且单类 recall 不下降超过 1pp 时，目标类别 precision 提高至少 3 个百分点，或 FP/90min 降低至少 15%。
3. 增益不能只来自 set-piece 汇总掩盖 shot/save 退化；每类单独报告。
4. reference/shared/fused、梯度和有效监督日志能解释变化来源。

若定位不改善，停止事件训练，先处理监督质量、时间对齐或图像分辨率。若定位改善但 E2 不优于 E1，判定“对象定位辅助对当前事件表示无增量”，不继续增大 loss、epoch 或 LoRA 范围。

### 8.2 路由阶段通过条件

E3-D 必须优于同预算 E3-C，并在自然无球、遮挡、高球、灯具/广告牌、门将普通移动等分层中不出现明显依赖崩溃。若 E3-C≈E3-D，收益来自局部 token/容量而非检测路由。

### 8.3 最终确认

单种子 1–2 epoch 只作筛选。进入正式结论前：

- 至少 3 seeds；
- 阈值只在 calibration 拟合；
- evaluation 比赛未参与方案选择；
- 视频级配对区间与逐比赛差值完整报告；
- 报告训练与推理成本、显存、吞吐和 teacher 生产成本。

## 9. 建议执行顺序

1. 完成 R0 源码/产物对齐和实验 ledger。
2. 完成 E0 三路只读重评，决定唯一 anchor。
3. 实施 R1/R2，先解决标准输出、配置校验、loss/梯度可观测性和 monkey patch。
4. 按回归测试推进 R3/R4，不做与新假设无关的大规模文件整理。
5. 实现机制 A，运行 E1/E2/E2-S 的小规模工程 smoke。
6. 用户确认 GPU 和正式数据条件后，再启动 1–2 epoch pilot。
7. 只有 E2 过门槛才开发并运行 E3；只有 E3 显示定位路由有增量才进入交互状态 E4。

这个顺序把“模型是否用到了检测信号”“信号是否改善了共享表示”“定位是否能选择更有用的局部像素”“交互语义是否必要”拆成四个可以被分别否证的问题。
