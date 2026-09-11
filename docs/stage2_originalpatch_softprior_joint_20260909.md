# Stage2：原始 patch、软位置先验与联合时序适配

这是当前证据下优先级最高的一组探索配置，不是已证明最优或必然正向的方案。正式实验未启动，旧实验继续暂停。

## 为什么不硬裁上下各 20% / 25%

统计 Stage1 全量清单中权重有效的足球自动参考框中心：训练 128 个视频、2,719,580 帧；验证 15 个视频、254,669 帧。

|过滤区域|训练参考框被排除|验证参考框被排除|
|---|---:|---:|
|顶部 20%|1.94%|1.23%|
|底部 20%|6.35%|7.93%|
|上下各 20%|8.28%|9.17%|
|顶部 25%|4.71%|3.32%|
|底部 25%|8.61%|12.09%|
|上下各 25%|13.32%|15.41%|

统计对象是含噪声的自动参考框，并非人工真球标注；不能把被排除比例解释为真实召回损失。相邻帧相关，统计不是独立样本显著性检验。验证分布仅用于诊断，不据此搜索最优边界。

已知灯具峰 y≈152/720≈21.1%，硬裁顶部 20% 本身不能覆盖它。硬裁顶部 25% 能去掉这一位置，也会去掉合法高球。底部还可能出现近景球、脚部动作和开球位置，当前不采用底部降权。该案例属于校准视频，不能把其错误参考当训练纠正标签。

对已知帧 native_frame=31868（1062.2667 秒）用实际 720P 事件预处理和 Stage1 模型复测：无位置先验与本方案均得到候选 (104,152)、(744,248)，两个候选没有变化。不能声称软先验已经修复这帧；第二坐标也未经人工确认是真球。该结果说明灯具之外的备选本来就存在，本轮优先解决候选内容的辨别和时序融合，而非把软先验当主要修复。详见输出目录 known_lamp_prior_probe.json。

位置先验不能识别“灯还是球”，也不能修复 Stage1 伪标签；背景错误可能出现在画面任何位置。它在本实验中只负责候选分配。真正需要验证的是原始外观、上下文和事件监督能否让网络合理使用或忽略候选。

## 唯一主实验：seed42

|项目|配置|
|---|---|
|输入|1280×720，16 帧，10 秒窗口；保持原时间采样和数据划分|
|原事件权重|720p_fromlast_e8_20260829/best.pt 对应的只读 source_720p_best.pt 快照|
|定位指导|Stage1 epoch2 定位头和微调后 DINO，全部冻结|
|局部特征来源|原事件模型 DINO 的原始 patch，全部冻结；不是 Stage1 微调后 patch|
|候选 0|原始热图全图最大峰，无高度限制|
|候选 1|先抑制候选 0 周围 Chebyshev 半径 3 patch，再最大化 logit + log(height_weight)|
|高度权重|y/H≤0.25 时 0.25；0.25→0.40 线性升至 1；其余为 1；永不为零|
|每候选 token|精确 3×3 原始 patch（9 个）＋11×11 邻域 ROIAlign 2×2 网格（4 个）|
|元信息|位置、相对位置、尺度、核心/上下文类型、原始热图相对响应、帧时间、候选编号|
|融合|原帧 token 查询前/当前/后帧局部 token，加入原时序 Transformer 输入；残差输出零初始化|
|缺失|NULL token；无须球门同时出现；不按峰值强制判“有球”或串成轨迹|
|优化|epoch1 仅适配器；epoch2–6 适配器＋原 4 层时序 Transformer＋分类头|
|学习率|适配器 2e-4；时序/分类头 2e-5；余弦衰减；AdamW，weight decay 0.01|
|batch|训练/评测 64；缓存每卡 2 个解码 worker、每次前向 8 帧|
|增强|整窗局部证据丢弃 0.15、帧丢弃 0.20、候选丢弃 0.10|
|约束|残差 L2 权重 0.01，跨窗口局部证据错配一致性权重 0.05|
|训练顺序|只运行此配置 seed42；获得可靠开发收益后，再显式运行同配置 seed43|

顶部权重 0.25 只相当于 logit 减约 1.386，强错误响应仍可能入选。保留全图候选防止把高球硬排除，也意味着灯具候选可能继续存在，必须由内容和事件上下文判断。所有相对响应描述仍来自未改写的热图，权重不被包装成存在概率。

全局分支始终参与。显式全无效候选时使用独立冻结原事件模型的 logits 和阈值精确回退；自然无球时的热图仍可能有峰，需要后续真实片段分层验证，不能用显式全空测试替代。

DINO 在本轮只参与新缓存的前向提取，不参与反向微调。事件时序层和分类头在预热后实际更新。这把定位信息送入事件特征形成过程，同时保留原事件模型的特征基础。旧核心缓存使用了 Stage1 改动后的 patch，且没有保存任意新位置的完整特征图，因此不能直接复用。本次仅复用完整的原全局特征/原 logits 缓存，并逐值校验提取一致性。

## 运行命令

工作目录 `/home/new_users/qiuqi/code/dinov3-main`。配置已 prepare，正式缓存尚未开始。以下命令供用户决定启动时执行。

```bash
cd /home/new_users/qiuqi/code/dinov3-main
/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python scripts/run_football_stage2_position_prior.py --phase preflight
/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python scripts/run_football_stage2_position_prior.py --phase pipeline --seed 42
```

pipeline 顺序完成新缓存→数组整理→六轮训练及评测→最终报告。缓存阶段配置 GPU 3、4、6、1，联合训练使用 GPU 3；只在指定卡空闲时启动，否则明确报错，不停止其他任务。GPU 编号使用服务器物理编号。中断后重新执行同一 pipeline 命令复用已提交窗口与 epoch checkpoint；中断的 epoch 从上一完整 epoch 重跑。

如果缓存已完整，希望只执行训练及报告：

```bash
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python scripts/run_football_stage2_position_prior.py --phase train --seed 42
/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python scripts/run_football_stage2_position_prior.py --phase report --seed 42
```

预留约 76 GiB 新空间，保留新逐窗缓存和整理后的数组，不清理旧实验。依赖代码、权重和基础数组被哈希固定；训练前验证新数组哈希，发生改动则停止，避免混用结果。若需要改配置，请使用新输出目录和对应的 --config 重新 prepare，不修改已启动配置。

只有决定复现时才执行 `--phase pipeline --seed 43`；它复用此配置缓存，不自动运行其他消融。

推理入口为 `football_stage2_position_prior.PositionPriorEventModel(checkpoint).cuda().eval()`，输入 uint8 `[B,16,3,720,1280]` 和真实帧时间。不要使用旧 JointTemporalEventModel 读取本实验权重，因为它调用旧候选提取器。

## 完成标准与结论边界

输出目录：`outputs/football_localization_stage2/720p_originalpatch_softprior_joint_20260909`。

- `joint_temporal_seed42/epoch_*.json`：逐轮校准窗口 AP、P/R、梯度、时序更新状态。
- `joint_temporal_seed42/best.pt`、`resume.pt`：所选及末轮模型。原模型 epoch0 也参与校准选择，不能把分支被关闭解释成新模型有效。
- `FINAL_SUMMARY_seed42.json`、`FINAL_REPORT_seed42.md`：完整开发评测、相对原模型的视频级配对 bootstrap。
- `KNOWN_CASE_CANDIDATES_seed42.json`：已知校准灯具片段的原始双峰/先验双峰坐标；不作真实定位精度判定。

模型和阈值仅在校准集选择。开发检查预先固定为：训练模型被选中、窗口宏 AP 和宏 precision 提升、各类 recall 相对原模型不降超过 1 个百分点且误报窗口/小时不增加、视频 bootstrap 宏 AP 差的 95% 区间下界大于 0、半帧缺失与跨窗错配的各类 P/R 不降超过 1 个百分点、显式空输入精确回退。原阈值结果另行完整报告，避免只展示重新调阈值的收益。

所选模型和末轮启用模型都评测空输入、半帧缺失、跨窗错配、局部时间反转。自然无球、灯具干扰、高球、近景的鲁棒性仍需人工核验分层；合成测试不替代它们。

这是历史开发视频上的单种子探索，AP 为窗口 AP，不是事件 spotting mAP。新方案同时改变特征来源和时序适配，单组成功不能单独证明高度先验有效；用户不要求继续的“无定位、仅时序微调”对照不运行，不能把全部收益归因于定位信息。后续若要证明先验的独立贡献，只需追加完全相同配置、position_prior.mode=none 的配对对照，必须重新提取相应 patch。先验参数不根据这个已知校准坏例反复搜索。

工程验证记录见 `VERIFICATION_COMPLETE.json`；12 窗口、2 轮的验证结果不构成正式实验的正向证据。

## 2026-09-09 工程整理后的代码位置

当前实现迁入 `football_events/stage2/`。推荐入口是 `python -m football_events.stage2`；本页原脚本命令仍保持兼容。模型公开入口为 `football_events.stage2.model.PositionPriorEventModel`，旧 `football_stage2_position_prior` 导入也保持可用。

本次重构未改变配置、损失、采样、指标或 checkpoint 参数键；小样本两轮训练的参数、优化器和预测与重构前逐值一致。仅更新尚未启动的正式实验代码快照；旧快照保存在 `archive/2026-09-09/refactor_snapshot/`，旧工程夹具的证明文件保持原样。当前验证见 `docs/codebase_audit_20260909/`。
