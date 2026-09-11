# Object Motion v9：让普通长视频窗口进入训练

基于 v8 提交 `2e805358ff4f8fc7bc1ff0538a3220939ee9f637`。本次是采样覆盖实验，未训练模型，未声称事件指标已经提升。

## 修复什么

旧 `require_true_pairs=true` 会让外层 dataset 和 sampler 只访问配对记录，底层 online-simulation 生成的普通背景不保证被遍历。v9 增加 `sampling_mode=dense_mixed`：

1. 在训练 split 的原始视频记录上直接调用现有 `build_online_eval_records`，得到与验证一致的窗口起止、步长、视频末尾补窗。默认仍为 10 秒窗口／5 秒步长。不会借用验证或测试视频。
2. 窗口标签继续由现有函数计算，使用 E1.6 的 accepted/rejected context 与跨类别 ambiguity 权重。保留源记录的类别 mask，不把未知类别改成负例；清除代表记录遗留的 pair/cohort 字段。自然窗口不是 E1.6 的 central/edge 对，因此不继承 edge 的角色权重。
3. dataset 暴露“完整普通窗口池 + 配对别名”。补建 reviewed negative 只追加到内部存储，不改变普通池的边界。同视频没有该类正例的视频仍能通过普通池参与训练。
4. 普通窗口按随机排列跨 epoch 轮转，走完一轮才重复；配对有独立配额。DDP 按全局顺序分片，所有 rank 批数、流类型一致，不固定丢弃尾部记录。小池耗尽会重新排列后重复，并报告重复数。
5. batch size 1 时，pair 两侧是连续的两个 microbatch；batch size 2 时是同一批、正例在前。普通样本与 pair 不混在同一 microbatch。epoch 开始和普通/验证批次会清空 pair queue。每个 epoch 使用固定索引池，仅改变采样顺序，不会导致 persistent worker 持有过期记录。
6. 解码失败或换记录重试后，不能把实际返回的另一段视频当成合法 reviewed pair。该对的排序监督跳过；原有解码失败 mask 保留。batch size 2 的无效监督也安全跳过。

这是窗口分布与配对调度的对齐，不等同于“训练推理所有行为完全相同”。随机图像/帧采样增强、正负损失重加权、融合分数目标、难例挖掘仍是独立问题。整场视频不需要一次性进入 GPU；720p、33 帧及模型结构沿用 v8。

## A/B 启动

在此分支的 checkout 中执行。默认沿用原服务器 Python、权重和数据位置；新 checkout 需指定 SOURCE_CHECKPOINT / BASE_CONFIG，并准备配置所引用的其他数据资源。

```bash
# A：配对池，固定预算
MOTION_V9_MODE=control bash scripts/run_object_motion_adapter_v9_dense_sampling.sh

# B：完整网格池 + 配对子流，相同预算
MOTION_V9_MODE=dense bash scripts/run_object_motion_adapter_v9_dense_sampling.sh
```

两组默认都使用 v8 `grad` 模式。也可两组同时指定 `MOTION_V8_MODE=control`，使模型使用旧 v7 的梯度边界；不要只改其中一组。所有新采样行为仅在 v9 或显式配置后生效。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `MOTION_SAMPLING_BATCHES_PER_RANK` | 2016 | 每个 epoch 每个 rank 的 microbatch 数；是可控预算的试跑起点，并非已调优的训练量 |
| `MOTION_NATURAL_WINDOW_FRACTION` | 0.5 | dense 模式的普通窗口样本份额；其余来自 reviewed 正负配对 |
| `MOTION_POS_WEIGHT` | `[1.0,1.0,1.0]` | 两组锁定相同 BCE 正例权重，防止扩大记录池时 `auto` 自动改变损失 |
| `MOTION_V8_MODE` | `grad` | 两组相同的模型配置 |
| `OUTPUT_DIR` | 含 v9 模式与 v8 模式 | 分开保存实验，重复运行需要新目录 |

50% 普通 + 50% 配对约等于 50% 普通、25%配对正例、25%配对负例；普通窗口也包含正例，多标签有效比例并不严格等于上述数字。没有任何可用配对时 dense 自动使用普通池；缺失/错误的 reviewed manifest 仍报错，不能用未经审核的数据替代。control 没有配对则报错。

固定正例权重是 A/B 的共同控制条件，并不意味着旧 BASE_CONFIG 的实际权重就是 1。若要沿用旧实验，请从旧日志取出实际 `pos_weight`，两组都设置相同值。未修改 BCE/帧损失的正负归一化公式；本轮没有实现自然风险加权或最终融合分数排序。

预算必须能整除 `GRAD_ACCUM_STEPS`；batch size 1 的 pair 要求累积步数为偶数，以免在 pair 两侧之间更新权重。默认 2016 / 12 = 每 epoch 168 次优化更新（各 rank 同步），两组一致。不要用不同 epoch 长度比较结果。

**完整窗口池可访问，不代表小预算下每个 epoch 已扫完整场。** 日志中的 `epochs_per_natural_pass` 给出覆盖一轮普通池所需 epoch 数。如果它大于你的总训练轮数，需要增加两组共同预算或共同训练轮数，再评估长跑价值。默认预算只适合作为可控试跑起点。

## 应检查的证据

- `object_motion_dense_pool`：视频/窗口总数、各类正例、有效负例及 mask 为零的窗口数。对照训练视频清单；上游根本没有记录的视频不能凭空被补回。
- `object_motion_sampling`：普通/配对样本配额、全局唯一普通窗口数、重复数、唯一配对数及普通池覆盖周期。
- 依旧使用现有验证协议。默认 v7/v8 配置含三视频 sentinel，它用于快速回归；结论需要固定的完整比赛验证/测试集合，不能仅依赖 sentinel。
- 固定比赛划分、checkpoint、模型模式、优化步数、融合/NMS/容差，比较 P@目标 recall、FP/90min、逐类定位误差及逐比赛退化情况。阈值只在验证集选择。

自然窗口的负监督依赖上游标注/mask 的可信度。如果某类未完整标注，必须在上游保持该类 unknown；本次代码不会自动识别漏标。窗口端点的标签包含规则复用已有实现，没有另定义一套边界语义。

## 测试与当前限制

已在交付环境通过 11 项不依赖 PyTorch 的采样测试（使用 NumPy），以及 7 项启动链路测试；Python 编译和 Shell 语法检查通过。采样测试通过 AST 加载现有生产网格、标签、安全权重、配对构建函数及加载入口，覆盖末尾补窗、无配对视频、context ignore、mask 保留、DDP 分片、跨 epoch 覆盖、恢复顺序和验证入口不变。

```bash
python -m unittest discover -s tests -p test_football_object_motion_v9_sampling.py -v
python -m unittest discover -s tests -p test_football_object_motion_v8_launcher.py -v
python -m pytest -q tests/test_football_object_motion_v9_integration.py
```

当前环境无 PyTorch，因此真实 DataLoader（含 worker）、队列梯度及 production wrapper 接线测试已编写但尚未运行。v9 launcher 会先执行这些测试，再执行 v8 的张量测试，失败不会启动训练。AST 测试不等价于真实 DINO/CUDA 的短程 smoke；本次没有启动训练或远端 CI，也没有提交数据、权重或环境依赖改动。
