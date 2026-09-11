# Object Motion v8：事件监督驱动的对象上下文读出

基于 `d8b2f7578e16e1d50e54d522fd27f8a49b7c63c7`。这是待验证代码，不是已取得事件收益的模型。未启动训练、未修改数据、未合并 main。

## 修复与边界

1. v7 cross-attention 对 relation token 的 detach 阻断了主要事件损失到 relation projection/temporal 的梯度。v8 用 `event_relation_grad_enabled` 单独控制该边界；检测热图仍受保护。
2. 可选 `event_context_enabled` 从同一 720p、patch16 特征读取球核心、球门、球周围 3×3/7×7 patch 区域。区域权重 detach；事件投影可学习。先池化后投影，避免额外存一份全分辨率特征图。这里是近邻上下文，并非人物身份/门将角色识别。
3. 上下文 token 添加区域类型及基于真实时间戳的时间编码。可选逐帧 correction 与 clip correction 共用事件监督，逐帧 correction 只乘一次 curriculum alpha，不重复加 legacy residual。
4. `event_context_feature_grad=false` 时，事件投影/时序/融合可训练，但 dense patch 输入梯度被隔离。设为 true 才允许上下文路径更新 dense BallLoRA；检测权重仍不接收该事件梯度。原有 shared anchor BallLoRA 路径不变。
5. `event_ranking_target=final` 可把已有同视频 reviewed pair 排序作用于最终事件 logits；默认仍为 residual。该目标变化必须另做消融。
6. no-evidence 惩罚现在要求 ball/goal 都有接近 1 的有效 absence mask。漏检 unknown、弱 absence 或未启用的目标不能用来抑制事件。
7. 从旧 checkpoint 开启新上下文时，保留检测器和 relation temporal，但重新初始化 cross-attention，防止旧非零 correction 作用于新 token。clip/frame 末层为零。完整 v8 context checkpoint 则正常恢复。关闭 correction 只能恢复当前 shared anchor，不代表恢复无 BallLoRA 的原始基线。

所有新增结构/梯度选项默认关闭，旧 v7 的前向不变。no-evidence 的 mask 修正对旧开启 frame residual 的实验有意改变训练语义；v7 的 frame residual 原本为零，不受该 loss 修正影响。

## 首轮只运行 control 与 grad

在当前检出的仓库目录运行。v8 将测试与训练都绑定到该目录，避免旧脚本跳回硬编码的另一份仓库；v3/v7 单独运行时仍保留原目录默认值。数据及权重配置沿用 v7，换目录需显式指定 SOURCE_CHECKPOINT/BASE_CONFIG，并确保 reviewed manifests 等资源存在。Python 默认沿用原服务器环境，可通过 PYTHON_BIN/TORCHRUN_BIN 指定。这些命令会训练；本次代码交付没有执行它们。

```bash
MOTION_V8_MODE=control bash scripts/run_object_motion_adapter_v8.sh
MOTION_V8_MODE=grad bash scripts/run_object_motion_adapter_v8.sh
```

二者都从 v7 launcher 的相同默认 v4 epoch4 初始化，使用相同 pseudo-label、720p 输入、33 帧、优化器、curriculum 和损失。若指定 SOURCE_CHECKPOINT/BASE_CONFIG，两组必须传相同值，不能一组继承另一组。grad 只改变 relation autograd 边界，不改变初始前向数值。

grad 有收益后再跑：

```bash
MOTION_V8_MODE=context bash scripts/run_object_motion_adapter_v8.sh
MOTION_V8_MODE=uniform bash scripts/run_object_motion_adapter_v8.sh
```

context 同时加入上下文读出和逐帧出口，属于结构组合实验，不是单项缺陷修复。uniform 只把新增上下文读出的区域权重替换成均匀权重；旧 relation/geometry 和 visibility 仍保留检测信息。因此 context-vs-uniform 检验的是**新增区域读出的价值**，不是“所有检测信息”的总价值。

更后续的两项消融需要新的 OUTPUT_DIR，分别开启，不要同时改变：

```bash
MOTION_V8_MODE=context MOTION_EVENT_CONTEXT_FEATURE_GRAD=true \
  OUTPUT_DIR=/your/new/run/context_live_features \
  bash scripts/run_object_motion_adapter_v8.sh

MOTION_V8_MODE=context MOTION_EVENT_RANKING_TARGET=final \
  OUTPUT_DIR=/your/new/run/context_final_rank \
  bash scripts/run_object_motion_adapter_v8.sh
```

排名目标改变后，现有 relation BCE 仍存在；本轮未偷偷移除它。是否降低/关闭该项也需要独立验证。

## 验证

v8 launcher 会先执行下列 tensor/gradient 测试，失败则不启动训练：

```bash
python -m pytest -q tests/test_football_object_motion_v6.py \
  tests/test_football_object_motion_v8.py \
  tests/test_football_object_motion_v8_forward.py
```

覆盖：旧模式梯度隔离、grad 模式时序梯度、前向相等、旧权重安全初始化、完整新权重恢复、逐帧事件监督、可选 dense BallLoRA 梯度、坐标梯度隔离、uniform 对照、时间编码、unknown mask、生产 `_motion_forward` 的 clip/frame alpha 与 shared anchor 梯度。

`v8_forward` 从生产文件 AST 加载原函数，用 tiny backbone 替代 DINO，覆盖 wrapper 接线但不替代真实 DINO/CUDA 运行。新上下文头初始为零，首次 backward 上游梯度为零是预期行为；测试会开启末层以验证后续训练阶段的梯度路径。

交付环境没有 PyTorch，依赖安装请求未获网络授权。因此本次只执行 Python 编译、Shell 语法和不依赖 torch 的静态/启动参数测试；tensor/gradient 测试已编写但未在该环境运行。原 v4 GPU smoke 契约不覆盖 v8，不能拿旧 smoke 产物证明 v8 显存或训练有效。正式长跑前需在目标 GPU 做短程训练检查：有限 loss/梯度、梯度更新后 temporal 参数变化、clip/frame 修正幅度及峰值显存。

## 验收与停止条件

本次已通过 4 项 dependency-free launcher 测试（含四种模式子案例），验证完整 v8→v7→v3 配置传递、当前 checkout 绑定及 preflight 失败退出；训练入口在测试中被替换为参数捕获程序。复现命令：

```bash
python -m unittest discover -s tests -p test_football_object_motion_v8_launcher.py -v
```

- 固定 annotation/video split/checkpoint/采样/score fusion/NMS；阈值只在验证集拟合，测试集不重新调阈。
- 同时报告完整视频 P@目标 recall、FP/90min、逐类定位误差、检测有效支持数；按比赛配对比较，不能把重叠窗口当独立样本。
- 关注 corrected frame curve，不只看 correction BCE 或训练 teacher-hit。
- 若 grad 不提升，记录其实际梯度和参数更新量，不能以 requires_grad=true 代替检查。
- 若 context 与 uniform 都提升但二者接近，不能把收益归因于检测路由。
- 不新增事件阶段伪标签，不把漏检当背景，不扩大残差上限，不改变帧数。高 fps、人物角色、真实短时交互监督留待后续证据支持。
