# 工程审计与整理结果 · 2026-09-09

本次完成了全工程静态审计、当前 Stage2 主线的结构重构、可核实的备份清理和行为回归验证。正式训练及正式缓存未启动，旧实验仍暂停。

审查范围为重构前 778 个 Python 文件、172,879 行。静态扫描覆盖本工作区源码，排除权重、输出、归档、虚拟环境和 node_modules；人工深入检查主训练入口、Stage1/Stage2 的调用关系、缓存和指标接口、当前进程入口及待清理备份。没有把静态扫描描述为对全部 17 万行的逐行语义审计。

## 结论与处置

|优先级|发现与证据|本次处置|
|---|---|---|
|高|`train_football_events.py` 共 20,694 行，89 个文件直接引用；`VideoEventClassifier` 4,060 行，`train()` 3,137 行，`make_model()` 686 行|保留共享兼容基础及已有参数键，明确后续拆分边界；本轮不改变其代码，历史哈希一致|
|高|当前 Stage2 从多个历史实验 CLI 导入训练、缓存读取和指标；修改一个实验易影响另一个入口|当前逻辑迁入 `football_events/stage2/`，训练器有唯一的公共实现，两个联合时序 CLI 共用；旧命令和公开导入保留|
|高|旧核心实验固定了源文件哈希，直接搬动或改写会阻断原实验恢复|35 个受保护 Python 文件哈希全部保持一致。仅为确认未启动的新 Stage2 更新准备快照，完整保存旧源码和证明文件|
|中|原位置先验模型和执行脚本混合多种职责，存在大量单行复合语句|拆分采样、模型、数据、缓存、训练、指标、报告、原子文件操作及执行管理；展开复合语句和长容器，计算表达式不变|
|中|根 README 缺失，旧 ACTIVE 地图仍将 9 月 3 日检测/运动分支作为默认路线|新增根 README，更新当前代码地图、入口、目录职责、兼容和归档约定|
|中|根测试未设置发现边界；旧 Stage2 测试有函数位于 main 调用之后，直接运行文件会漏跑|设置根 `testpaths=["tests"]` 和运行资产排除规则；新增无 pytest、无 GPU 的统一 Stage2 测试入口，执行 10 项检查|
|中|`setup.py` 引用不存在的根 LICENSE，且 `install_package_data` 不是有效参数|改为实际 `docs/LICENSE.md` 和 `include_package_data`；元数据读取通过。没有改许可证正文或安装依赖|
|低|`conda.yaml` 重复声明 pandas；活动目录残留编辑器备份及探针|移除重复声明；移出 7 个备份/探针，完整保留可恢复副本|

## 当前代码结构

```text
football_events/stage2/
  sampling.py       峰值、软位置先验、核心和上下文 token
  model.py          原始 patch 提取和原始视频推理
  data.py           mmap 缓存读取、窗口/时间映射、预取
  cache.py          提取、整理、完整性校验
  training.py       联合/冻结时序训练、校准选模、扰动评测
  metrics.py        窗口 AP、配对视频 bootstrap
  reporting.py      开发检查、已知片段诊断、最终报告
  artifacts.py      原子写入、哈希、窗口排序
  experiment.py     配置准备、只读预检查、进程生命周期
  __main__.py       python -m football_events.stage2
```

`football_stage2_joint.py` 中的 Reader、历史骨干以及 `football_stage2_metrics.py` 的 EventCurves 仍是被复用的共享基础。没有为了目录整齐而改动历史核心实验固定的文件。

|兼容入口|整理前行数|整理后行数|当前职责|
|---|---:|---:|---|
|`football_stage2_position_prior.py`|121|8|公开模型/采样接口转发|
|`scripts/run_football_stage2_position_prior.py`|345|15|当前 CLI 转发|
|`scripts/run_football_stage2_joint.py`|211|124|保留历史顺序执行和报告，使用公共训练器|

全工程 Python 文件数从 778 变为 792、总行数从 172,879 变为 173,374；新增的是职责模块、审查/回归入口及展开后的可读代码。本次减少的是入口耦合和训练实现复制，没有声称全仓库总代码量下降。

## 删除、归档与保留

7 个活动目录备份/探针共约 27.4 KiB：

- 1 个 `test_football_detection_aware.py.orig` 与 9 月 3 日归档逐字节一致，仅移除活动副本。
- 其余 6 个唯一备份/探针移入 `archive/2026-09-09/workspace_artifacts/`。
- 每项原路径、归档路径、SHA256 和操作见 [cleanup.json](cleanup.json)。归档后再次核对内容哈希。
- 源码修改前副本及尚未启动实验的原证明文件，见 `archive/2026-09-09/refactor_snapshot/MANIFEST.json`。

未删除任何正式 split 文件、checkpoint、缓存或实验结果。没有停止其他训练、评测或标注服务。

静态扫描发现 104 组至少 8 行的精确同名函数 AST 重复，其中 61 组涉及旧 `train_football_events_v1.py`。这类重复不能单凭函数文本删除：旧 v1 文档描述了独立的 centered-anchor 监督实验，未找到足以判定其可弃用的最终依据，且相同函数可能依赖不同模块状态。另有 `football_roi_scoring.py` 在独立导出工具中完整复制，这是独立交付用途，予以保留。

本次新增公共模块仍与冻结的历史文件保留少量相同辅助实现，用于摆脱对历史 CLI 的运行时依赖；旧固定文件作为实验复现证据保留。因此整理后精确重复函数组为 105，未宣称这些历史副本已经全部合并。

文件名含 v1/v2、没有被 import、长期未修改，都不足以证明脚本可删。审计工具只列证据，不执行自动删除。

## 验证结果

|验证|结果|
|---|---|
|当前及旧候选选择、精确核心、边界、无效值、NULL、梯度、指标、导入兼容|10 项 CPU 检查全部通过|
|公共 train 函数迁移前后 AST|一致，计算表达式不变|
|12 个真实缓存窗口、2 epochs，重构前后训练|所有可学习 tensor 和优化器状态逐值一致|
|开发预测、末轮启用预测、所选空输入和末轮错配预测|逐值一致|
|已完成的小样本任务再次进入 train|不重新训练、不重写 checkpoint|
|加载重构前检查点，真实 720P 推理对缓存推理|最大 logits 差 0，容差 1e-6|
|显式空证据|logits 和最终判定精确回退原模型|
|当前 43,585 窗口配置 preflight|通过；正式新缓存 0 个窗口，正式训练未开始|
|历史核心实验 Python 哈希|35/35 保持一致|
|全工程 Python AST 解析|无语法错误|

工程测试不是新的模型收益实验。没有重新搜索阈值、调整损失权重或变更数据划分。GPU 检查仅在独立工程夹具中运行。

详细证据：[训练回归](training_regression.json)、[推理回归](runtime_regression.json)、[最终核验](validation.json)、[证明文件转换记录](provenance_transition.json)、[整理前静态清单](before.json)、[整理后静态清单](after.json)。原工程夹具证明文件保持原样；恢复旧版本时需要对应旧源码快照，不能把本轮新哈希追写到历史完成记录中。

## 主训练器的后续拆分顺序

这部分是审计发现和重构边界，尚未实施；不能把当前 Stage2 的回归结果外推为所有旧实验都已完成回归。

1. **配置与标签语义**：先把配置解析、标签 schema 和全局可变 LABELS 的关系显式化。新接口以 schema 参数传入；旧 `train_football_events` 仍提供兼容导出。验收多标签 schema、旧配置和旧 checkpoint 解码一致。
2. **数据和采样**：拆原生帧时间、窗口标签、split 读取、负例安全边界与 Dataset。验收同 seed 下逐窗 frame indices、labels、label_mask 和时间映射一致。
3. **模型与分支构造**：将 4,060 行分类器的独立 head 和融合组件移出，以明确注册关系替代持续扩展的条件分支。保留 state_dict 参数路径与原 forward 协议，逐模型配置对照中间特征和 logits。
4. **训练与评测**：最后拆 loss 组合、优化器、checkpoint/resume 和评测协议。窗口 AP、事件 recall、PointNMS 等口径分别命名，避免在重构中混用。
5. **历史实验归档**：按“明确最终结论＋依赖/进程检查＋配置和结果可恢复”逐组处理。不要把新目录整理变成旧实验批量失去入口。

这些边界的目的在于让下一次修改可验证；当前工作区的 89 个主入口依赖和历史哈希约束不适合一次性无回归搬迁。

## 日常使用

在仓库根目录和原 Python 环境中运行：

```bash
python tests/run_stage2_checks.py
python -m football_events.stage2 --phase preflight
python tools/audit_football_codebase.py --output /tmp/football_codebase_audit.json
```

根 pytest 发现范围已经配置，但当前环境没有安装 pytest，本次通过自带 unittest 的 Stage2 入口验证。没有为了代码整理更换训练环境。`setup.py --name` 已验证；没有将工作区足球模块宣称为可独立安装发布的包。
