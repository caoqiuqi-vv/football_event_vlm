# 当前足球工程代码地图

更新：2026-09-09。本页描述开发入口，不代表历史实验已失效。当前正式实验保持暂停。

2026-09-11 更新：当前快速验证入口为 `football_events/stage2/online.py`，配置 `configs/football/stage2_online_softprior_2ep_20260911.json`；[在线两轮方案](stage2_online_2ep_20260911.md)取代本轮的缓存准备流程。

## 当前 Stage2 开发位置

```text
football_events/
└── stage2/
    ├── sampling.py      # 峰值选择、软位置先验、原始核心与上下文 token
    ├── model.py         # Stage1 指导原始 patch 的提取与原始视频推理
    ├── data.py          # 全局/局部 mmap 缓存、窗口对齐、CPU 预取
    ├── cache.py         # 新特征提取、整理和数组校验
    ├── training.py      # 联合/冻结时序训练、校准选择及扰动评测
    ├── metrics.py       # 窗口 AP 和视频配对 bootstrap
    ├── reporting.py     # 预先约定的开发检查与最终报告
    ├── artifacts.py     # 原子写入、哈希和窗口排序
    ├── experiment.py    # prepare、preflight、进程调度和 CLI
    └── __main__.py      # python -m football_events.stage2
```

导入关系：执行模块调用缓存/训练/报告模块；训练模块使用数据读取、公共指标和现有 JointTemporalReader；模型模块使用采样逻辑和已验证的冻结骨干。公共训练模块不再从历史实验 CLI 导入训练函数。

当前配置：`configs/football/stage2_originalpatch_softprior_joint_20260909.json`。

保留兼容入口：

- `football_stage2_position_prior.py` 转发公开模型和采样接口。
- `scripts/run_football_stage2_position_prior.py` 转发当前 CLI，原命令参数保持不变。
- `scripts/run_football_stage2_joint.py` 保留历史顺序执行入口，但训练函数共用 `football_events.stage2.training.train`。本轮不启动它。

## 共享基础和历史复现

|区域|职责与边界|
|---|---|
|`train_football_events.py`|原事件模型、数据、训练与评测的共享兼容基础；89 个 Python 文件直接引用，当前不改模型键名和训练语义|
|`football_stage2_joint.py`|独立冻结教师、联合时序 Reader 和旧推理入口；当前包复用 Reader|
|`football_stage2_corepatch.py`、`football_stage2_optional.py`|历史实验固定的骨干、适配器、ROI 与旧提取逻辑；当前包仍复用验证过的基础类|
|`football_stage2_metrics.py`|唯一共享的窗口 precision / 事件 recall 口径|
|`scripts/train_football_localization_full.py`、`football_localization_full.py`|Stage1 全量原生帧数据与训练|
|`football_object_motion/`|运动证据模块，具有独立训练、推理和测试|
|`football_e2e_spotter/`、`football_longform_v2/`|独立子项目，保留各自代码、配置和实验记录|
|`tools/football_event_review/` 等|正在使用的标注、复核及 UI 服务，不能按文件名版本号删除|
|`football_roi_crop_tool_export/`|独立交付工具，包含有意保留的文件副本|
|`dinov3/`|DINOv3 共享基础和上游代码|

2026-09-03 地图列出的检测辅助、object teacher、motion 分支是历史/并行路线，不再作为本次 Stage2 的默认实现位置；相关代码和配置保留。

## 新增代码与清理规则

1. 当前 Stage2 业务逻辑加入上述包；CLI 只处理参数和任务调度。
2. 新实验优先复用训练、缓存和指标，通过配置表达差异。实验特有模型代码有明确入口。
3. 保留 checkpoint 的参数键、标签语义、帧时间和既有评测口径；改变它们需要单独实验说明。
4. 调整受 `run_provenance.json` 固定的代码时，保留旧源码/哈希。已运行实验不能静默改写证明文件；本次只更新确认未开跑的新 Stage2 准备快照。
5. 历史实验归档需要有明确结束/替代依据，并核对 import、启动命令、配置、进程和输出引用。没有被 import 的 CLI 不等于无用。
6. 唯一备份移入归档并记录路径及 SHA256；只有已有字节一致副本时才移除重复文件。配置划分、模型和结果不自动清理。

[详细审计与后续拆分顺序](codebase_audit_20260909/REVIEW.md) · [Stage2 实验协议](stage2_originalpatch_softprior_joint_20260909.md)
