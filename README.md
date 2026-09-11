# Football events / DINOv3

基于 DINOv3 的足球事件识别工程。当前开发方向是 Stage1 定位指导与 Stage2 事件时序融合；完整项目导航见 [当前代码地图](docs/ACTIVE_FOOTBALL_CODEBASE.md)。

当前快速验证改为 [在线两轮方案](docs/stage2_online_2ep_20260911.md)：`python -m football_events.stage2.online`，不生成特征缓存。下方原缓存入口继续保留用于历史方案。

## 常用入口

|任务|入口|
|---|---|
|当前 Stage2 实现|[`football_events/stage2/`](football_events/stage2/)|
|Stage2 配置与实验协议|[配置](configs/football/stage2_originalpatch_softprior_joint_20260909.json)、[协议](docs/stage2_originalpatch_softprior_joint_20260909.md)|
|Stage1 全量定位训练|[`scripts/train_football_localization_full.py`](scripts/train_football_localization_full.py)|
|原事件训练与共享模型|[`train_football_events.py`](train_football_events.py)|
|完整视频事件评测|[`scripts/eval_long_video_checkpoint.py`](scripts/eval_long_video_checkpoint.py)|
|代码审计与清理记录|[2026-09-09 审计](docs/codebase_audit_20260909/REVIEW.md)|
|DINOv3 上游文档与许可|[原 README](docs/README.md)、[LICENSE](docs/LICENSE.md)|

## 检查与运行

在仓库根目录、已配置的 `qiuqi_sam3` 环境中执行：

```bash
# CPU 行为检查，无需 pytest，也不会启动训练。
python tests/run_stage2_checks.py

# 当前 Stage2 只读预检查。
python -m football_events.stage2 --phase preflight

# 重新生成全工程静态审计，不自动删除文件。
python tools/audit_football_codebase.py --output /tmp/football_codebase_audit.json
```

训练需要显式指定 `--phase pipeline`；旧命令 `python scripts/run_football_stage2_position_prior.py ...` 保持兼容。GPU、缓存、续跑和评测命令见实验协议。本轮整理没有启动正式实验。

## 目录约定

- `football_events/`：按职责组织的当前开发代码。
- `scripts/`：命令行入口及有历史兼容要求的实验脚本。当前 Stage2 新逻辑写入包中。
- `configs/football/`：配置；`splits/` 保存数据划分，不作为临时文件清理。
- `tests/`：共享足球模块测试。独立子项目在各自目录运行测试。
- `docs/`：协议、设计、结果解释和审计。
- `outputs/`、`checkpoints/`：实验数据与模型资产。
- `archive/`：带原路径和校验记录的可恢复历史材料。

`football_object_motion/`、`football_e2e_spotter/`、`football_longform_v2/` 和 `tools/` 中的标注/审核服务有独立用途。`football_roi_crop_tool_export/` 是可独立拷贝的工具，不能仅因包含同名文件就删除。当前足球入口依赖仓库内的历史模块，须在完整工作区运行，并非独立发布的安装包。

## 代码仓库与数据存储

本工程使用 [football_event_vlm](https://github.com/caoqiuqi-vv/football_event_vlm) 管理代码、测试、配置和文档。模型、outputs、生成数据及私有账号配置不提交；换机器后需另行挂载数据存储。提交前运行 `python3 scripts/check_source_only_index.py`。详见 [Git 工作流程](docs/GIT_WORKFLOW.md)。
