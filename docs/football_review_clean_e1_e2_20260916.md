# 修正标注 E1 / E2 实验

启动日期：2026-09-16。用户授权按 E1 → E2 顺序训练，并纳入含待审核 case 的训练视频。

## 固定实验

- 输出：`outputs/football_events/review_clean_e1_e2_20260916/`。
- 两组各 2 epoch；同一历史 `best.pt` 的冻结副本初始化，E2 不接续 E1 权重。
- 原模型：`outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt`，epoch 3，严格匹配 452 个参数。
- 初始化 SHA256：`c68951b7a616e1a2fe65c63de26cdc2f6e769aea7105c3267e39374dc37ef8b0`。
- 冻结 DINO 基础权重，训练最后 4 block 的 rank-8 LoRA 与原时序/分类头。
- GPU 7；batch 4 × accumulation 20 = effective batch 80。
- 分类/时序全局 LR=5e-5，LoRA LR=1e-5，均为历史 8 卡全局 LR 的一半。
- 输入 16 帧、720×1280。直接从 `raw_video_1080P` 读取并 resize；固定窗口，关闭时序 jitter。
- 精确时间正例保持 10 秒窗口；弱时间正例允许更长支持窗口，以确保事件区间完整被输入覆盖，两组采样一致。
- 关闭历史 hard negative、在线 hard negative 与 positive retention，重新统计类别权重。
- E1: frame loss=0；E2: frame loss=0.15，heatmap/MIL/rank 内部权重为 0.5/0.3/0.2。

## 冻结数据与清洗

数据：`outputs/football_review_training/review_e1_e2_20260916/`。

- 只读数据库单事务快照，采集于 2026-09-16 14:04:20 UTC；不复制认证或会话数据。
- manifest SHA256：`34a2662ed277442026062e25616d9497f6533900e131bb44f228008fc0069566`。
- 包含 133 个原训练视频、9 个完成初审的原内部验证视频。实际有训练窗口的为 132 个视频，12,842 行；另一个视频无有效硬监督，未制造背景标签。
- 合并 474 条重复来源记录：同 GT 血缘、唯一明确的已确认 GT/AI 关联，或相同模型窗口的同时间重复证据。
- 清洗关联容差 3 秒，额外要求类别/子类兼容、支持区间重叠、无竞争近邻 GT；两个独立 GT 永不自动合并。
- 保留同刻 shot/save。145 对无法判定的同类近邻记录保留来源并进入冲突清单，相关监督屏蔽。
- 已确认 positive 使用修订值；未审核旧 GT 只保留 positive，不用于精确帧监督。
- 27 条待二次确认记录标记为 human_ambiguous；未审核 AI 候选为 unchecked_candidate，二者分开统计。
- 模棱两可的信息通过 unknown 区间保护监督：阻止争议类别的负采样/负梯度，不猜测 0.5 软标签。其他可信类别仍可训练；没有可信类别的 ambiguity 行不提供硬标签梯度。
- negative 仅从人工拒绝该类别的支持区间采样，减去正例与 unknown 支持区间及 2 秒余量。未标注背景不自动成为负例。
- 独立 `frame_label_masks` 同时关闭弱时间类别的 heatmap、MIL 和 rank；主分类 mask 保留。

## 验证协议

- 原训练、内部验证、test18 划分保持隔离。未完成初审的 6 个原验证视频不转入训练。
- 内部验证扫描 9 个完整视频、6,925 个窗口，但指标仅评估可信时间区间。已确认但时间不精确的新增正例周围区域不记作背景 FP。
- 当前可评估精确事件：shot 168、save 64、set_piece 80。每类报告覆盖分钟，不能解释为自然完整视频的 precision。
- 两组主比较均为 clip score + window center、stride 5 秒、NMS 5 秒、一对一匹配 ±3 秒；验证集拟合阈值，recall floor 为 90%/85%/85%。
- 全量媒体检查 142 个视频：1080P 112、720P 24、4K 5、1440P 1；与 UI 时长最大差异 0.283 秒，无起点/时长失败。
- test18 固定 v4，冻结 18 个原始 JSON 和 manifest，不改写原文件。发现同一终审 case 在 60 秒处的人工 free_kick / AI set_piece 重复，转换时合并并保存两份来源；最终 shot 682、save 370、set_piece 347。
- test18 只用各模型内部验证阈值，不在 test18 重定阈。两组同主比较协议；E2 额外保存 frame_detection 曲线和同 clip 分数/阈值下的 frame_peak 时间消融，不追加模型前向。

## 执行与检查

- 持久队列：`scripts/run_football_review_e1_e2.py`；E1 训练 → E2 训练 → E1 test18 → E2 test18。
- 队列失败即停止并写原因；每阶段启动前等待 GPU 7 空闲，不终止其他任务。
- `state.json` 保存队列/训练 PID、当前阶段、完成阶段和失败原因；`queue_console.log` 与各阶段 `train_console.log` 保存日志。
- `source_snapshot/` 冻结 816 个 Python 文件；`provenance.json` 记录实际源码指纹、Git 状态与初始化指纹。训练不读取之后的工作区代码修改。
- 12 项监督/清洗断言与 2 项历史阈值 unittest 通过；新增 test18 同终审 case 去重断言通过。
- 真实 720P batch-4 前向、反向与 optimizer 检查通过，峰值 allocated 显存约 6.38 GiB。确认 E1 仅事件梯度，E2 精确正例/可信负例产生帧分支梯度，弱时间正例帧梯度为零。

E1 已产生首轮模型和验证指标，第二轮仍在运行；实际进度见输出目录 `state.json` 和训练日志。

## 解码修复与耗时估计（2026-09-16）

首轮 E1 尚未完成一个 epoch 时，发现原视频 `1984140154622119938` 为 4K，OpenCV 原生 seek 单次阻塞数分钟，线程级 deadline 无法中断 native 调用。已归档首次尝试至运行目录的 `interrupted_attempts/opencv_seek_1/`，并从同一初始模型重新开始 E1。

E1/E2 同时采用 `single_seek_ffmpeg_highres`：超过 1080P 的原视频由 FFmpeg 直接解码、按源 PTS 采样并 bicubic resize 到 720P，单个子进程最长 45 秒；常规视频沿用 single_seek。没有改用既有 720P 转码文件。实际源帧时间用于 frame supervision。源快照与 provenance 已一致更新，冻结标注和初始模型不变。问题视频开头、中段、末尾的 16 帧实测约 11 秒。

单张 RTX5090、batch4、每实验2 epochs：原先健康速度约7秒/batch，每实验纯训练约12–13小时；E1+E2 加4次内部验证和2次 test18 评估暂估35–50小时，评估耗时尚未实测，需以重启后的实际速度更新。

## GPU4 E1 test18 dense（2026-09-17）

按用户授权停止 GPU4 上本人 `dino_guided_v2/train_native_to_e6_noeval_20260915` 任务及调度器，释放 GPU4。E1 第二轮尚在训练，因此此次冻结 `epoch_1.pt`（SHA256 `f8788b3ee34052c1aff9a51a2ef34844ac4b51a3a05a8d9d451df94f647899a9`），评测目录为运行根目录下 `E1_test18_dense_gpu4_epoch001_20260917/`。GPU7 的 E1→E2 训练不受影响。

评测 18 个完整视频、12343 个10秒窗口、5秒步长；阈值来自该 checkpoint 内部验证，使用已冻结清洗后的 v4 核心标签。输出固定阈值事件级 P/R（±3秒、NMS5秒、严格一对一），再从同一分数缓存导出逐视频窗口及指标，并运行 CPU LOOV 阈值诊断（每折其余17视频拟合 alpha/阈值）。固定窗口覆盖分别按±3秒与不扩展窗口边界报告。报告位置 `evaluation/dense_analysis/REPORT.md`，在完整推理结束后自动生成。LOOV 是 test18 内部交叉验证校准诊断，不能替代独立固定阈值结果。

启动检查修复了评测脚本 ConfigDict 的点赋值没有改变字典项，以及局部事件 time 变量覆盖 time 模块两个问题；无推理的首次失败已归档在 `startup_failure/`。重启日志确认加载冻结 E1 权重，452键严格匹配；评测配置与源快照/provenance一致更新。自动汇总已通过18视频合成数据的逐视频事件计数、GT总数、固定窗口和18折LOOV一致性检查。

## 远端 E2 四卡训练（2026-09-17）

用户已明确取消本机后续 E2，迁移到远端四卡。`cancel_football_review_queued_e2.py` 只暂停本机队列父进程，当前 E1 子进程继续训练；E1 完成后退出父队列，避免启动 E2。GPU4 的独立 test18 评测继续。

四卡配置：`configs/football/review_clean_e2_ddp4_20260917.yaml`。每卡 batch4、accumulation5，effective batch80；分类 LR_per_gpu=1.25e-5，LoRA LR_per_gpu=2.5e-6，运行时全局 LR 保持5e-5/1e-5。训练2 epochs，frame loss0.15，严格从原始历史模型初始化，不接续 E1。四卡的样本顺序与单卡不同，不承诺逐步数值一致。

远端需要安装本项目训练环境和 FFmpeg，并同步以下输入。代码仓库不包含这些运行数据或权重：

- 标注目录中的 `training_manifest.json`、`train_video_ids.txt`、`val_video_ids.txt`，来源 `outputs/football_review_training/review_e1_e2_20260916/`；不可重新导出或改写 manifest。
- `outputs/football_events/review_clean_e1_e2_20260916/initial_best.pt`，与指定历史模型指纹一致。
- `checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`。
- 原视频目录，按 `<video_id>.mp4` 命名；训练132个实际有效视频、验证9个，启动校验包含 manifest 中全部142个视频。

在远端仓库根目录，按实际存储位置修改路径并选择四张卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export REVIEW_DATA_DIR=/remote/data/review_e1_e2_20260916
export VIDEO_ROOT=/remote/data/raw_video_1080P
export INIT_CHECKPOINT=/remote/models/initial_best.pt
export DINO_WEIGHTS=/remote/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
export E2_OUTPUT_DIR="$PWD/outputs/football_events/review_clean_e1_e2_20260916/E2_ddp4"
# PYTHON_BIN 默认为当前训练环境的 python，可覆盖为环境内的绝对路径。
bash scripts/train_football_review_e2_ddp4.sh --check-only
bash scripts/train_football_review_e2_ddp4.sh
```

若同步到与仓库默认路径对应的位置，只需 `CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_football_review_e2_ddp4.sh`。启动脚本内部使用 `python -m torch.distributed.run --standalone --nproc_per_node=4 train_football_events.py --config "$E2_OUTPUT_DIR/launch_config.yaml"`；不直接用单卡 E2 配置启动四卡，否则学习率会变成4倍、effective batch变成320。

准备脚本校验冻结标注及初始化 SHA256、split IDs、FFmpeg、142个视频、独立输出目录，并记录实际配置与 Git 版本。`review_video_root` 只覆盖输入媒体路径，不改动 manifest 内容/指纹和清洗监督。输出目录应为新目录，脚本不会覆盖已有训练。数据/初始化/环境输入校验以及14个函数回归、7个 unittest 均通过；四卡训练需在远端实际执行。
