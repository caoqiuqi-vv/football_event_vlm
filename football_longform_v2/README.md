# Football LongForm V2

独立的 RGB-first 足球长视频事件定位链路。当前主方案 LF-A2 对原视频做一次顺序解码，以互不重叠的 4 秒 VideoMAE 块生成 0.5 秒 tubelet 时间轴，再进行长上下文事件定位。检测和跟踪默认关闭且不能 gate 候选。LF-A0 的 DINO context + RGB motion 方案保留为失败分析和对照基线。

## 核心流程

输入视频按 4 秒非重叠块读取，每块均匀取 16 帧。VideoMAEv2 ViT-B 产生 8 个 0.5 秒 tubelet，每个 tubelet 拼接局部动作与全局场景表示。六层 dilated temporal blocks 的感受野约 126 秒，直接输出 shot/save/corner/freekick/penalty 点响应和时间 offset：

- shot 是最高优先级独立头；
- save 使用过去 6 秒、只看历史的 shot 概率轨迹；
- corner/freekick/penalty 共享宽时域 restart state，但精确时刻与类别由独立子类输出；
- family 只提供有界残差和辅助监督，不是删除候选的硬门控；
- 推理按完整时间轴一次计算，每个 tubelet 只产生一次 backbone 计算，不是原视频滑窗分类。

训练块只是在已经缓存的连续时间轴上随机索引 192 秒上下文，不会重复解码或重复计算 VideoMAE。采样是 video-balanced + event-label-balanced，hard-negative mining 关闭。

固定第三方 18 视频列表复用 ../configs/football/splits/thirdparty18_holdout_all_rest_train/。在最终外部对齐前只冻结 ID、数量和摘要，不读取其视频或标注；模型选择、阈值与消融只能使用 train 和 calibration。

## 外部模型

两个外部模型只通过命令适配器引用且默认关闭：det_and_track 的 YOLO 权重用于快速 person/goal/field-line 结构证据（没有 ball 类）；SoccerMind 的 rf-detrm-p2.pth 用作较慢的高质量足球 teacher。

## 快速验证

```bash
cd football_longform_v2
PYTHONPATH=src python scripts/verify_environment.py --config configs/lf_a0_rgb_only.yaml
PYTHONPATH=src python scripts/preflight_real_data.py --config configs/lf_a0_rgb_only.yaml
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py" -v
PYTHONPATH=src python scripts/smoke_train.py --config configs/lf_a0_rgb_only.yaml --device cpu
```

构建单视频真实 RGB timeline：

```bash
PYTHONPATH=src python scripts/build_rgb_timeline.py --config configs/lf_a0_rgb_only.yaml --video-id <VIDEO_ID> --device cuda:0
```

timeline 缓存准备好后运行 scripts/train_locator.py。缓存契约是 <feature_store>/<video_id>/timeline.npz，必需字段为 timestamps/context/motion/context_valid/motion_valid。

## LF-A2 连续 VideoMAE 链路

完整数据缓存由三个物理 GPU 静态分片，每张卡内部使用 CPU 顺序解码 / GPU 前向的有界流水线：

```bash
python scripts/coordinate_videomae_chunk_store.py \
  --canonical-config configs/lf_a0_official_fullscale.yaml \
  --a1-config configs/lf_a1_videomae_clean5.yaml \
  --checkpoint <SELECTED_A1_EPOCH_PT> \
  --output-root /mnt/data_16t/football/feature_store_v2/videomae_k710_a1_best_4s_v1 \
  --gpus 0,6,7 --batch-size 24
```

缓存契约为 `<store>/<split>/<video_id>/{features.npy,timestamps.npy,metadata.json}`。数组保持未压缩以支持 mmap 随机读取；metadata 强制记录源视频大小/mtime、A1 checkpoint/config SHA256、chunk 和 tubelet 时间尺度。

```bash
CUDA_VISIBLE_DEVICES=0,6,7 python scripts/train_chunk_locator.py \
  --config configs/lf_a2_sequential_locator.yaml \
  --device cuda:0 --device-ids 0,1,2
```

每个 epoch 都在 18 个完整 calibration 长视频上按视频进行严格一对一事件匹配，并同时报告 2/3/5 秒容差、AP、目标 recall 下 precision/FP/min、positive P10 和 false-peak P90。checkpoint 选择顺序是：达到 shot 90% → 其他支持类别 85% → recall 缺口 → precision → AP。calibration support 为 0 的 penalty 明确记为 N/A，不参与主 checkpoint 选择。

冻结阈值还要求通过 `deployment_gate`：不能通过无限降低阈值、制造大量人工候选来伪装高召回。当前阶段门禁同时约束各类最低 precision 与最大 FP/min；不通过时只保留诊断报告，不生成生产 operating points。

Penalty 使用 135 个训练视频上的五折 video-grouped OOF。27 个 penalty anchor 的预测都来自未训练过该视频的模型；固定主模型 epoch 后才运行 OOF，禁止按每折结果重新挑 epoch。OOF 的 FP/min 被转移到主模型 calibration 背景分数，原始跨模型阈值不直接复用。

## LF-A0 全量门禁

`configs/lf_a0_official_fullscale.yaml` 固定使用官方冻结 DINOv3 ViT-L/16。规范数据为 135 个独立训练媒体和 18 个校准媒体；同一长视频的短 ID 人工复核标注只作为 annotation override，不重复采样媒体。缓存和 checkpoint 必须匹配 source video、配置 SHA256、backbone ID 与权重 SHA256。校准评估要求 18/18 完整参与，缺一个即失败。

训练采用 video-balanced + event-label-balanced anchor sampling。16 视频 cohort 让每个 batch 保持 4 个不同长视频，同时把同一视频的 8 个 block 路由给同一 worker LRU，减少重复解压。人工明确判错的候选点按 family 和 class ignore radius 从负监督中移除；稠密 focal loss 按输出类别分别归一化正负质量，避免海量背景以及高频 shot 淹没 save/penalty。具体事件同时使用窄 Gaussian 做点定位、宽 state Gaussian 学习准备—执行过程（corner/freekick 8 秒、penalty 10 秒）；严格 recall 仍只按点匹配计算。

A0 首先是高召回候选器，同时训练可验证的条件化类级头：

- 校准报告同时给出固定 proposal budget recall，以及达到 shot 90%/其他类 85% recall 所需的最高阈值、precision 和 FP/min；epoch 0/1/2 都评估并按射门优先规则选模。
- 若联合条件化头仍不足，再把同一逻辑拆成候选后的独立 verifier，而不重新引入全局滑窗。
- 若放宽 budget 后 family recall 仍低于 90%，说明 2 Hz final-layer pooled DINO + magnitude-only RGB flow 信息不足，应升级为多层 patch token 与可学习短时 RGB motion encoder，而不是继续调 calibration。
- 检测与跟踪只能作为可缺失的 rerank/teacher 证据，不能决定是否生成候选。

## 长视频部署入口

全量门禁完成后会生成 `official_a_gate_best.pt` 与 `operating_points.json`。后者只由完整 18 视频 calibration 报告冻结，外部测试和第三方对齐阶段禁止重调。

```bash
PYTHONPATH=src python scripts/run_long_video_pipeline.py \
  --config configs/lf_a0_official_fullscale.yaml \
  --video-id <VIDEO_ID> --video-root <RAW_VIDEO_ROOT> \
  --operating-points experiments/lf_a0_official_fullscale/operating_points.json \
  --cache-root <DEPLOY_CACHE_ROOT> --output <PREDICTIONS_JSON> --device cuda:0
```

输出包含事件类别、秒级时间戳、分数、校准状态、选择方式和人工复核优先级。无 calibration 正例的类别不会伪造阈值，而是使用高召回 proposal budget fallback 并强制标记为高优先级人工复核。

LF-A2 在主 calibration 和 penalty OOF 都通过后，使用以下入口处理任意未标注长视频：

```bash
python scripts/run_a2_long_video.py \
  --video <LONG_VIDEO> --video-id <VIDEO_ID> \
  --a1-config configs/lf_a1_videomae_clean5.yaml \
  --a1-checkpoint <SELECTED_A1_EPOCH_PT> \
  --a2-checkpoint experiments/lf_a2_sequential_locator/best.pt \
  --operating-points experiments/lf_a2_sequential_locator/operating_points.json \
  --work-root <CACHE_ROOT> --output <PREDICTIONS_JSON> --gpu 0
```

该入口先生成可复用的连续 tubelet 缓存，再执行定位头；重复调阈值或读取标签不是部署流程的一部分。
