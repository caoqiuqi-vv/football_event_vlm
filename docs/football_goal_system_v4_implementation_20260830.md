# 足球长视频目标系统 v4：实现与验收契约

## 目标（不可降级）

- 最终一对一、同类、`±3s`：shot recall `>=90%`，其余五类各 `>=85%`。
- 人工审核片段采用所有保留类别时间区间的并集，`union_duration / total_video_duration <35%`。
- 不用时间 NMS，不合并相邻 GT/预测；同刻 shot+save 和相邻同类事件保持独立实例。
- calibration 完成并冻结模型、融合权重、视频自适应规则和阈值后，才允许打开 18 个外部 test 视频。

## 为什么旧链路不算完成

旧 Stage-1 + clip teacher 在 18 个 calibration 视频上虽达到 shot 93.19%、save 88.96%、set-piece 88.18% recall，但全类别审核时长并集为 85.52%，明确未通过 35% 门槛。旧 test18 point-NMS 结果只能作为历史参考，不能作为本系统无 NMS 验收结果。

## 新模型

### 1. 多速率证据库

- 1 FPS、288×512 DINOv3 ViT-L/16 外观向量（2048 维）。
- 相邻帧先估计全局相机平移，再计算 residual motion；保留 8×8 空间网格（64 维）。这不是 DINO 特征差，也不依赖球/球门/球员检测。
- 4 FPS log-mel 在每秒聚合为 64 维音频；modality dropout 保证无音频回退。
- 每视频独立 mmap 数组，训练时不重复解码 DINO。

### 2. 180 秒长上下文检索器

- 180 秒 input memory、中心 60 秒 core；相邻 core 不共享输出区。
- 多尺度 dilated TCN + 3 倍降采样双向 GRU。
- 每 2 秒 cell 两个 anchor query，共 60 个 slot/core；局部 anchor 初始化后对完整 180 秒 memory 做四层 cross-attention。
- 输出五类+no-event、时间、uncertainty、quality、12 秒审核区域和 slot embedding。
- Hungarian 一对一训练，不做 NMS；两个近邻 shot 或 shot+save 可由不同 slot 表示。

### 3. 训练与漏标保护

- `点球/其他射门类型 -> shot`，补齐 `中圈开球/kickoff`，不继承旧四类 parser。
- 每个 GT 每 epoch 至少一次；不足 600 个 focus window 的稀有类增加不同窗口位置视图。kickoff 原始 GT 不复制到 validation。
- no-event focal 权重 0.15，降低漏标的错误负监督；背景仍保持 1:1。
- 固定监督审核区域到 12 秒，禁止通过输出极短区间“作弊”降低人工时长。

### 4. 视频内自适应（无标签）

- 对每类 logit 做 calibration 全局 robust-z 与当前视频 robust-z/rank 融合。
- 融合权重只在 calibration 选择；test 视频只计算其自身无标签分数分布，不能使用 test GT 调阈值。

### 5. 高分辨率 Motion Examiner

- 25 张全图帧，全部 `>=512×896`；不依赖 ROI/detector。
- 17 张中心高密度帧具有显式相对时间 embedding；旧 verifier 缺少的有序动作信息被补齐。
- 独立 17 帧相机补偿 residual-motion Transformer；输出 pre/contact/post/no-action phase。
- 音频 token 有显式时间，restart 使用 whistle MIL 辅助；另有 pass/cross/clearance/tackle/interception/camera-cut confounder 头。
- Examiner 不是“同一 DINO 特征再接分类头”：其新增证据是高分辨率 patch、8 FPS 有序短时动作、相机残差运动和 whistle 时间结构。

## 运行中的自动链路

- 像素/音频库：`/mnt/data_16t/football/set_spotter_4fps_288x512`
- 新特征库：`/mnt/data_16t/football/goal_feature_bank_dino_v1`
- watchdog：`football_e2e_spotter/scripts/watch_goal_retriever_v4.py`
- 日志/状态：`football_e2e_spotter/experiments/goal_retriever_v1/`
- 模型输出：`outputs/football_goal_retriever/longctx_dino_motion_audio_v1_20260830`

watchdog 顺序：135 train DINO/motion/audio banks -> 18 calibration banks -> 0/1/2/4/5/7 六卡 20 epoch v4 训练 -> 每 epoch calibration 无 NMS 全视频扫描与硬门槛评估。

只有 calibration `gate_pass=true` 才进入 Examiner OOF 和 sealed test18；否则报告失败项并继续针对具体瓶颈改模型，不把未达标结果描述为完成。

