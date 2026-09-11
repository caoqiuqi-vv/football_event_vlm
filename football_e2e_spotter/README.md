# Football E2E Spotter

面向固定/自动跟拍足球长视频的端到端事件定位链路。主任务只有
`shot / save / corner / freekick`；按当前产品优先级，不优化 `penalty`。

## 为什么另开这条链路

历史 DINO 和 VideoMAE 实验都把视频先压成冻结或弱微调的 clip embedding，
再训练片段分类器/时序头。跨视频验证显示正样本下尾与背景严重重叠；已有
VideoMAE clip 实验在高召回下的 precision 约为 shot 21%、save 10%、corner
6%、freekick 5%。这说明继续更换 embedding 后面的头没有足够依据。

本工程改为从像素端到端训练 action spotter：

- ImageNet 初始化的轻量 RegNet-Y 只作为初始化，所有层参与本域训练；
- 在中后层特征图上同时学习空间注意、局部时间变化和跨片段运动；
- 视觉特征、显式相邻帧差、音频 log-mel 经 BiGRU 建模约一分钟上下文；
- shot 独立输出；save 额外使用过去 6 秒的 shot 轨迹；corner/freekick
  共享 restart 状态，但状态不能硬删除类别候选；
- 训练输出整段 128 帧的稠密时间轴，不是正/负 clip 分类；
- 推理顺序读取整场视频，每个 core frame 只输出一次，块间只保留很小的
  temporal halo，避免传统密集滑窗重复计算；
- 不依赖 ball/person/goal detection 或 tracking；外部检测最多作为后续可选证据，
  不能 gate 高召回候选；
- 不使用 hard-positive/hard-negative mining。采样单位是长视频和唯一事件。

方法依据是端到端 Precise Event Spotting，而不是再尝试通用视频 embedding。
E2E-Spot 证明像素到逐帧预测可以优于两阶段特征方案；CVPR 2025 的 ASTRM
工作进一步报告端到端 RegNet、局部/全局时空细化、BiGRU 与类分离损失在
SoccerNet-V2 上优于冻结大模型特征。这里实现的是针对当前远景单机位数据的
独立工程版本，不直接复制论文代码。

## 数据与隔离

训练/校准沿用 `football_longform_v2` 已冻结的 135/18 长视频划分，只复用
媒体 ID、标注解析和最终评估协议。18 个第三方对齐视频仍是密封测试集：在
模型、阈值和所有消融冻结前禁止读取视频、音频或标注。

原视频只做一次无模型预处理：2 FPS、保持 16:9 的 JPEG-LMDB，以及同步
64-bin log-mel。缓存的是像素和音频谱，不是模型特征，因此视觉 backbone
仍然可以端到端学习足球域表示。

## 验收顺序

1. 解码、时间戳、模型前后向和跨视频评估的 smoke test；
2. 24 个训练视频对完整 18 个 calibration 视频做可证伪实验，只判断是否
   学到优于均匀采样的排序，不据此宣称最终效果；
3. 135 个训练视频全量训练；
4. calibration 上 shot recall ≥ 0.90，其余三类 ≥ 0.85，并在此约束下选择
   precision/FP-min 最优阈值；
5. 模型和阈值冻结后，固定 18 视频只运行一次最终测试。

