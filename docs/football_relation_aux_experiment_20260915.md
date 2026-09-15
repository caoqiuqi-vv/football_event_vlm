# 足球事件关系辅助机制 R1 实验

## 结论先行

机制 A 的球定位已经收敛，但在统一的 window-center 评估下，A2 相比 A1 仅有
约 `+0.123pp` macro F1，属于中性结果。R1 不再只要求 patch token 知道球在哪里，
而是要求事件时序头实际消费的 frame token 保留球门关系与球周围的人体/接触上下文。

检测结果仍只在训练阶段生成监督目标。事件 logits 不读取检测框、检测分数、关系头
输出或任何检测残差，推理图与原事件模型一致。

## 因果假设

当前球热图损失可以由 patch token 和热图头局部解决，事件分支读取的 CLS/frame token
不必保存这部分信息。因此球定位提高并不自动转化为事件收益。

R1 增加两条直接作用于事件 frame token 的训练路径：

1. 从 frame token 预测球门可见性以及球相对球门的离散几何关系；
2. 从 frame token 预测模型 Top-K 球假设周围的小、中、大三尺度 DINO patch 上下文。

第二条路径不需要 player/goalkeeper 标注。DINO patch 自身提供人体与局部交互表征，
池化后的目标停止梯度，避免辅助目标和预测器共同坍塌。

## 对错误球与多球的处理

- 球位置不作为单个硬框输入事件头；模型从球热图取 3 个局部极大值候选；
- 候选之间使用半径 2 patch 的 NMS，避免 Top-K 全落在同一个球附近；
- 候选以热图置信度做集合加权，局部上下文使用 1.5、4、8 patch 三个尺度；
- 几何关系只在 `ball_confidence >= 0.20`、`quality >= 0.75` 且球门可见时监督；
- 球/球门缺失或歧义帧对几何关系保持 unknown，不制造硬负样本；
- 球门不可见负样本仅使用 0.05 的弱权重。

首轮不引入速度、轨迹或相机运动补偿。只有 R1 先证明表征迁移有效，才进入运动关系
和检测引导证据路由实验。

## 数据

- 球教师：v9 修复轨迹
  `/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout/offline_index_v1`
- 球门教师：高覆盖 compact ROI 索引
  `/mnt/data_16t/football/roi_indices/conditional_goal_crowd_v1`
- 球门索引覆盖训练视频 128/129，calibration7 为 7/7；
- calibration7 均匀时间探针：球有效约 76.97%，球门可见约 27.89%，高质量球门关系
  有效约 11.31%。

## 梯度探针

单卡真实 batch 探针输出：

`outputs/football_events/mechanism_r_20260915/r0_gradient_probe_seed42/mechanism_a_gradient_probe.json`

- 关系/事件 LoRA 原始梯度比：`0.616, 0.264, 1.224, 1.814`；
- 原始比值中位数约 `0.920`；
- 梯度余弦中位数约 `+0.071`；
- 16/16 个共享 LoRA 参数张量均有非零梯度；
- 探针权重 0.25 的加权中位数约 0.230；正式实验下调到 0.20，预计约 0.184。

## 正式实验

R1 输出：

`outputs/football_events/mechanism_r_20260915/r1_relation_aux_seed42`

固定条件：

- 初始化 checkpoint、训练/验证切分、seed、采样、事件 loss 与 A2 一致；
- 首阶段训练 1 个 epoch 并验证，通过事件指标门槛后再续跑第 2 个 epoch；
- 每卡 batch 2、梯度累积 12，有效 batch 保持 48；
- DINO frame chunk 为 8，关闭 full-extractor gradient checkpointing；
- 每卡 2 个 worker，验证 batch 为每卡 4；
- 实测约 1.26 秒/step、3.17 clips/s，较原配置吞吐提升约 69%；
- 事件头全局 LR `8e-5`，共享 LoRA 全局 LR `1e-5`；
- 球/球门热图权重 `0.45`，关系辅助权重 `0.20`；
- event candidate time 固定为 10 秒窗口中心；
- frame event head 没有定位监督，因此禁止用其 argmax 生成候选时刻。

公平基线采用 A2 seed42 的缓存按 window-center 重算：

- macro precision `0.107460`
- macro recall `0.873808`
- macro F1 `0.189899`
- window mAP `0.266118`

## 判定门槛

首轮进入多 seed 的必要条件：

1. 关系 BCE、三尺度上下文 loss 与球/球门热图 loss 均呈正向收敛；
2. macro recall 相对基线下降不超过 `0.5pp`；
3. macro precision 在固定 recall floor 下至少提高 `1.5pp`，或 shot/save 在达到各自
   recall floor 时 FP 同时下降至少 5%；
4. window mAP 不作为主要成功标准，仅用于确认分类排序未退化；
5. 所有事件结果必须来自显式记录的 `candidate_time_mode=window_center`。

若关系目标收敛而事件仍中性，说明 frame token 可以解码空间关系，但当前关系目标仍不够
事件特异；下一步才加入 event-vs-same-video-visible-ball-negative 对比目标。若关系目标本身
不收敛，则优先检查候选熵、目标覆盖和球门关系离散化，不进入证据路由。
