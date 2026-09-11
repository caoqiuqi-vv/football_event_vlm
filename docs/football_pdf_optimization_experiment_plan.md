# 足球优化建议 PDF 实验计划

## 目标

在六个长视频 `window-overlap` 评测中保持高召回，优先提高 `shot/save` precision。所有实验从相同数据划分、负样本比例和评测协议出发，避免同时修改 backbone、采样、loss 和时序头。

## 已有结论

- Frame Actionness 能区分部分 `shot/save` 正负窗口，但简单乘法或硬筛选只带来很小 precision 收益。
- attention pooling、class query、event top-k、hard-negative 和多种 ROI 后验融合已经做过，没有形成稳定增益。
- A-best validation：macro P/R/F1 为 `65.86/75.97/69.00`，mAP `71.13`。
- A-best 六长视频 window-overlap：
  - shot P/R：`40.89/87.67`
  - save P/R：`21.40/76.81`
  - set_piece P/R：`67.95/63.92`

## 实验顺序

### E1 Temporal Difference（已启动）

- A-best 的 DINO/LoRA 冻结。
- 在 `frame_proj` 后显式构造 `z_t`、`Δz_t` 和 `Δ²z_t`。
- 零初始化 residual adapter，初始化时严格保持 A-best 输出。
- 继续训练原 frame head、Temporal Transformer 和 classifier。
- 16 帧、640x1120、human-repair 标签和 negative ratio 5 均保持不变。

继续条件：validation mAP 提升至少 0.5pp 或 macro precision 提升至少 1pp，同时 macro recall 下降不超过 1pp。六长视频要求 shot/save precision 至少提升 2pp，recall 下降不超过 1.5pp。

### E2 Pure Multi-scale TCN

仅当 E1 有收益时进行。用 dilation 1/2/4 的 TCN 提取短时运动，再由小 Transformer 建模长上下文。不得同时加入 ROI、hard-negative 或 focal loss。

### E4 Camera-residual Dynamic Patches

如果 E1 无收益，跳过继续扩大全局时序头。对相邻帧 patch cosine change 减去帧内 median，选 Top-M residual-changing patches，验证小区域运动是否被全局 pooling 稀释。

### E5 High-resolution Local Evidence

仅当 E4 明显提升后进行。由时序/空间响应选择原始分辨率 crop，不再使用 detector ROI 作为唯一硬前置。

## 当前运行

```text
output: outputs/football_events/vitl16_temporal_difference_delta2_from_A_best_16f_hr
gpu: 2,3
per-GPU batch: 8
effective batch: 64
head LR: 1e-4 global
epochs: 4
```
