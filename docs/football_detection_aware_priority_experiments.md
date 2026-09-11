# Robust Detection-Aware：高优实验执行说明

## 固定条件

第一轮只使用 detector-covered 的 47 train / 11 val，且保持以下条件不变：

- 初始化：`/mnt/data_16t/football/qiuqi/checkpoints/lora_r2_best.pt`
- seed：42
- 32帧、10秒、temporal jitter 1.5秒
- `negative_ratio=2`
- `pos_weight=[1.0,2.5,4.0]`
- sampled validation 与 dense 阈值：0.5
- dense：5秒 stride、point-NMS 5秒、匹配容差5秒
- global-only 与 robust-gate 均训练6 epochs，batch size 1、gradient accumulation 24、学习率0.0002

全局分辨率先通过冻结模型预检。在相同的1024条分层配对验证样本上，如果192×320相对384×640的mAP下降不超过0.01且recall下降不超过0.01，则两个训练实验都使用192×320全局视图；否则都使用384×640。

## 任务1：免训练基线与ROI审计

脚本：`scripts/run_football_da_audit.sh`

该任务只需要一次索引构建，随后完成：

1. 在11个验证视频生成版本化 `robust_window_roi_v2` manifest。
2. 输出四种ROI模式、高/中/无效置信度、正样本/背景窗口的可视化。
3. 生成 `manual_audit.csv`，人工填写 `reviewer_correct` 后重跑 `roi_audit` 即可计算高置信ROI正确率。
4. 计算有效ROI中位面积和相邻重叠窗口中位IoU。
5. 在6个dense视频上重新评测：原始global、旧max-goal/all-ball hard crop、新robust hard crop。
6. 自动汇总precision、FP下降比例和recall验收结果。

```bash
mkdir -p logs
nohup bash scripts/run_football_da_audit.sh 0 \
  > logs/football_da_audit.log 2>&1 &
```

主要输出：

- `outputs/football_roi_audit/robust_v2_val11/roi_manifest.jsonl`
- `outputs/football_roi_audit/robust_v2_val11/summary.json`
- `outputs/football_roi_audit/robust_v2_val11/manual_audit.csv`
- `outputs/football_roi_audit/robust_v2_val11/visualizations/`
- `outputs/football_eval_runs/detection_aware_comparison.json`

## 实验2：global-only控制组

配置：`configs/football/dinov3_vitl16_detector47_global_control.yaml`

冻结DINO/LoRA，训练原有global frame projection、temporal transformer和分类头。它与门控组使用完全相同的数据、初始化、epoch、seed、有效batch、学习率、pos_weight和checkpoint选择规则。

```bash
nohup bash scripts/run_football_da_global_control.sh 0 \
  > logs/football_da_global_control.log 2>&1 &
```

## 实验3：global + robust ROI门控组

配置：`configs/football/dinov3_vitl16_robust_dual_exp2.yaml`

冻结DINO/LoRA和global分类分支，仅训练复制初始化的local分支与类别级门控。损失为 `fused BCE + 0.3 × valid-local BCE`。训练噪声包括15%强制无效ROI、球/球门随机丢失、5%孤立误检ROI以及clip级平移/缩放；所有32帧使用同一个增强后ROI。

```bash
nohup bash scripts/run_football_da_robust_gate.sh 0 \
  > logs/football_da_robust_gate.log 2>&1 &
```

## 验收顺序

先审查ROI本身，再比较模型：

1. 高置信ROI人工正确率 ≥ 0.90。
2. 有效ROI中位面积 ≤ 0.60。
3. 相邻有效ROI中位IoU ≥ 0.60。
4. 相对原始dense基线：precision ≥ 0.225、FP下降 ≥ 15%、recall ≥ 0.715。
5. robust-gate相对同数据global-only：sampled validation macro mAP下降不超过0.01。

只有上述条件通过后，才补跑缺失的80个训练和24个验证视频检测，并扩展到完整127/35 split。
