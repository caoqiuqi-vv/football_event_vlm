# Football Precision-60：本地诊断、远端顺序实验

所有路径均相对仓库根目录。严格按阶段执行；上一阶段未审阅时不要启动下一阶段。

## 阶段 0：已有 dense 分数的上限诊断（CPU）

前台命令：

```bash
bash scripts/run_football_precision60_experiments.sh stage0
```

它分别计算 `window_overlap` 的 tolerance=5s/2s，排除
`2027572406738604033:set_piece`，并比较 checkpoint 阈值、分类别阈值、共享阈值和
每视频 oracle 阈值。主文件位于：

```text
outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/dense_pr_ceiling_fused_tol5/
```

返回该目录的 `diagnostic_summary.json`、
`summary_metrics_exclude_unlabeled_setpiece.json`、`per_video_per_class.csv` 和两条 PR 曲线。

停止条件：若分类别阈值已同时达到 P>=0.60 和 R>=0.8245，则停止训练，只做独立
calibration split 上的阈值固化。

## 阶段 1：冻结 DINO+LoRA 的三个时序头

完整基础配置：
`configs/football/dinov3_vitl16_stage1_frozen_head_probe.yaml`。三个命令只覆盖
`temporal_fusion` 和输出目录，其他训练参数完全相同。当前 long-video 动态窗口缓存格式
与仓库的 clip feature cache 不兼容，因此这里冻结已加载的 DINO+LoRA，并在 `no_grad`
下前向；不更新 backbone。

前台 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_football_precision60_experiments.sh stage1-smoke
```

正式命令（一次只启动一个）：

```bash
mkdir -p logs
nohup bash scripts/run_football_precision60_experiments.sh stage1-cls \
  > logs/stage1_cls.log 2>&1 &
nohup bash scripts/run_football_precision60_experiments.sh stage1-class-query \
  > logs/stage1_class_query.log 2>&1 &
nohup bash scripts/run_football_precision60_experiments.sh stage1-attn-pool \
  > logs/stage1_attn_pool.log 2>&1 &
```

资源保守命令（减少 batch，不改变有效 batch）：

```bash
CUDA_VISIBLE_DEVICES=0 python train_football_events.py \
  --config configs/football/dinov3_vitl16_stage1_frozen_head_probe.yaml \
  'output_dir=outputs/football_events/vitl16_stage1_frozen_head_probe_cls_safe' \
  'model.temporal_fusion=cls_transformer' 'gpu_ids=[0]' \
  train.per_gpu_batch_size=4 train.grad_accum_steps=16
```

每个 epoch checkpoint 都做六视频 dense 评测；不要用 sampled-val F1 选择头：

```bash
bash scripts/run_football_precision60_experiments.sh \
  eval outputs/football_events/vitl16_stage1_frozen_head_probe_cls/checkpoint_epoch_1.pt \
  stage1_cls_epoch1_tol5
```

选择规则：以 tol=5s 的 `R@P60` 为主，tol=2s 为辅助。新头提升不足 1pp 就停止继续扩展
时序头；胜者提升至少 2pp 才走阶段 2 last-4。

## 阶段 2：一个单流高分辨率完整训练

配置：`configs/football/dinov3_vitl16_stage2_single_hr_hardneg.yaml`。

先从已有 train dense mining 结果生成少量、分类别 mask 的 shot/save hard negatives：

```bash
bash scripts/run_football_precision60_experiments.sh hard-negatives
```

前台 smoke test（将 `INIT_CHECKPOINT` 换成阶段 1 胜者；若没有胜者，保留基线）：

```bash
CUDA_VISIBLE_DEVICES=0 python train_football_events.py \
  --config configs/football/dinov3_vitl16_stage2_single_hr_hardneg.yaml \
  model.init_checkpoint=/path/to/accepted_head.pt \
  'gpu_ids=[0]' train.epochs=1 train.per_gpu_batch_size=1 \
  train.grad_accum_steps=4 data.num_workers_per_gpu=0
```

路径 A（时序头 `R@P60` 提升至少 2pp，保持 LoRA last-4）：

```bash
INIT_CHECKPOINT=/path/to/accepted_head.pt TEMPORAL_FUSION=class_query_transformer \
nohup bash scripts/run_football_precision60_experiments.sh stage2-last4 \
  > logs/stage2_last4.log 2>&1 &
```

路径 B（三个头差距小于 1pp，扩展 LoRA last-8）：

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
TEMPORAL_FUSION=cls_transformer \
nohup bash scripts/run_football_precision60_experiments.sh stage2-last8 \
  > logs/stage2_last8.log 2>&1 &
```

默认是 4xA800、每卡 batch=1、accum=16、BF16、gradient checkpointing。若显存稳定，
可在直接训练命令中改为 `train.per_gpu_batch_size=2 train.grad_accum_steps=8`，保证有效
batch 不变。显存不足时保持默认，并将 `data.num_workers_per_gpu=0` 排除 worker 预取影响。

对 `checkpoint_epoch_*.pt`、`best.pt` 和 `last.pt` 逐一运行上面的 `eval` 命令。若 last-8
只提升 sampled train/val、不提升 dense `R@P60`，回退 last-4。

## 阶段 3：只做 FP verifier 的 ROI 分支

只有阶段 2 单流 PR 曲线可接受时启动。完整配置：
`configs/football/dinov3_vitl16_stage3_roi_verifier.yaml`。

实现约束：DINO、LoRA、global frame projection、frame-event head、temporal 和 classifier
全部冻结；`local_loss_weight=0`；ROI confidence<0.60 或 ROI 无效时 fused logits 与
global logits 严格相等；positive-retention loss 仅惩罚标注正类被 verifier 向下修改。

前台 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 INIT_CHECKPOINT=/path/to/stage2_best.pt \
bash scripts/run_football_precision60_experiments.sh stage3-smoke
```

正式命令：

```bash
INIT_CHECKPOINT=/path/to/stage2_best.pt LORA_BLOCKS=4 \
TEMPORAL_FUSION=cls_transformer \
nohup bash scripts/run_football_precision60_experiments.sh stage3 \
  > logs/stage3_roi_verifier.log 2>&1 &
```

若 stage-2 使用 last-8，设置 `LORA_BLOCKS=8`。显存不足时使用一张卡并保持有效 batch：

```bash
CUDA_VISIBLE_DEVICES=0 python train_football_events.py \
  --config configs/football/dinov3_vitl16_stage3_roi_verifier.yaml \
  model.init_checkpoint=/path/to/stage2_best.pt 'gpu_ids=[0]' \
  train.per_gpu_batch_size=1 train.grad_accum_steps=64 \
  data.num_workers_per_gpu=0
```

接受条件：precision 比相同 stage-2 checkpoint 高至少 3pp 且 recall 下降不超过 0.5pp，
或 `R@P60` 至少提高 1pp。超过 recall 下降上限就停止 ROI 路径。

## Dense 评测与返回文件

默认 tolerance=5s；辅助 2s：

```bash
bash scripts/run_football_precision60_experiments.sh eval /path/to/best.pt candidate_tol5
MATCH_TOLERANCE_SEC=2 bash scripts/run_football_precision60_experiments.sh \
  eval /path/to/best.pt candidate_tol2
```

评测后 wrapper 会自动执行排除缺标 set-piece 的 PR 分析。阈值只能在独立 calibration
split 上选择；六视频 test 的 oracle 结果仅用于判断上限，不能作为最终部署阈值。

每轮返回：完整日志、保存的 `config.yaml`、`best_metrics.json`、所有
`metrics_epoch_*.json`、dense run 的 `run_config.json`、tol5 诊断目录下的
`summary_metrics_exclude_unlabeled_setpiece.json` 和 `per_video_per_class.csv`，以及前 50
step 和稳定阶段的：

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
```
