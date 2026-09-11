# Stage2 在线两轮快速验证

按 2026-09-11 的新要求，当前快速验证使用在线 DINO 前向，不先构建或读取特征缓存。正式训练由用户在远端启动，本次只开发和执行工程测试。

## 配置

- 输入：720P、16 帧，保持原窗口、原始 patch、Stage1 热图及软位置先验。
- 数据：每轮全部 32,072 个训练窗口；每轮完整评测 5,618 个校准窗口和 5,919 个开发窗口，没有暗中缩小数据集。
- 参数：DINO 和 Stage1 定位模块冻结；在线生成当前 batch 的特征后训练适配器、原时序层和分类头。
- 四卡 DDP：每卡 micro-batch 2，累积 8 步，等效 batch 64。
- 预热：前 50 次优化器更新只更新适配器，随后联合更新时序层和分类头。全量每轮 502 次更新，第一轮约前 10% 为预热。
- 学习率：适配器 2e-4，时序层和分类头 2e-5，按两轮计划余弦衰减；其他损失权重沿用此前配置。
- 总轮数：2。每轮输出完整校准/开发指标；最终 checkpoint 只按校准集窗口宏 AP 选择，包含原模型 epoch0 回退候选。
- 除模型/优化器 checkpoint、配置、哈希、预测和指标外，不保存视觉特征文件。

与原六轮方案相比，除了不缓存，还明确改成了短预热和两轮学习率计划；不是旧六轮缓存方案的数值等价复现。在线方式省去全量缓存准备等待，但每轮都重复 DINO 前向，整体运行时间不保证更短。

## 远端命令

在同一路径与既有环境下运行。先把 GPU 列表替换成远端四张可用卡；不要根据示例编号推断它们当前空闲。

```bash
cd /home/new_users/qiuqi/code/dinov3-main
STAGE2_PY=/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python
ONLINE_OUT=/home/new_users/qiuqi/code/dinov3-main/outputs/football_localization_stage2/720p_online_softprior_joint_2ep_20260911

"$STAGE2_PY" -m football_events.stage2.online --preflight

mkdir -p "$ONLINE_OUT"
nohup env CUDA_VISIBLE_DEVICES=3,4,6,1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  "$STAGE2_PY" -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  --module football_events.stage2.online \
  > "$ONLINE_OUT/train.log" 2>&1 < /dev/null &
echo "Launcher PID: $!"
```

新入口直接开始在线训练，不要用旧 `--phase pipeline` 启动本轮。当前默认配置是 `configs/football/stage2_online_softprior_2ep_20260911.json`。新入口不需要旧基础特征 arrays，但仍需要清单中指定的视频和三个只读源检查点。

查看进度与结果：

```bash
tail -n 50 -f "$ONLINE_OUT/train.log"
cat "$ONLINE_OUT/heartbeat.json"
cat "$ONLINE_OUT/epoch_001.json"
cat "$ONLINE_OUT/epoch_002.json"
cat "$ONLINE_OUT/FINAL_REPORT.md"
```

每个完整 epoch 训练和评测结束后保存 `resume.pt`。确认旧进程已结束后重跑同一命令可继续；中断的 epoch 从上一个完整 epoch 重跑。改变 world size、累积步数或总 epochs 应使用新配置和新输出目录，不能混用原实验状态。

## 判断标准

第一轮先看校准及开发窗口 AP、逐类 P/R 和误报窗口/小时，并同时查看原阈值结果。最终输出视频级配对 bootstrap，以及选中训练模型、宏 AP/precision 提升、各类 recall 不降超过 1 个百分点且误报不增加、AP 差区间下界大于 0 等检查。

每轮在同一次视觉前向上计算原模型预测、当前模型预测及空输入、半帧缺失、局部时间反转检查。跨窗口错配用于训练正则；这个快速版本不宣称已完成全量自然无球/灯具/高球分层鲁棒性验证。

两轮有改善只能作为进一步验证的信号，两轮无改善也不能直接判定技术路线无效。开发集为历史使用过的视频，指标为窗口 AP，不能写成独立测试集事件 spotting mAP。

## 已完成的工程验证

- 真正的数据/权重存在性预检查通过，不依赖特征缓存。
- CPU/Gloo 双进程验证通过：每个真实训练行只出现一次、尾部填充不重复样本、梯度累积同步、某个 rank 只有填充时仍能同步、预热期间时序参数和优化器状态均不更新、之后分类头更新、教师保持不变。
- CPU 合成帧完整流程通过：两轮训练、逐轮预测/报告、checkpoint、完成后重复进入不重训，未生成 cache/arrays 目录。
- 本次没有空闲 GPU 可用于新的真实 720P 在线多卡全流程 smoke；复用了此前已验证的真实特征提取器。上述 CPU 测试不能替代远端首次运行的 GPU 验证，也不构成模型效果证据。

测试入口：`tests/stage2_online_ddp_smoke.py`、`tests/test_stage2_online_workflow.py`。
