# DINO 足球事件人工质检工具

> 多人 8775/v40 的启动、账号、视频准备和运维见 [UI 统一管理 README](../ui/README.md)，性能与可靠性问题见 [8775 审查](../ui/audits/8775_review_2026-09-08.md)。下文主要介绍历史基础版 `server.py`；其无登录、媒体限速及页面导出说明不适用于 v40。

这是一条本地运行的后处理链路：将 DINO 长视频评测结果和 frame detection 证据转换成审核候选，通过浏览器播放原始视频、跳转事件时间点、保留、删除或修正事件，最后导出经过人工质检的事件列表。

## 设计原则

- frame detection 只作为第二证据展示，不自动硬过滤，避免损失 DINO recall。
- 默认把间隔不超过 5 秒的同类相邻预测合并为一个审核候选，减少重复劳动。
- 原始预测永久保留；人工修改写入 SQLite，所有操作均有历史快照并支持撤销。
- 导出的 JSON/CSV 保留 `source_label/source_time_sec`，可统计人工删除率、改类率和时间修正量。
- 视频仅在本机按 HTTP Range 分段读取，不复制、不转码整段视频。

## 1. 生成审核 manifest

以下命令可直接适配现有最优 dual-token-fusion 评测输出：

```bash
python tools/football_event_review/build_review_manifest.py \
  --run-dir outputs/football_eval_runs/vitl16_e1_dual_token_fusion_d7_last_6videos_window_overlap_tol2_frame \
  --video-root /mnt/data_16t/football/raw_video_720P \
  --output outputs/football_event_review/dual_token_fusion_full_val/review_manifest.json \
  --candidate-merge-sec 5
```

只生成指定视频：

```bash
python tools/football_event_review/build_review_manifest.py \
  --run-dir <评测输出目录> \
  --video-root <原始视频或HQ视频目录> \
  --output <review_manifest.json> \
  --video-ids <video_id_1> <video_id_2>
```

如果需要逐个审核评测目录中的原始窗口预测，使用 `--candidate-merge-sec 0` 关闭相邻候选合并。

## 2. 启动 UI

```bash
python tools/football_event_review/server.py \
  --manifest outputs/football_event_review/dual_token_fusion_full_val/review_manifest.json \
  --db outputs/football_event_review/dual_token_fusion_full_val/reviews.sqlite3 \
  --host 127.0.0.1 \
  --port 8765
```

本机访问 `http://127.0.0.1:8765`。服务器远程运行时，用 SSH 端口转发：

```bash
ssh -L 8765:127.0.0.1:8765 <user>@<server>
```

服务建议绑定 `127.0.0.1`；不要直接暴露到公网，因为工具没有登录层。

## 3. 快捷键

- `Enter`：保留并进入下一个未审核候选
- `X` / `Delete`：删除并进入下一个未审核候选
- `1`–`6`：选择射门、扑救、任意球、点球、角球、射正
- `J` / `K`：上一个 / 下一个候选
- `R`：从事件前 4 秒重新播放当前候选
- `Space`：播放 / 暂停
- 时间轴滚轮缩放、拖动平移、点击跳转

## 4. 输出

页面右上角可导出 `reviewed_events.csv` 和 `reviewed_events.json`。默认导出已保留和已修改事件，删除及未审核候选不会进入最终结果。SQLite 数据库是完整审核审计记录，建议与导出文件一起保存。再次启动同一个 manifest 和数据库会恢复进度。

## 数据契约

每个视频评测目录至少需要 `predicted_events.csv` 和 `window_predictions.csv`。`frame_event_window_scores.csv` 可缺失，缺失时 frame detection 分数显示为 0。模型评测需加 `--save-frame-event-logits` 才能得到完整 frame detection 证据。适配器不依赖 checkpoint，不会重新做 DINO 前向。

生成逐窗口 FP 审核队列时使用 `--candidate-merge-sec 0 --evaluation-filter fp --match-tolerance-sec 2`。对缺少某类 GT 的视频，必须额外传入 `--exclude VIDEO_ID:LABEL`，避免把未标注事件错误地当成 FP。
