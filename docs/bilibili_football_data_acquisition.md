# Bilibili 足球视频数据储备

该工具通过 `yt-dlp` 的 Bilibili 公开搜索提取器发现视频，完成标题过滤、BV 号去重、UP 主配额控制、限速下载和 JSONL 清单生成。它不绕过登录、付费、地域限制或 DRM。

## 使用边界

- 只采集公开可访问且你有权用于目标研究或训练用途的视频。
- 下载前应人工检查 UP 主授权、视频声明和适用的平台规则；不要重新分发原视频。
- `weak_event_labels` 只表示视频被哪个搜索词找到，不是事件 GT。一个“射门集锦”视频也包含大量非射门帧。
- 监督训练前必须进一步做候选切片和人工复核。未复核数据更适合作为足球域自监督预训练储备。

## 依赖

```bash
python -m pip install -U yt-dlp
ffmpeg -version
```

`yt-dlp` 更新较频繁，Bilibili 页面变化后应先升级它，不要在本工具中硬编码平台私有 API。

## 1. 检查采集计划

此命令不联网，也不下载：

```bash
python scripts/acquire_bilibili_football_videos.py \
  --config configs/football/bilibili_football_acquisition.yaml \
  plan
```

默认数据根目录是：

```text
/mnt/data_16t/football/qiuqi/bilibili_football
```

可通过 `--output-root` 覆盖。

## 2. 发现和筛选公开元数据

```bash
python scripts/acquire_bilibili_football_videos.py \
  --config configs/football/bilibili_football_acquisition.yaml \
  discover
```

主要输出：

```text
metadata/discovered.jsonl       所有查询命中的行，允许重复 BV 号
metadata/accepted.jsonl         过滤、BV 去重和 UP 主限额后的清单
metadata/rejected.jsonl         被拒绝的视频及原因
metadata/discovery_summary.json 标签分布、拒绝原因和查询状态
logs/discover_*.stderr.log      每个查询的 yt-dlp 错误日志
```

建议先检查 `accepted.jsonl` 的标题、时长、UP 主分布和用途授权，再下载。

## 3. 小批量验证下载

只有明确确认所选公开视频可用于目标用途后，才传入确认参数：

```bash
python scripts/acquire_bilibili_football_videos.py \
  --config configs/football/bilibili_football_acquisition.yaml \
  download \
  --labels shot,save,penalty \
  --limit 20 \
  --rights-acknowledged
```

确认格式、画质、内容命中率和存储开销后，再去掉 `--limit`。下载采用低并发和随机等待；`metadata/download_archive.txt` 使重复运行自动跳过已完成视频，并能从中断处继续。

下载输出：

```text
videos/<BVID>/<BVID>.<ext>
videos/<BVID>/<BVID>.info.json
metadata/selected_for_download.jsonl
metadata/downloaded.jsonl
metadata/download_archive.txt
metadata/download_summary.json
logs/download.log
```

## 数据进入训练前的建议流程

1. 将完整比赛和事件视频用于足球域自监督预训练，学习球场、转播视角、球员和比赛节奏。
2. 对事件类视频按 5 秒 stride 生成 10 秒候选窗口。
3. 用当前 DINO 模型、字幕/标题时间信息或运动响应做候选排序，但不直接生成正标签。
4. 人工确认 `shot/save/set_piece/negative/unknown`，同时记录事件 anchor 和可信度。
5. 按原视频或比赛维度划分 train/val，禁止同一视频的相邻切片跨 split。
6. 对 UP 主、赛事、相机类型和分辨率做分层统计，避免模型只学习单一转播风格。

查询配置同时覆盖完整比赛、业余/青少年比赛和关键事件。完整比赛提供真实高比例负样本及上下文；集锦提供事件密度，但不能代替人工时序标注。
