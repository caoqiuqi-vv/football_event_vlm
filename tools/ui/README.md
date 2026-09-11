# UI 工具统一管理

这里集中管理工程中的浏览器 UI：AI 结果人工初审、独立终审、漏标片段浏览和 GT 差异查看。源码仍位于原工具目录，统一入口通过独立 Python 进程启动，避免历史版本的全局补丁在同一进程中相互污染，也兼容已有部署脚本与测试导入。

## 目录与清单

```text
tools/
├── README.md
├── ui/
│   ├── README.md                    # 使用与运维说明
│   ├── catalog.json                 # 工具注册表
│   ├── manage.py                    # list / run 统一入口
│   ├── audit_8775.py                # 临时数据并发、事务与 HTTP 诊断
│   ├── audit_8775_frontend.cjs       # 实际生成 JS 的乱序响应诊断
│   └── audits/                      # 审查报告与脱敏证据
├── football_event_review/           # 初审、GT 差异、数据准备和历史版本
├── football_final_qc/               # 终审和导出
└── football_missed_clips_ui/         # 漏标片段只读浏览
```

| 管理名称 | 默认端口 | 用途与权限 | 输入 / 输出 |
| --- | --- | --- | --- |
| `event-review` | 8775 | v40 多人初审，账号密码、会话及整视频独占分配 | manifest + access config + 媒体 → 初审 SQLite 与审计历史 |
| `final-qc` | 8776 | GT/初审差异的独立终审，共享 token；reviewer 为审计姓名 | cases 快照 + 媒体 → 独立终审 SQLite、严格导出 JSON |
| `missed-clips` | 8777 | token 保护的只读漏标片段浏览 | `manifest.json` + `clips/`，无标注写入 |
| `gt-discrepancy` | 8774 | 原 GT ↔ 人工结果的只读差异查看 | 单视频 snapshot JSON；token 可选，远程使用时应设置 |
| `basic-review` | 8765 | 历史基础审核器，无登录层 | 基础 manifest → SQLite、页面 JSON/CSV 导出 |

端口来自对应入口默认值，可用 `--port` 覆盖；不是所有端口都一定在运行。历史 `server_hierarchical_v*.py`、`server_public_token_v*.py`、`server_multiuser_v30…v39.py` 是继承/补丁链依赖，不应按“旧文件”直接删除。`scripts/watch_full_review_annotation_pipeline.py` 另有默认 8773 的 v31 流程，不是当前 8775/v40 服务。`scripts/analyze_whistle_setpiece_ui.py` 是离线统计脚本，不是 UI 服务。

## 环境与统一启动

从项目根目录执行示例。UI 服务使用 Python 标准库（包括 SQLite），不需要启动模型推理或安装前端构建系统。建议使用项目现有 Python 3.10+ 环境；本次验证为 Python 3.13.13。服务器需可读取所有视频路径，并可写入对应数据库目录。准备视频需要 `ffmpeg`/`ffprobe`，现有压缩代理生成器还需要 CUDA/NVENC；这些不是浏览已有视频的运行依赖。前端诊断额外使用 Node.js。

```bash
python tools/ui/manage.py list
python tools/ui/manage.py run event-review -- --help
python tools/ui/manage.py run final-qc -- --help
```

若 shell 没有 `python`，激活项目 conda 环境，或使用本机已验证解释器 `/home/new_users/qiuqi/miniconda3/bin/python`。入口使用同一个解释器启动原脚本，保留调用者的工作目录、退出码和信号行为。`--` 后的参数原样传递。入口不自动创建账号、选择生产数据库或启动守护进程。

## 8775：AI 结果多人初审

### 输入准备

已有任务应复用匹配的 manifest、数据库和 access config。新任务可参考 [原工具 README](../football_event_review/README.md) 和 `build_review_manifest.py --help` 构建基础候选；需要 GT 强制质检、多标签与 dense 证据时，使用目录内 `build_full_dense_review_queue.py` / `build_multilabel_segment_manifest.py` 对应的数据流程，不能仅改 manifest 文件名代替转换。

manifest 顶层包含 `videos`、标签配置；视频含 `video_id`、`video_path`、`duration_sec`、`events`；事件包含稳定 `id`、`video_id`、标签、分数、时间及所属 `segment_id` 等。同一段允许多个标签。路径必须指向服务器可读文件；人工操作以事件 ID 与 revision 关联。不要将不相关任务的 manifest 复用到同一数据库。

以下使用新建任务目录，路径按实际数据替换：

```bash
REVIEW_DATA=outputs/football_event_review/my_review
python tools/football_event_review/create_multiuser_access.py \
  --manifest "$REVIEW_DATA/review_manifest.json" \
  --output-config "$REVIEW_DATA/access_config.json" \
  --output-credentials "$REVIEW_DATA/credentials_private.json" \
  --base-url http://127.0.0.1:8775 \
  --reviewers '质检员1,质检员2,质检员3,质检员4'
```

当前账号生成脚本严格要求 **4 个姓名**，按视频时长平衡分配完整视频；服务本身不限定 4 人，但要求配置内每个视频恰好分配一次且与 manifest 完整对应。修改账号数需要另行生成符合相同契约的配置。不要对已有任务重复执行账号生成命令，否则会覆盖密码和分配。`credentials_private.json` 包含明文密码，不提交 Git；access config 存储盐与哈希，也作为私有配置管理。

### 媒体准备与服务

原画质 faststart 仅重新封装 MP4：

```bash
python tools/football_event_review/build_original_faststart_streams.py \
  --manifest "$REVIEW_DATA/review_manifest.json" \
  --output-dir "$REVIEW_DATA/media_faststart_v40" --jobs 1

python tools/ui/manage.py run event-review -- \
  --manifest "$REVIEW_DATA/review_manifest.json" \
  --db "$REVIEW_DATA/reviews.sqlite3" \
  --access-config "$REVIEW_DATA/access_config.json" \
  --proxy-root "$REVIEW_DATA/media_faststart_v40" \
  --host 127.0.0.1 --port 8775
```

访问 `http://127.0.0.1:8775/`，选择账号并输入密码。每人只访问所分配的视频。HTTPS 另加 `--tls-cert /path/server.crt --tls-key /path/server.key`，必须成对提供。远程本地绑定可用 `ssh -L 8775:127.0.0.1:8775 user@server`；转发目标应与服务实际绑定地址一致。当前部署绑定特定网卡地址，不能假设它也监听 loopback。

v40 优先使用 `--proxy-root/VIDEO_ID.mp4`（存在且大于 1 MiB），否则回退到 manifest 的 `video_path`。HTTP 响应 `X-Review-Media-Variant: proxy` 仅表示来自优先目录，并不代表压缩画质。`quality=original` 可强制回退源路径。faststart 不降低码率，也不缩短 GOP；更低带宽可另用 `build_streaming_proxies.py` 离线生成压缩副本，参数以 `--help` 为准，该脚本会产生新 manifest。当前页面没有自适应码率切换；压缩质量需要人工验证小球与事件细节后再选用。

### 标注与持久化

选择视频及事件，播放证据区间，确认多个标签、细分类、队伍及场地区域；视频级队伍设置通常每个视频确认一次。常用快捷键：Enter 确认并继续、X/Delete 删除、Q 待二次确认、Space 播放/暂停。其余快捷键以当前页面提示为准，基础版快捷键不全部适用于 v40。

2026-09-09 18:34（北京时间）已上线事件归属新规则，前后端均已生效；请刷新已有页面。规则如下：

| 已选事件 | 队伍颜色 | 左／右半场 |
| --- | --- | --- |
| 扑救 | 不显示、不要求 | 必选左半场或右半场 |
| 射门、角球、任意球、点球 | 必选队伍 A 或 B 的颜色 | 不显示、不要求 |
| 界外球 | 不显示、不要求 | 不显示、不要求，可直接提交 |

同片段多标签分别校验，例如“射门＋扑救”须选择射门队伍和扑救所在半场。上述事件不再单独选择目标球门。上线前已用真实数据副本预启动验证，切换后标注表与备份一致；见 [发布验收记录](audits/8775_attribution_2026-09-09.json)。提交时清理隐藏归属字段，避免携带旧队伍或旧区域；已有历史标注不批量改写。“待二次确认”仍可保存不完整草稿。更新后请刷新页面；旧保存队列若缺少新必填信息，会保留草稿，补齐后重新提交。

“已加入保存队列”表示操作已同步持久化在当前浏览器；状态面板为“全部操作已保存”才表示队列已获得服务端确认。网络请求超时 10 秒后以相同 operation_id 重试，退避最长 10 秒；每个账号/任务最多保留 100 条待处理操作。刷新、关页重开或重新登录后，队列会恢复。队列按数据库 namespace 和账号隔离；同片段的待发送操作未处理前不能重复编辑覆盖。

400/409 等明确失败会保留原草稿，点击“读取最新结果并重新编辑”，核对后重新提交；也可以明确放弃失败草稿。401 暂停发送，登录后重试。浏览器存储满或不可用时会拒绝操作，不提前标记成功。队列保存在同一浏览器、同一站点的 localStorage；清除站点数据或更换浏览器不会迁移草稿。清理浏览器前先确认队列为空。

服务端将全部标签、归属、历史、审计与幂等回执放入单个事务；失败完整回滚，重复 operation_id 不重复写入，复用 ID 携带不同内容返回冲突。撤销也使用 revision 检查，版本号保持递增。v40 保留原有 GT 删除播放确认保护。待二次确认结果不会作为已确认训练标注导出，`/api/export` 仍对审核员返回 403。

视频加载使用请求代次和取消控制，仅最新选择可以更新画面；加载中禁止保存。视频失败后可点击“重新加载视频”，恢复当前事件与时间位置。详见 [v40 修复与验证记录](audits/8775_v40_fixes_2026-09-08.md)。

### 2026-09-09 播放卡顿修复

同一视频内跳转直接更新播放器时间，复用已加载的 MP4 索引；仅首次加载、切换视频或明确重试时重载资源。服务完整响应浏览器请求的字节范围，不再把 Range 固定截为 8 MiB；传输仍以固定内存块发送，并保留 16 个媒体传输槽和 128 个总连接槽。视频 JSON 按浏览器协商启用 gzip，减少首次加载数据量。缓冲持续 10 秒时显示“视频缓冲慢，重试”，不会自动反复重载。

原码率与画质保持不变。长视频索引本身较大，首次打开仍取决于带宽；之后切换事件不再重复下载索引。详见 [故障分析与验证](audits/8775_streaming_2026-09-09.md)。

## 8776：独立终审

终审从初审数据库与 GT 生成 cases 快照，写独立数据库：

```bash
FINAL_DATA=outputs/football_final_qc/my_review
python tools/football_final_qc/build_cases.py \
  --manifest "$REVIEW_DATA/review_manifest.json" \
  --review-db "$REVIEW_DATA/reviews.sqlite3" \
  --gt-run-dir /path/to/gt_run --output "$FINAL_DATA/cases.json"
python tools/ui/manage.py run final-qc -- \
  --cases "$FINAL_DATA/cases.json" --db "$FINAL_DATA/final.sqlite3" \
  --proxy-root "$REVIEW_DATA/media_faststart_v40" \
  --host 127.0.0.1 --port 8776 --token "$FINAL_QC_TOKEN"
python tools/football_final_qc/export_final.py \
  --cases "$FINAL_DATA/cases.json" --db "$FINAL_DATA/final.sqlite3" \
  --output "$FINAL_DATA/final_annotations.json"
```

启动前通过私有环境设置非空 `FINAL_QC_TOKEN`。页面 URL 为 `http://127.0.0.1:8776/?token=<令牌>`；API 也接受 `Authorization: Bearer <令牌>`。token 会作为现有服务启动参数传入，本管理器未改变其凭据机制。需滚动更新快照时使用 `watch_cases.py --help`，默认间隔 60 秒。来源变化会使旧决定 stale；初审未完成则 provisional，严格导出拒绝未完成或不一致结果。不要用 `--allow-incomplete-video` 绕过正式交付检查。详见 [终审 README](../football_final_qc/README.md)。

## 8777 与 8774：只读检查

```bash
python tools/ui/manage.py run missed-clips -- \
  --root /path/to/missed_clips --host 127.0.0.1 --port 8777 \
  --token "$MISSED_CLIPS_TOKEN"
python tools/ui/manage.py run gt-discrepancy -- \
  --snapshot /path/to/discrepancy_snapshot.json \
  --host 127.0.0.1 --port 8774 --token "$FOOTBALL_DISCREPANCY_TOKEN"
```

分别预先设置非空令牌；页面用 `/?token=<令牌>` 访问。漏标根目录需要 `manifest.json`（含 `clips`，条目含 `filename`）和 `clips/` 视频文件，`source_video` 不在公开 manifest 返回。差异快照由 `build_gt_human_discrepancy_snapshot.py` 生成，包含 `video_path`、视频/差异/汇总数据，服务器启动时加载；更新快照后需重启该只读服务。

## 运维、备份与排障

2026-09-08 观察到 8775/v40、8776、8777 和本地 8774 监听。8775 已完成 v40 切换，148/148 个 faststart 媒体就绪，标注表与切换备份一致；见 [发布验收记录](audits/8775_v40_deployment_2026-09-08.json)。8775 当前数据目录为 `outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902/review_platform_remaining148/`；它是环境记录，不是新任务默认目录。配套守护入口为 `scripts/watch_remaining148_review_v40_service.py`，检查全部 faststart 媒体后可用 `--activate` 切换经身份校验的旧 8775 服务；它会停止对应旧 watcher、做切换前后备份，再启动 v40，随后每 30 秒检测健康、连续三次失败重启、每 10 分钟备份。它只适用于此固定任务，不是通用启动命令。

当前守护脚本会切换旧服务版本，启动它会操作现有服务。已有监听时不要再启动第二个实例共享初审数据库；v40 的 BEGIN IMMEDIATE 和幂等记录已通过两个 Store 实例同时写入测试，但完整多 worker 容量尚未验收，当前部署仍使用单进程多线程。`/healthz` 返回存活和版本，`/readyz` 验证数据库元数据及实际媒体 faststart 覆盖，未就绪返回 503。启动参数 `--require-faststart` 可在覆盖不足时拒绝启动。8774 没有同样的健康接口。

使用 SQLite 在线 backup 获得一致副本，不要只复制活跃 WAL 数据库的主文件。示例在替换路径后执行：

```python
import sqlite3
from contextlib import closing
from pathlib import Path

source = Path('/path/to/reviews.sqlite3').resolve()
destination = Path('/path/to/backup/reviews_YYYYMMDD_HHMMSS.sqlite3')
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists():
    raise FileExistsError(destination)
with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
    with closing(sqlite3.connect(destination)) as dst:
        src.backup(dst)
```

备份同时记录 manifest、配置版本和媒体映射；凭据单独私有保管。恢复时先停止写入服务，保留当前库及关联文件，选用已验证备份路径重新启动，不将旧主文件覆盖到仍活跃的 WAL/SHM 上。

| 现象 | 检查 |
| --- | --- |
| 登录等待，健康检查正常 | 确认已刷新到 v40；旧 v39 的重定向边界问题已修复，核对健康版本、证书及网络 |
| 未分配视频 / 403 | 账号与视频唯一分配；8775 禁止审核员导出，属预期行为 |
| 冲突 / 保存失败 | 在保存队列中读取最新结果并恢复草稿；明确失败会回滚，相同 ID 重试不会重复提交 |
| 播放慢 / 跳转慢 | 确认实际媒体路径、moov 位置、Range 206、文件码率、客户端带宽及等待时间 |
| 源视频缺失 | manifest 的绝对路径与服务读取权限；优先目录缺文件会静默回退 |
| 数据库锁等待 | v40 认证只读，会话活跃时间在后台批量更新；检查其他写进程与磁盘延迟 |
| 页面内容与选中视频不一致 | v40 已加请求代次保护；确认使用最新脚本与正确版本，保留控制台错误用于排查 |

## 审查复现与维护

```bash
python tools/ui/audit_8775.py --version v40 --output /tmp/ui-audit-8775/results.json
node --check /tmp/ui-audit-8775/results.effective.js
node tools/ui/audit_8775_frontend.cjs /tmp/ui-audit-8775/results.effective.js
python -m unittest discover -s tests -p test_football_event_review_reliable.py
node --test tests/test_football_review_outbox.cjs
```

诊断仅创建临时合成数据，监听随机 loopback 端口；不会打开生产数据库或对 8775 做压测。输出包括诊断 JSON 和最终拼接的 JavaScript，已知问题会以结果字段呈现，程序成功退出不代表被审工具没有问题。该测量不包含浏览器视频解码、公网 RTT、TLS 和真实多人操作节奏。

新增工具时在 `catalog.json` 添加唯一名称、默认端口、项目内 `tools/` 入口及说明，并补齐本 README 的输入、认证与输出说明。升级初审版本时验证整条 HTML/JS 补丁链；更新注册表并不改变已有 watcher 的版本配置，实际发布需单独安排。


真实浏览器回归可运行 `python tests/run_reliable_browser_test.py`，需要环境中已安装 Node Playwright 与浏览器。可用 `REVIEW_PLAYWRIGHT_MODULE` 指定已安装包的绝对路径、`PLAYWRIGHT_BROWSERS_PATH` 指定已有浏览器目录。此脚本只启动临时服务、使用合成账号和 VP9/MP4 视频，不访问生产 8775。
