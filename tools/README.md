# 项目工具

UI 工具统一管理入口为 [`ui/`](ui/README.md)，包含工具清单、启动命令、数据准备、备份、排障和 8775 性能审查。

```bash
python tools/ui/manage.py list
python tools/ui/manage.py run event-review -- --help
```

| 目录 | 用途 |
| --- | --- |
| [`ui/`](ui/README.md) | 统一 UI 入口、注册表和审查文档 |
| [`football_event_review/`](football_event_review/README.md) | AI 结果初审、GT 差异查看、历史版本及数据准备 |
| [`football_final_qc/`](football_final_qc/README.md) | 独立终审、来源校验和严格导出 |
| [`football_missed_clips_ui/`](football_missed_clips_ui/README.md) | 漏标片段只读查看 |
| `bin/` | 已有辅助可执行文件（如 cloudflared） |

实现目录保留原路径；`scripts/watch_*review*` 等已有部署脚本仍可运行。新增 UI 在 `ui/catalog.json` 注册，并同步更新统一 README。运行数据、数据库、视频和凭据放在任务数据目录，不放入工具源码目录。
