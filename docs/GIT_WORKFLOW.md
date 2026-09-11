# Git 管理范围

目标仓库：https://github.com/caoqiuqi-vv/football_event_vlm

只管理源代码、测试、可复现配置、数据划分 ID 清单和必要文档。模型权重、数据集、视频、特征、预测、标注数据库、outputs、缓存、日志、实验自动产物和私有配置均留在存储盘，不通过 Git/Git LFS 上传。

`.gitignore` 防止新产物被加入；`scripts/check_source_only_index.py` 检查实际暂存内容，可发现已被跟踪的大文件及常见凭据格式。它是启发式检查，不能替代人工确认。配置用 JSON 的文件保留在 configs 下；如确需新的非配置 JSON 资源，应添加精确例外，不能直接批量强制添加。

```bash
git status --short
git add <代码或配置路径>
python3 scripts/check_source_only_index.py
git diff --cached --stat
git commit -m "描述具体变更"
git push
```

新仓库使用当前源码的独立根提交，避免携带旧仓库约 11 GiB 的历史对象。旧本地历史仍保留，不使用 `push --mirror` 或 `push --all` 推送它；旧 remote 保留为 legacy-origin。

换机器时 clone 本仓库，再单独挂载数据/模型存储、配置本机路径。Git clone 不会包含已经迁往 /mnt/data_16t/qiuqi/outputs 的文件。8775 服务迁移步骤见 ../tools/ui/MIGRATE_8775.md。

上游 DINOv3 许可保留在 licenses/DINOV3_LICENSE.md，项目中的上游代码继续受其约束。
