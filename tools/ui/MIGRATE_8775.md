# 8775 当前任务迁移操作手册

适用：remaining148、server_multiuser_v40.py。此文档未执行停服或迁移。

目标机须能被标注人员稳定访问；先让标注员测试目标机的视频下载，避免迁到同样拥堵的线路。当前媒体约 297 GiB，建议至少 400 GiB 可用空间。只运行审核服务不需要 GPU、PyTorch 或模型权重。使用 Linux、Python 3.10+（含 sqlite3/SSL）、rsync、SSH；目标机需有对应端口的访问许可。不要把旧 IP 的证书当成新地址的有效证书。

## 1. 在旧服务器设置变量

替换 NEW_USER 和 NEW_HOST。下列旧机命令须在同一个 shell 执行。

```bash
REVIEW_ROOT=/home/new_users/qiuqi/code/dinov3-main
REVIEW_DATA="$REVIEW_ROOT/outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902/review_platform_remaining148"
REVIEW_TARGET=NEW_USER@NEW_HOST
REVIEW_PY=/home/new_users/qiuqi/miniconda3/bin/python
export REVIEW_DATA
ssh "$REVIEW_TARGET" 'mkdir -p ~/football-review/app ~/football-review/data/media_faststart_v40 ~/football-review/tls; chmod 700 ~/football-review ~/football-review/data ~/football-review/tls; python3 --version; df -h ~'
```

## 2. 在线预传代码和视频（旧机执行，不停当前服务）

复制整个工具目录，它包含 v40 依赖的历史补丁链和静态资源，不能只复制入口 py。不要迁移旧 watcher，它硬编码了旧机器的 IP 和目录。

```bash
rsync -a --exclude='__pycache__/' "$REVIEW_ROOT/tools/football_event_review/" "$REVIEW_TARGET:football-review/app/"
rsync -a "$REVIEW_DATA/review_manifest.json" "$REVIEW_DATA/access_config.json" "$REVIEW_TARGET:football-review/data/"
rsync -a --partial --info=progress2 --bwlimit=4096 "$REVIEW_DATA/media_faststart_v40/" "$REVIEW_TARGET:football-review/data/media_faststart_v40/"
```

4096 的单位是 KiB/s，约 4 MiB/s，用于控制迁移占用；根据实际出口调整。297 GiB 在此上限下理论约 21 小时，可随时中断并重跑续传。不要直接无上限复制导致现有标注更卡。如两机可读同一块存储，可直接挂载媒体以减少传输，但必须确认读取链路不再依赖旧机拥堵出口。

保持 access_config 原样，账号密码、视频分配沿用原值。不要重新运行 create_multiuser_access.py。credentials_private.json 含明文密码，不是服务启动必需文件；需要留存时单独安全传递。

## 3. 目标机先准备路径和验证媒体

在目标机执行。仅修改 manifest 中本机视频路径，不修改视频 ID、事件 ID、时间或标签。

```bash
cd ~/football-review
python3 - <<'PY'
import json
from pathlib import Path
root=Path.home()/'football-review'
p=root/'data/review_manifest.json'
d=json.loads(p.read_text())
assert len(d['videos'])==148
for v in d['videos']:
    media=root/'data/media_faststart_v40'/f"{v['video_id']}.mp4"
    assert media.is_file() and media.stat().st_size>1024*1024, media
    v['video_path']=str(media)
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
import sys
sys.path.insert(0,str(root/'app'))
from mp4_layout import is_faststart
assert all(is_faststart(Path(v['video_path'])) for v in d['videos'])
import server_multiuser_v40
print('148 个视频存在、faststart 检查通过；服务模块导入成功')
PY
```

这一步只检查存在和布局。如需确认传输内容，旧机执行 `rsync -anc` 比对媒体：

```bash
rsync -anc --out-format='%n' "$REVIEW_DATA/media_faststart_v40/" "$REVIEW_TARGET:football-review/data/media_faststart_v40/"
```

无输出表示 rsync 校验未发现差异；该操作会顺序读取约 297 GiB，两端应安排在空闲时执行。

## 4. 最终切换：先清空浏览器保存队列，再停止旧服务

提前通知标注员结束当前操作，确认保存成功、待保存/冲突草稿均已处理，然后关闭旧页面。浏览器 localStorage 中的未提交草稿不在 SQLite 内，新 IP/域名无法自动读取它们。

旧机执行以下命令。它会检查近期登录活动及连接，核对 PID 身份，并先停 watcher，避免旧服务自动重启。若检查拒绝，待标注员退出后重新执行。

```bash
"$REVIEW_PY" - <<'PY'
import os,signal,sqlite3,subprocess,time
from pathlib import Path
p=Path(os.environ['REVIEW_DATA'])
with sqlite3.connect((p/'reviews.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as c:
    assert not c.execute('SELECT COUNT(*) FROM review_sessions WHERE expires_at>? AND last_seen_at>?',(time.time(),time.time()-60)).fetchone()[0], '最近60秒仍有活动，请稍后重试'
assert not subprocess.check_output(['ss','-Htn','state','established','( sport = :8775 )'],text=True).strip(), '仍有连接，请稍后重试'
pids=[]
for name,entry in [('watch_v40.pid','watch_remaining148_review_v40_service.py'),('server_v40.pid','server_multiuser_v40.py')]:
    pid=int((p/name).read_text())
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert entry in cmd, 'PID 身份不匹配'
    pids.append(pid)
for pid in pids:
    os.kill(pid,signal.SIGTERM)
    for _ in range(100):
        stat=Path(f'/proc/{pid}/stat')
        if not stat.exists() or stat.read_text().split(') ')[1].startswith('Z'): break
        time.sleep(.1)
    else: raise RuntimeError('旧进程未退出，停止迁移')
assert not subprocess.check_output(['ss','-Hltn','( sport = :8775 )'],text=True).strip(), '端口仍有监听'
print('旧服务和 watcher 已停止')
PY
```

即使停止后仍存在 WAL 文件，也通过 SQLite backup 导出最终完整快照，不只复制主 sqlite3 文件：

```bash
"$REVIEW_PY" - <<'PY'
import os,sqlite3
from pathlib import Path
p=Path(os.environ['REVIEW_DATA'])
out=p/'migration_final.sqlite3'
assert not out.exists(), '已有迁移快照，请另存或核对，不自动覆盖'
with sqlite3.connect((p/'reviews.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as src:
    with sqlite3.connect(out) as dst: src.backup(dst)
out.chmod(0o600)
with sqlite3.connect(out) as c:
    assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    print('最终状态统计',c.execute('SELECT status,count(*) FROM reviews GROUP BY status').fetchall())
PY
rsync -a "$REVIEW_DATA/migration_final.sqlite3" "$REVIEW_TARGET:football-review/data/reviews.sqlite3"
```

目标服务此时必须尚未运行。若曾用临时库测试，必须使用独立 staging 路径，不能把最终快照覆盖在运行中的库或残留的 staging WAL 上。

## 5. 目标机启动（HTTPS）

为新域名/IP配置客户端信任的证书，放到 `~/football-review/tls/server.crt` 和 `server.key`；私钥权限 600。生产入口必须使用 HTTPS，前端可靠保存依赖安全上下文。不要把旧证书复制过去就默认浏览器会信任新地址。

以下采用用户级 systemd，使用目标机的绝对路径生成配置，不依赖当前 SSH 会话。需 `systemctl --user` 可用；为退出登录后保持服务，由管理员启用该用户的 linger。

```bash
cd ~/football-review
chmod 600 data/access_config.json data/reviews.sqlite3 tls/server.key
mkdir -p ~/.config/systemd/user
REVIEW_NEW_ROOT="$HOME/football-review"
REVIEW_NEW_PY="$(command -v python3)"
cat > ~/.config/systemd/user/football-review.service <<EOF
[Unit]
Description=Football annotation review 8775

[Service]
WorkingDirectory=$REVIEW_NEW_ROOT
ExecStart=$REVIEW_NEW_PY $REVIEW_NEW_ROOT/app/server_multiuser_v40.py --manifest $REVIEW_NEW_ROOT/data/review_manifest.json --db $REVIEW_NEW_ROOT/data/reviews.sqlite3 --access-config $REVIEW_NEW_ROOT/data/access_config.json --proxy-root $REVIEW_NEW_ROOT/data/media_faststart_v40 --host 0.0.0.0 --port 8775 --tls-cert $REVIEW_NEW_ROOT/tls/server.crt --tls-key $REVIEW_NEW_ROOT/tls/server.key --require-faststart
Restart=on-failure
RestartSec=5
UMask=0077

[Install]
WantedBy=default.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now football-review.service
systemctl --user status football-review.service --no-pager
```

管理员按目标用户名执行 `sudo loginctl enable-linger TARGET_USER`。以上模板假设目标家目录不含空格。若无 systemd 用户服务，应改用该机器的服务管理方式，不要依赖裸 SSH 前台进程。

新 watcher 未复用，必须另行安排数据库周期备份（仍使用 SQLite backup API），例如目标机定时任务；至少每天备份并保留异机副本。避免备份写满视频盘。

## 6. 验收后分发新链接

目标机执行（`-k` 仅用于本机诊断，不代表用户浏览器证书有效）：

```bash
curl -ksS https://127.0.0.1:8775/healthz
curl -ksS https://127.0.0.1:8775/readyz
python3 - <<'PY'
import sqlite3
from pathlib import Path
p=Path.home()/'football-review/data/reviews.sqlite3'
with sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) as c:
    print(c.execute('PRAGMA integrity_check').fetchone())
    print(c.execute('SELECT status,count(*) FROM reviews GROUP BY status').fetchall())
PY
```

期望 health 的 media_revision 为 adaptive-forward-buffer-20260910；ready 为 148/148。核对状态统计和旧机最终快照一致。用真实标注员浏览器访问 `https://新地址:8775/`，重新登录原账号，验证：首次出画、远距离跳转、下一事件、队伍归属和一次真实标注成功保存。新地址通常需要重新登录。

切换完成后旧服务保持停止，不允许两套数据库分别接受标注。旧数据先保留，不删除。

## 回滚

新机尚未产生标注时，可以停止新服务，重启旧机原 watcher，并恢复旧链接。新机已产生标注时，不得直接恢复旧库：先暂停新机标注、备份新机最新库并迁回旧机，再启动旧服务，否则会丢失切换后的工作。账号配置与任务 ID 始终保持一致。
