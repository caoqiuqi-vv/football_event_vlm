if ! pgrep -a git | grep -Eq 'gc|repack|fetch|clone'; then rm -f .git/objects/pack/tmp_* && git gc --prune=now && echo "清理完成: $(du -sh .git | cut -f1)"; else echo "有 git 进程在跑,跳过"; fi

