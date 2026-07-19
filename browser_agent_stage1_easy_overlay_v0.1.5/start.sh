#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  cat <<'EOF'
Browser Agent Stage 1 — 傻瓜版

只需编辑 server.toml，然后运行：
  ./start.sh setup   # 首次创建环境
  ./start.sh check   # 完整冒烟检查
  ./start.sh probe   # 8卡两次PPO更新
  ./start.sh train   # 正式训练/自动续训
  ./start.sh resume  # 强制从检查点恢复
  ./start.sh status  # 查看状态
EOF
  exit 0
fi

case "$COMMAND" in
  setup|check|probe|train|resume|status) ;;
  *)
    echo "[ERROR] Unknown command: $COMMAND" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi

exec "$PYTHON_BIN" "$ROOT/scripts/easy_launcher.py" "$@"
