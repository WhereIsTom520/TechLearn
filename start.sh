#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

SERVER_TOML="$ROOT/server.toml"
[[ -f "$SERVER_TOML" ]] || {
  echo "[ERROR] Missing server.toml" >&2
  exit 1
}

COMMAND="${1:-}"
if [[ -z "$COMMAND" ]]; then
  cat <<'EOF'
Browser Agent Stage 1 — 傻瓜版 v0.1.6

只需编辑 server.toml，然后运行：
  ./start.sh setup   # 首次创建/修复环境
  ./start.sh check   # 完整冒烟检查
  ./start.sh probe   # 8卡两次PPO更新
  ./start.sh train   # 正式训练/按配置自动续训
  ./start.sh resume  # 强制从现有检查点恢复
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

# 依赖零第三方包的最小TOML字符串读取，仅用于找到Python和venv。
# 完整TOML解析、类型校验和Schema校验由启动器完成。
toml_string() {
  local section="$1"
  local key="$2"
  local fallback="$3"
  local value
  value="$(awk -v section="$section" -v key="$key" '
    BEGIN { active=0 }
    /^[[:space:]]*\[/ {
      line=$0
      gsub(/[[:space:]]/, "", line)
      active=(line == "[" section "]")
      next
    }
    active && $0 ~ "^[[:space:]]*" key "[[:space:]]*=" {
      line=$0
      sub(/^[^=]*=/, "", line)
      sub(/[[:space:]]*#.*/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      if (line ~ /^\".*\"$/) {
        sub(/^\"/, "", line)
        sub(/\"$/, "", line)
      }
      print line
      exit
    }
  ' "$SERVER_TOML")"
  printf '%s\n' "${value:-$fallback}"
}

SYSTEM_PYTHON="${PYTHON_BIN:-$(toml_string environment python python3)}"
VENV_DIR="$(toml_string project venv_dir .venv)"
BOOTSTRAP_DIR="$(toml_string project bootstrap_venv_dir .easy_bootstrap)"
PIP_INDEX="$(toml_string environment pip_index '')"

[[ "$VENV_DIR" = /* ]] || VENV_DIR="$ROOT/$VENV_DIR"
[[ "$BOOTSTRAP_DIR" = /* ]] || BOOTSTRAP_DIR="$ROOT/$BOOTSTRAP_DIR"

LAUNCHER="$ROOT/scripts/easy_launcher_bootstrap.py"
[[ -f "$LAUNCHER" ]] || {
  echo "[ERROR] Missing $LAUNCHER" >&2
  exit 1
}

# 只有健康的项目venv才直接使用。残缺venv交给启动器自动重建。
if [[ -x "$VENV_DIR/bin/python" ]] && \
   "$VENV_DIR/bin/python" -c \
     'import tomli, tomli_w, playwright, torch, browser_agent' \
     >/dev/null 2>&1
then
  exec "$VENV_DIR/bin/python" "$LAUNCHER" "$@"
fi

command -v "$SYSTEM_PYTHON" >/dev/null 2>&1 || {
  echo "[ERROR] Python not found: $SYSTEM_PYTHON" >&2
  exit 1
}

# Python 3.11+自带tomllib；Python 3.10若已有tomli也可直接启动。
if "$SYSTEM_PYTHON" -c \
   'try:
 import tomllib
except ModuleNotFoundError:
 import tomli' >/dev/null 2>&1
then
  exec "$SYSTEM_PYTHON" "$LAUNCHER" "$@"
fi

# 首次Python 3.10启动的独立Bootstrap；它不是最终项目环境。
echo "[BOOTSTRAP] Installing the TOML parser into $BOOTSTRAP_DIR"
rm -rf "$BOOTSTRAP_DIR"
"$SYSTEM_PYTHON" -m venv "$BOOTSTRAP_DIR"

PIP_ARGS=()
if [[ -n "$PIP_INDEX" ]]; then
  PIP_ARGS=(-i "$PIP_INDEX")
fi

"$BOOTSTRAP_DIR/bin/python" -m pip install \
  --disable-pip-version-check \
  'tomli>=2.2,<3' \
  "${PIP_ARGS[@]}"

exec "$BOOTSTRAP_DIR/bin/python" "$LAUNCHER" "$@"
