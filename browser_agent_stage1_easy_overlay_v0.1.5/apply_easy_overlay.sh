#!/usr/bin/env bash
set -Eeuo pipefail

OVERLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_ZIP="${1:-}"
OUTPUT_DIR="${2:-$PWD}"
VERSION="v0.1.5-easy8a100"

fail() {
  echo "[ERROR] $*" >&2
  exit 1
}

if [[ -z "$BASE_ZIP" ]]; then
  for candidate in \
    "$PWD/browser_agent_stage1_v0.1.4.zip" \
    "$OVERLAY_DIR/browser_agent_stage1_v0.1.4.zip" \
    "$OVERLAY_DIR/../browser_agent_stage1_v0.1.4.zip"
  do
    if [[ -f "$candidate" ]]; then
      BASE_ZIP="$candidate"
      break
    fi
  done
fi

[[ -n "$BASE_ZIP" ]] || fail "Place browser_agent_stage1_v0.1.4.zip beside this script or pass its path"
[[ -f "$BASE_ZIP" ]] || fail "Base ZIP not found: $BASE_ZIP"
command -v python3 >/dev/null 2>&1 || fail "python3 is required"
command -v unzip >/dev/null 2>&1 || fail "unzip is required"

BASE_ZIP="$(cd "$(dirname "$BASE_ZIP")" && pwd)/$(basename "$BASE_ZIP")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

EXTRACT="$WORK/extract"
STAGE="$WORK/browser_agent_stage1_easy_${VERSION}"
mkdir -p "$EXTRACT" "$STAGE"

printf '%s\n' "============================================================"
printf '%s\n' "Browser Agent Stage 1 傻瓜版构建器"
printf '%s\n' "Base: $BASE_ZIP"
printf '%s\n' "Output: $OUTPUT_DIR"
printf '%s\n' "============================================================"

echo "[1/7] Extracting v0.1.4..."
unzip -q "$BASE_ZIP" -d "$EXTRACT"

BASE_ROOT=""
while IFS= read -r candidate; do
  BASE_ROOT="$(dirname "$candidate")"
  break
done < <(find "$EXTRACT" -type f -name pyproject.toml | sort)

[[ -n "$BASE_ROOT" ]] || fail "pyproject.toml was not found in the base ZIP"
for required in config.toml scripts/train.py run_train_8xa100.sh run_smoke_linux.sh; do
  [[ -e "$BASE_ROOT/$required" ]] || fail "Base package is missing $required"
done

echo "[2/7] Copying the complete base project..."
cp -a "$BASE_ROOT"/. "$STAGE"/
rm -rf \
  "$STAGE/.venv" \
  "$STAGE/runtime" \
  "$STAGE/runs" \
  "$STAGE/.pytest_cache"
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGE" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.log' \) -delete

echo "[3/7] Installing the unified TOML launcher..."
cp "$OVERLAY_DIR/server.toml" "$STAGE/server.toml"
cp "$OVERLAY_DIR/start.sh" "$STAGE/start.sh"
cp "$OVERLAY_DIR/README_EASY_CN.md" "$STAGE/README_EASY_CN.md"
mkdir -p "$STAGE/scripts"
cp "$OVERLAY_DIR/scripts/easy_launcher.py" "$STAGE/scripts/easy_launcher.py"

cat > "$STAGE/tomllib.py" <<'PY'
"""Python 3.10 compatibility wrapper for code that imports tomllib."""
from tomli import TOMLDecodeError, load, loads
__all__ = ["TOMLDecodeError", "load", "loads"]
PY

printf '%s\n' "$VERSION" > "$STAGE/EASY_RELEASE_VERSION.txt"
chmod +x \
  "$STAGE/start.sh" \
  "$STAGE/scripts/easy_launcher.py" \
  "$STAGE/run_train_8xa100.sh" \
  "$STAGE/run_smoke_linux.sh"

echo "[4/7] Adapting project metadata for Python 3.10..."
python3 - "$STAGE/pyproject.toml" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text, count = re.subn(
    r'requires-python\s*=\s*"[^"]+"',
    'requires-python = ">=3.10"',
    text,
    count=1,
)
if count != 1:
    raise SystemExit("Unable to patch requires-python in pyproject.toml")
path.write_text(text, encoding="utf-8")
PY

echo "[5/7] Running build-time syntax checks..."
bash -n "$STAGE/start.sh" "$STAGE/run_train_8xa100.sh" "$STAGE/run_smoke_linux.sh"
python3 -m py_compile "$STAGE/scripts/easy_launcher.py" "$STAGE/tomllib.py"

# The full project is compiled later inside the project environment because
# the original package intentionally depends on PyTorch and Playwright.

echo "[6/7] Generating content manifest..."
rm -f "$STAGE/PACKAGE_CONTENT_SHA256SUMS.txt"
(
  cd "$STAGE"
  find . -type f ! -name PACKAGE_CONTENT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > PACKAGE_CONTENT_SHA256SUMS.txt
)

echo "[7/7] Creating the complete modified ZIP..."
OUTPUT_ZIP="$OUTPUT_DIR/browser_agent_stage1_easy_${VERSION}.zip"
OUTPUT_SHA="$OUTPUT_ZIP.sha256"
rm -f "$OUTPUT_ZIP" "$OUTPUT_SHA"

python3 - "$STAGE" "$OUTPUT_ZIP" <<'PY'
from pathlib import Path
import sys
import zipfile

stage = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
archive_root = stage.name
with zipfile.ZipFile(
    output,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as archive:
    for path in sorted(stage.rglob("*")):
        if path.is_file():
            archive.write(path, f"{archive_root}/{path.relative_to(stage).as_posix()}")
PY

sha256sum "$OUTPUT_ZIP" > "$OUTPUT_SHA"

echo
echo "[PASS] Complete modified package created:"
echo "  $OUTPUT_ZIP"
echo "  $OUTPUT_SHA"
echo
echo "Server usage:"
echo "  unzip $(basename "$OUTPUT_ZIP")"
echo "  cd browser_agent_stage1_easy_${VERSION}"
echo "  ./start.sh setup"
echo "  ./start.sh check"
echo "  ./start.sh probe"
echo "  ./start.sh train"
