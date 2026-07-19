#!/usr/bin/env bash
set -Eeuo pipefail

OVERLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_ZIP="${1:-}"
OUTPUT_DIR="${2:-$PWD}"
VERSION="v0.1.6-easy8a100"

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
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required"

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

echo "[1/9] Extracting v0.1.4..."
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

grep -q -- '--updates' "$BASE_ROOT/run_train_8xa100.sh" \
  || fail "Base run_train_8xa100.sh does not support --updates"
grep -q -- '--resume' "$BASE_ROOT/run_train_8xa100.sh" \
  || fail "Base run_train_8xa100.sh does not support --resume"

echo "[2/9] Copying the complete base project..."
cp -a "$BASE_ROOT"/. "$STAGE"/
rm -rf \
  "$STAGE/.venv" \
  "$STAGE/.easy_bootstrap" \
  "$STAGE/runtime" \
  "$STAGE/runs" \
  "$STAGE/.pytest_cache"
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$STAGE" -type f \( -name '*.pyc' -o -name '*.pyo' -o -name '*.log' \) -delete

echo "[3/9] Installing the unified TOML launcher..."
cp "$OVERLAY_DIR/server.toml" "$STAGE/server.toml"
cp "$OVERLAY_DIR/start.sh" "$STAGE/start.sh"
cp "$OVERLAY_DIR/README_EASY_CN.md" "$STAGE/README_EASY_CN.md"
mkdir -p "$STAGE/scripts"
cp "$OVERLAY_DIR/scripts/easy_launcher.py" "$STAGE/scripts/easy_launcher.py"
cp "$OVERLAY_DIR/scripts/easy_launcher_bootstrap.py" "$STAGE/scripts/easy_launcher_bootstrap.py"

# Replace the compatibility writer as text before compiling. The bootstrap
# bridge applies the same replacement in memory when running the overlay.
python3 - "$STAGE/scripts/easy_launcher.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
start_marker = "def write_compatibility_shims(site: Path) -> None:\n"
end_marker = "\ndef create_project_environment(\n"
start = text.find(start_marker)
end = text.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("Unable to locate compatibility-shim function")
replacement = '''def write_compatibility_shims(site: Path) -> None:
    content = """\\
\"\"\"Python 3.10 compatibility for the Stage-1 runtime.\"\"\"
import datetime
import enum
import typing

try:
    import typing_extensions
except ImportError:
    typing_extensions = None

if typing_extensions is not None:
    for name in (
        \"Self\", \"LiteralString\", \"Never\", \"NotRequired\",
        \"Required\", \"TypeVarTuple\", \"Unpack\", \"override\",
    ):
        if not hasattr(typing, name) and hasattr(typing_extensions, name):
            setattr(typing, name, getattr(typing_extensions, name))

if not hasattr(datetime, \"UTC\"):
    datetime.UTC = datetime.timezone.utc

if not hasattr(enum, \"StrEnum\"):
    class StrEnum(str, enum.Enum):
        def __str__(self):
            return str(self.value)
    enum.StrEnum = StrEnum
"""
    (site / "sitecustomize.py").write_text(content, encoding="utf-8")

'''
path.write_text(text[:start] + replacement + text[end + 1 :], encoding="utf-8")
PY

cat > "$STAGE/tomllib.py" <<'PY'
"""Python 3.10 compatibility wrapper for code that imports tomllib."""
from tomli import TOMLDecodeError, load, loads
__all__ = ["TOMLDecodeError", "load", "loads"]
PY

printf '%s\n' "$VERSION" > "$STAGE/EASY_RELEASE_VERSION.txt"
chmod +x \
  "$STAGE/start.sh" \
  "$STAGE/scripts/easy_launcher.py" \
  "$STAGE/scripts/easy_launcher_bootstrap.py" \
  "$STAGE/run_train_8xa100.sh" \
  "$STAGE/run_smoke_linux.sh"

echo "[4/9] Adapting project metadata for Python 3.10 and Torch 2.11..."
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
text = re.sub(
    r'"torch[^"\n]*"',
    '"torch>=2.11,<2.12"',
    text,
    count=1,
)
path.write_text(text, encoding="utf-8")
PY

echo "[5/9] Running build-time syntax checks..."
bash -n \
  "$STAGE/start.sh" \
  "$STAGE/run_train_8xa100.sh" \
  "$STAGE/run_smoke_linux.sh"
python3 -m py_compile \
  "$STAGE/scripts/easy_launcher.py" \
  "$STAGE/scripts/easy_launcher_bootstrap.py" \
  "$STAGE/tomllib.py"

echo "[6/9] Checking unified configuration structure..."
python3 - "$STAGE/server.toml" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
required = (
    "[project]", "[environment]", "[hardware]", "[ports]",
    "[training]", "[browser]", "[tasks]", "[curriculum]",
    "[security]", "[check]", "[bindings]",
)
missing = [section for section in required if section not in text]
if missing:
    raise SystemExit(f"server.toml missing sections: {missing}")
PY

echo "[7/9] Writing release validation instructions..."
cat > "$STAGE/SERVER_VALIDATION_REQUIRED.txt" <<'TXT'
This package passed build-time Shell and Python syntax checks.
The following target-server checks remain mandatory:

  ./start.sh setup
  ./start.sh check
  ./start.sh probe

Formal training must not start unless all three commands pass.
TXT

echo "[8/9] Generating content manifest..."
rm -f "$STAGE/PACKAGE_CONTENT_SHA256SUMS.txt"
(
  cd "$STAGE"
  find . -type f ! -name PACKAGE_CONTENT_SHA256SUMS.txt -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > PACKAGE_CONTENT_SHA256SUMS.txt
)

echo "[9/9] Creating the complete modified ZIP..."
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
