#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

LAUNCHER = Path("scripts/easy_launcher.py")
BUILDER = Path("apply_easy_overlay.sh")


def replace_between(text: str, start_marker: str, end_marker: str, replacement: str) -> str:
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise SystemExit(f"Unable to locate patch markers: {start_marker!r} / {end_marker!r}")
    return text[:start] + replacement + text[end + 1 :]


def patch_launcher() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    shim = '''def write_compatibility_shims(site: Path) -> None:
    content = (
        "# Python 3.10 compatibility for the Stage-1 runtime.\\n"
        "import datetime\\n"
        "import enum\\n"
        "import typing\\n"
        "try:\\n"
        "    import typing_extensions\\n"
        "except ImportError:\\n"
        "    typing_extensions = None\\n"
        "if typing_extensions is not None:\\n"
        "    for name in (\\\"Self\\\", \\\"LiteralString\\\", \\\"Never\\\", \\\"NotRequired\\\", \\\"Required\\\", \\\"TypeVarTuple\\\", \\\"Unpack\\\", \\\"override\\\"):\\n"
        "        if not hasattr(typing, name) and hasattr(typing_extensions, name):\\n"
        "            setattr(typing, name, getattr(typing_extensions, name))\\n"
        "if not hasattr(datetime, \\\"UTC\\\"):\\n"
        "    datetime.UTC = datetime.timezone.utc\\n"
        "if not hasattr(enum, \\\"StrEnum\\\"):\\n"
        "    class StrEnum(str, enum.Enum):\\n"
        "        def __str__(self):\\n"
        "            return str(self.value)\\n"
        "    enum.StrEnum = StrEnum\\n"
    )
    (site / "sitecustomize.py").write_text(content, encoding="utf-8")

'''
    text = replace_between(
        text,
        "def write_compatibility_shims(site: Path) -> None:\n",
        "\ndef create_project_environment(\n",
        shim,
    )

    if '    "tasks.mode": (' not in text:
        anchor = '    "tasks.train_template_variants": (\n'
        insertion = '''    "tasks.mode": (
        "tasks.mode", "task.mode", "environment.task_mode",
        "task_mode",
    ),
'''
        if anchor not in text:
            raise SystemExit("Unable to locate TARGET_PATTERNS task anchor")
        text = text.replace(anchor, insertion + anchor, 1)

    if "def validate_selected_gpu_ids(" not in text:
        anchor = "\ndef port_is_free(host: str, port: int) -> bool:\n"
        insertion = '''
def validate_selected_gpu_ids(
    config: dict[str, Any],
    torch_info: dict[str, Any],
) -> None:
    available = int(torch_info["gpu_count"])
    selected = [int(item) for item in config["hardware"]["gpu_ids"]]
    invalid = [gpu_id for gpu_id in selected if gpu_id >= available]
    if invalid:
        fail(
            f"Configured GPU IDs do not exist: {invalid}; "
            f"available IDs are 0..{available - 1}"
        )

'''
        if anchor not in text:
            raise SystemExit("Unable to locate GPU validation insertion point")
        text = text.replace(anchor, insertion + anchor, 1)

    old = '''    torch_info = inspect_project_torch(config)
    ports = resolve_ports(config)
'''
    new = '''    torch_info = inspect_project_torch(config)
    validate_selected_gpu_ids(config, torch_info)
    ports = resolve_ports(config)
'''
    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit("Unable to locate common_resolution GPU validation call")

    LAUNCHER.write_text(text, encoding="utf-8")


def patch_builder() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    start_marker = "# Replace the compatibility writer as text before compiling."
    end_marker = "cat > \"$STAGE/tomllib.py\" <<'PY'\n"
    start = text.find(start_marker)
    if start >= 0:
        end = text.find(end_marker, start)
        if end < 0:
            raise SystemExit("Unable to locate builder source-rewrite block end")
        text = text[:start] + end_marker + text[end + len(end_marker) :]
    if "Replace the compatibility writer" in text:
        raise SystemExit("Builder still contains source-rewrite logic")
    BUILDER.write_text(text, encoding="utf-8")


def main() -> None:
    patch_launcher()
    patch_builder()


if __name__ == "__main__":
    main()
