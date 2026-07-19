#!/usr/bin/env python3
"""Compile and execute the fail-closed launcher after repairing its shim writer.

The final package builder also performs this replacement on disk and then runs
py_compile. This bridge keeps the source overlay directly runnable before the
final package is built.
"""
from __future__ import annotations

from pathlib import Path

SOURCE = Path(__file__).with_name("easy_launcher.py")


def repaired_source() -> str:
    text = SOURCE.read_text(encoding="utf-8")
    start_marker = "def write_compatibility_shims(site: Path) -> None:\n"
    end_marker = "\ndef create_project_environment(\n"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("Unable to locate compatibility-shim function")

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
    return text[:start] + replacement + text[end + 1 :]


def main() -> None:
    text = repaired_source()
    code = compile(text, str(SOURCE), "exec")
    namespace = {
        "__name__": "__main__",
        "__file__": str(SOURCE),
        "__package__": None,
        "__cached__": None,
    }
    exec(code, namespace, namespace)


if __name__ == "__main__":
    main()
