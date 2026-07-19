#!/usr/bin/env python3
"""Build-host self-test for the Browser Agent easy overlay."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "scripts" / "easy_launcher_bootstrap.py"
SOURCE = ROOT / "scripts" / "easy_launcher.py"
SERVER_TOML = ROOT / "server.toml"


def load_bridge():
    spec = importlib.util.spec_from_file_location("easy_launcher_bootstrap", BRIDGE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load easy launcher bootstrap")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    assert BRIDGE.is_file(), BRIDGE
    assert SOURCE.is_file(), SOURCE
    assert SERVER_TOML.is_file(), SERVER_TOML

    bridge = load_bridge()
    repaired = bridge.repaired_source()
    compile(repaired, str(SOURCE), "exec")

    required_code_markers = (
        "Unknown server.toml sections",
        "ambiguous in base config.toml",
        "A checkpoint is required, but none was found",
        "ports.ports_per_gpu must be >= hardware.environments_per_gpu",
        "Chromium local HTTP probe",
        "tasks.mode must be http",
        "security.allow_external_network must remain false",
    )
    missing_code = [marker for marker in required_code_markers if marker not in repaired]
    if missing_code:
        raise RuntimeError(f"Launcher lost fail-closed markers: {missing_code}")

    toml_text = SERVER_TOML.read_text(encoding="utf-8")
    required_sections = (
        "[project]", "[environment]", "[hardware]", "[ports]",
        "[training]", "[browser]", "[tasks]", "[curriculum]",
        "[security]", "[check]", "[bindings]",
    )
    missing_sections = [section for section in required_sections if section not in toml_text]
    if missing_sections:
        raise RuntimeError(f"server.toml missing sections: {missing_sections}")

    print("[PASS] easy overlay repaired launcher compiles")
    print("[PASS] fail-closed markers present")
    print("[PASS] server.toml section structure present")


if __name__ == "__main__":
    main()
