#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, NoReturn

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SERVER_TOML = ROOT / "server.toml"

# server.toml中允许出现的字段。未知字段直接失败，避免“写了但没有生效”。
SCHEMA: dict[str, set[str]] = {
    "project": {
        "name", "base_config", "auto_create_environment",
        "auto_repair_environment", "reuse_system_torch", "venv_dir",
        "bootstrap_venv_dir", "runtime_dir", "run_dir",
        "checkpoint_dir", "log_dir",
    },
    "environment": {
        "python", "expected_python", "expected_torch_prefix",
        "expected_cuda", "required_gpu_count", "pip_index",
        "chromium_mode", "chromium_executable",
        "install_playwright_browser",
    },
    "hardware": {
        "gpu_ids", "distributed_backend", "environments_per_gpu",
        "mixed_precision", "allow_tf32",
    },
    "ports": {
        "host", "base_port", "ports_per_gpu", "auto_find_free_block",
        "search_until_port", "master_port", "auto_find_master_port",
    },
    "training": {
        "updates", "probe_updates", "seed", "rollout_steps",
        "ppo_epochs", "minibatch_size", "learning_rate", "gamma",
        "gae_lambda", "clip_ratio", "value_clip_ratio", "entropy_coef",
        "value_coef", "text_aux_coef", "max_grad_norm", "target_kl",
        "checkpoint_every_updates", "evaluation_every_updates",
        "log_every_updates", "resume", "checkpoint_path",
    },
    "browser": {
        "headless", "page_width", "page_height", "navigation_timeout_ms",
        "action_timeout_ms", "episode_timeout_seconds",
        "disable_service_workers", "allow_downloads",
        "restart_failed_environment", "max_environment_restarts",
    },
    "tasks": {
        "mode", "train_template_variants", "evaluation_template_variants",
        "max_episode_steps", "max_elements", "max_arguments",
        "max_generated_text_tokens", "enabled",
    },
    "curriculum": {
        "enabled", "minimum_episodes_per_task", "success_rate_threshold",
        "replay_previous_tasks", "previous_task_probability",
    },
    "security": {
        "allow_external_network", "allowed_hosts",
        "strict_origin_whitelist",
    },
    "check": {
        "run_python_syntax", "run_shell_syntax", "run_unit_tests",
        "run_full_browser_smoke",
    },
    "bindings": set(),
}

# server.toml源字段 -> 基础config.toml中允许的精确/后缀路径。
# 优先精确全路径；只有唯一匹配才会应用。
TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "training.seed": ("training.seed", "ppo.seed", "seed"),
    "training.rollout_steps": (
        "training.rollout_steps", "ppo.rollout_steps", "rollout_steps",
    ),
    "training.ppo_epochs": (
        "training.ppo_epochs", "ppo.ppo_epochs", "ppo.update_epochs",
        "ppo_epochs", "update_epochs",
    ),
    "training.minibatch_size": (
        "training.minibatch_size", "ppo.minibatch_size",
        "ppo.mini_batch_size", "minibatch_size", "mini_batch_size",
    ),
    "training.learning_rate": (
        "training.learning_rate", "ppo.learning_rate",
        "ppo.policy_learning_rate", "learning_rate", "policy_learning_rate",
    ),
    "training.gamma": ("training.gamma", "ppo.gamma", "gamma"),
    "training.gae_lambda": (
        "training.gae_lambda", "ppo.gae_lambda", "gae_lambda",
    ),
    "training.clip_ratio": (
        "training.clip_ratio", "ppo.clip_ratio", "ppo.clip_coef",
        "clip_ratio", "clip_coef",
    ),
    "training.value_clip_ratio": (
        "training.value_clip_ratio", "ppo.value_clip_ratio",
        "ppo.value_clip_coef", "value_clip_ratio", "value_clip_coef",
    ),
    "training.entropy_coef": (
        "training.entropy_coef", "ppo.entropy_coef", "entropy_coef",
    ),
    "training.value_coef": (
        "training.value_coef", "ppo.value_coef", "value_coef",
    ),
    "training.text_aux_coef": (
        "training.text_aux_coef", "ppo.text_aux_coef", "text_aux_coef",
    ),
    "training.max_grad_norm": (
        "training.max_grad_norm", "ppo.max_grad_norm", "max_grad_norm",
    ),
    "training.target_kl": (
        "training.target_kl", "ppo.target_kl", "target_kl",
    ),
    "training.checkpoint_every_updates": (
        "training.checkpoint_every_updates", "checkpoint.every_updates",
        "checkpoint.checkpoint_every_updates", "checkpoint_interval",
        "checkpoint_every_updates",
    ),
    "training.evaluation_every_updates": (
        "training.evaluation_every_updates", "evaluation.every_updates",
        "evaluation.eval_interval", "evaluation_every_updates",
        "eval_interval",
    ),
    "training.log_every_updates": (
        "training.log_every_updates", "logging.every_updates",
        "logging.log_interval", "log_every_updates", "log_interval",
    ),
    "hardware.environments_per_gpu": (
        "hardware.environments_per_gpu", "environment.envs_per_rank",
        "training.envs_per_rank", "environments_per_gpu", "envs_per_rank",
        "num_envs",
    ),
    "hardware.mixed_precision": (
        "hardware.mixed_precision", "training.mixed_precision",
        "training.amp", "mixed_precision", "amp", "use_amp",
    ),
    "ports.base_port": (
        "browser.base_port", "server.base_port", "tasks.base_port",
        "base_port", "server_base_port", "task_server_port",
    ),
    "browser.headless": ("browser.headless", "headless"),
    "browser.page_width": (
        "browser.page_width", "browser.viewport_width", "page_width",
        "viewport_width",
    ),
    "browser.page_height": (
        "browser.page_height", "browser.viewport_height", "page_height",
        "viewport_height",
    ),
    "browser.navigation_timeout_ms": (
        "browser.navigation_timeout_ms", "navigation_timeout_ms",
    ),
    "browser.action_timeout_ms": (
        "browser.action_timeout_ms", "action_timeout_ms",
    ),
    "browser.episode_timeout_seconds": (
        "browser.episode_timeout_seconds", "browser.episode_timeout_s",
        "episode_timeout_seconds", "episode_timeout_s",
    ),
    "browser.disable_service_workers": (
        "browser.disable_service_workers", "security.disable_service_workers",
        "disable_service_workers",
    ),
    "browser.allow_downloads": (
        "browser.allow_downloads", "security.allow_downloads",
        "allow_downloads",
    ),
    "browser.restart_failed_environment": (
        "browser.restart_failed_environment", "environment.restart_failed_environment",
        "restart_failed_environment",
    ),
    "browser.max_environment_restarts": (
        "browser.max_environment_restarts", "environment.max_environment_restarts",
        "max_environment_restarts",
    ),
    "tasks.train_template_variants": (
        "tasks.train_template_variants", "synthetic.train_template_variants",
        "train_template_variants",
    ),
    "tasks.evaluation_template_variants": (
        "tasks.evaluation_template_variants", "tasks.eval_template_variants",
        "synthetic.eval_template_variants", "evaluation_template_variants",
        "eval_template_variants",
    ),
    "tasks.max_episode_steps": (
        "tasks.max_episode_steps", "environment.max_episode_steps",
        "max_episode_steps", "max_steps",
    ),
    "tasks.max_elements": (
        "tasks.max_elements", "observation.max_elements", "max_elements",
    ),
    "tasks.max_arguments": (
        "tasks.max_arguments", "observation.max_args", "max_arguments",
        "max_args",
    ),
    "tasks.max_generated_text_tokens": (
        "tasks.max_generated_text_tokens", "model.max_generated_text_tokens",
        "max_generated_text_tokens",
    ),
    "tasks.enabled": (
        "tasks.task_kinds", "tasks.enabled_tasks", "task_kinds",
        "enabled_tasks",
    ),
    "curriculum.enabled": (
        "curriculum.enabled", "training.curriculum_enabled",
        "curriculum_enabled",
    ),
    "curriculum.minimum_episodes_per_task": (
        "curriculum.minimum_episodes_per_task", "curriculum.min_episodes_per_task",
        "minimum_episodes_per_task", "min_episodes_per_task",
    ),
    "curriculum.success_rate_threshold": (
        "curriculum.success_rate_threshold", "success_rate_threshold",
    ),
    "curriculum.previous_task_probability": (
        "curriculum.previous_task_probability", "previous_task_probability",
    ),
    "security.allow_external_network": (
        "security.allow_external_network", "browser.allow_external_network",
        "allow_external_network",
    ),
    "security.allowed_hosts": (
        "security.allowed_hosts", "browser.allowed_hosts", "allowed_hosts",
    ),
    "security.strict_origin_whitelist": (
        "security.strict_origin_whitelist", "browser.strict_origin_whitelist",
        "strict_origin_whitelist",
    ),
}


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    if not isinstance(value, dict):
        fail(f"Invalid TOML root: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(command), flush=True)
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=check,
        text=True,
    )


def capture(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def server_config() -> dict[str, Any]:
    if not SERVER_TOML.is_file():
        fail(f"Missing unified configuration: {SERVER_TOML}")
    value = load_toml(SERVER_TOML)
    validate_server_config(value)
    return value


def project_path(config: dict[str, Any], key: str, default: str) -> Path:
    raw = str(config.get("project", {}).get(key, default))
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_path(data: dict[str, Any], dotted: str) -> Any:
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            fail(f"Missing configuration path: {dotted}")
        cursor = cursor[part]
    return cursor


def exact_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        fail(f"{name} must be true or false")
    return value


def positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int:
        fail(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        fail(f"{name} must be >= {minimum}")
    return value


def numeric(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        fail(f"{name} must be numeric")
    return float(value)


def validate_server_config(config: dict[str, Any]) -> None:
    unknown_sections = sorted(set(config) - set(SCHEMA))
    missing_sections = sorted(set(SCHEMA) - set(config))
    if unknown_sections:
        fail(f"Unknown server.toml sections: {unknown_sections}")
    if missing_sections:
        fail(f"Missing server.toml sections: {missing_sections}")

    for section, allowed in SCHEMA.items():
        value = config[section]
        if not isinstance(value, dict):
            fail(f"[{section}] must be a TOML table")
        if section == "bindings":
            continue
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown:
            fail(f"Unknown keys in [{section}]: {unknown}")
        if missing:
            fail(f"Missing keys in [{section}]: {missing}")

    bindings = config["bindings"]
    for source, target in bindings.items():
        if source not in TARGET_PATTERNS:
            fail(f"Unknown binding source: {source}")
        if not isinstance(target, str) or not target.strip():
            fail(f"Binding target for {source} must be a non-empty dotted path")

    project = config["project"]
    if not exact_bool(project["reuse_system_torch"], "project.reuse_system_torch"):
        fail("This release requires reuse_system_torch=true")
    exact_bool(project["auto_create_environment"], "project.auto_create_environment")
    exact_bool(project["auto_repair_environment"], "project.auto_repair_environment")

    environment = config["environment"]
    if environment["chromium_mode"] not in {"auto", "system", "playwright"}:
        fail("environment.chromium_mode must be auto, system or playwright")
    positive_int(environment["required_gpu_count"], "environment.required_gpu_count")
    exact_bool(
        environment["install_playwright_browser"],
        "environment.install_playwright_browser",
    )

    hardware = config["hardware"]
    if hardware["distributed_backend"] != "nccl":
        fail("hardware.distributed_backend must be nccl")
    gpu_ids = hardware["gpu_ids"]
    if not isinstance(gpu_ids, list) or not gpu_ids:
        fail("hardware.gpu_ids must be a non-empty integer array")
    if any(type(item) is not int or item < 0 for item in gpu_ids):
        fail("hardware.gpu_ids must contain non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        fail("hardware.gpu_ids contains duplicates")
    required_gpus = int(environment["required_gpu_count"])
    if len(gpu_ids) != required_gpus:
        fail(
            "hardware.gpu_ids length must equal environment.required_gpu_count "
            f"({len(gpu_ids)} != {required_gpus})"
        )
    envs_per_gpu = positive_int(
        hardware["environments_per_gpu"],
        "hardware.environments_per_gpu",
    )
    exact_bool(hardware["mixed_precision"], "hardware.mixed_precision")
    exact_bool(hardware["allow_tf32"], "hardware.allow_tf32")

    ports = config["ports"]
    if ports["host"] not in {"127.0.0.1", "localhost"}:
        fail("ports.host must be 127.0.0.1 or localhost")
    base_port = positive_int(ports["base_port"], "ports.base_port")
    search_limit = positive_int(ports["search_until_port"], "ports.search_until_port")
    master_port = positive_int(ports["master_port"], "ports.master_port")
    ports_per_gpu = positive_int(ports["ports_per_gpu"], "ports.ports_per_gpu")
    if not (1 <= base_port <= 65535 and 1 <= search_limit <= 65535):
        fail("Browser ports must be in 1..65535")
    if not (1 <= master_port <= 65535):
        fail("ports.master_port must be in 1..65535")
    if search_limit < base_port:
        fail("ports.search_until_port must be >= ports.base_port")
    if ports_per_gpu < envs_per_gpu:
        fail(
            "ports.ports_per_gpu must be >= hardware.environments_per_gpu "
            "to prevent under-allocation"
        )
    exact_bool(ports["auto_find_free_block"], "ports.auto_find_free_block")
    exact_bool(ports["auto_find_master_port"], "ports.auto_find_master_port")

    training = config["training"]
    for key in (
        "updates", "probe_updates", "rollout_steps", "ppo_epochs",
        "minibatch_size", "checkpoint_every_updates",
        "evaluation_every_updates", "log_every_updates",
    ):
        positive_int(training[key], f"training.{key}")
    if type(training["seed"]) is not int:
        fail("training.seed must be an integer")
    for key in (
        "learning_rate", "gamma", "gae_lambda", "clip_ratio",
        "value_clip_ratio", "entropy_coef", "value_coef", "text_aux_coef",
        "max_grad_norm", "target_kl",
    ):
        numeric(training[key], f"training.{key}")
    if float(training["learning_rate"]) <= 0:
        fail("training.learning_rate must be positive")
    if not 0 < float(training["gamma"]) <= 1:
        fail("training.gamma must be in (0, 1]")
    if not 0 < float(training["gae_lambda"]) <= 1:
        fail("training.gae_lambda must be in (0, 1]")
    if training["resume"] not in {"auto", "never", "required"}:
        fail("training.resume must be auto, never or required")
    if not isinstance(training["checkpoint_path"], str):
        fail("training.checkpoint_path must be a string")

    total_rollout = int(training["rollout_steps"]) * envs_per_gpu * len(gpu_ids)
    minibatch = int(training["minibatch_size"])
    if minibatch > total_rollout:
        fail(
            f"training.minibatch_size ({minibatch}) exceeds global rollout "
            f"batch ({total_rollout})"
        )
    if total_rollout % minibatch != 0:
        fail(
            f"Global rollout batch {total_rollout} must be divisible by "
            f"training.minibatch_size {minibatch}"
        )

    browser = config["browser"]
    for key in (
        "page_width", "page_height", "navigation_timeout_ms",
        "action_timeout_ms", "episode_timeout_seconds",
        "max_environment_restarts",
    ):
        positive_int(browser[key], f"browser.{key}")
    for key in (
        "headless", "disable_service_workers", "allow_downloads",
        "restart_failed_environment",
    ):
        exact_bool(browser[key], f"browser.{key}")
    if browser["allow_downloads"]:
        fail("browser.allow_downloads must remain false in Stage 1")

    tasks = config["tasks"]
    if tasks["mode"] != "http":
        fail("tasks.mode must be http; inline mode is probe-only and not a full trainer")
    for key in (
        "train_template_variants", "evaluation_template_variants",
        "max_episode_steps", "max_elements", "max_arguments",
        "max_generated_text_tokens",
    ):
        positive_int(tasks[key], f"tasks.{key}")
    if not isinstance(tasks["enabled"], list) or any(
        not isinstance(item, str) or not item.strip() for item in tasks["enabled"]
    ):
        fail("tasks.enabled must be an array of non-empty strings")

    curriculum = config["curriculum"]
    exact_bool(curriculum["enabled"], "curriculum.enabled")
    exact_bool(
        curriculum["replay_previous_tasks"],
        "curriculum.replay_previous_tasks",
    )
    positive_int(
        curriculum["minimum_episodes_per_task"],
        "curriculum.minimum_episodes_per_task",
    )
    threshold = numeric(
        curriculum["success_rate_threshold"],
        "curriculum.success_rate_threshold",
    )
    previous_probability = numeric(
        curriculum["previous_task_probability"],
        "curriculum.previous_task_probability",
    )
    if not 0 <= threshold <= 1:
        fail("curriculum.success_rate_threshold must be in [0, 1]")
    if not 0 <= previous_probability <= 1:
        fail("curriculum.previous_task_probability must be in [0, 1]")

    security = config["security"]
    if security["allow_external_network"]:
        fail("security.allow_external_network must remain false")
    if not security["strict_origin_whitelist"]:
        fail("security.strict_origin_whitelist must remain true")
    if set(security["allowed_hosts"]) != {"127.0.0.1", "localhost"}:
        fail("security.allowed_hosts must be exactly 127.0.0.1 and localhost")

    for key, value in config["check"].items():
        exact_bool(value, f"check.{key}")


def system_python(config: dict[str, Any]) -> str:
    return str(config["environment"].get("python", "python3"))


def inspect_system_torch(config: dict[str, Any]) -> dict[str, Any]:
    python = system_python(config)
    code = r'''
import json
import sys
from pathlib import Path
import torch
print(json.dumps({
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "python_full": sys.version,
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": torch.cuda.device_count(),
    "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch_site": str(Path(torch.__file__).resolve().parents[1]),
    "nccl_available": bool(torch.distributed.is_available() and torch.distributed.is_nccl_available()),
}))
'''
    raw = capture([python, "-c", code])
    value = json.loads(raw)

    environment = config["environment"]
    errors: list[str] = []
    if value["python"] != str(environment["expected_python"]):
        errors.append(
            f"Python expected {environment['expected_python']}, got {value['python']}"
        )
    if not str(value["torch"]).startswith(str(environment["expected_torch_prefix"])):
        errors.append(
            "PyTorch expected prefix "
            f"{environment['expected_torch_prefix']}, got {value['torch']}"
        )
    if str(value["torch_cuda"]) != str(environment["expected_cuda"]):
        errors.append(
            f"CUDA expected {environment['expected_cuda']}, got {value['torch_cuda']}"
        )
    if not value["cuda_available"]:
        errors.append("torch.cuda.is_available() is false")
    if int(value["gpu_count"]) < int(environment["required_gpu_count"]):
        errors.append(
            f"Expected at least {environment['required_gpu_count']} GPUs, "
            f"got {value['gpu_count']}"
        )
    if not value["nccl_available"]:
        errors.append("NCCL backend is unavailable")
    if errors:
        fail("Server environment mismatch:\n- " + "\n- ".join(errors))
    return value


def venv_python(config: dict[str, Any]) -> Path:
    return project_path(config, "venv_dir", ".venv") / "bin" / "python"


def current_is_venv(config: dict[str, Any]) -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python(config).resolve()
    except FileNotFoundError:
        return False


def project_environment_health(
    config: dict[str, Any],
    torch_info: dict[str, Any],
) -> tuple[bool, str]:
    python = venv_python(config)
    if not python.is_file():
        return False, "project Python is missing"
    code = r'''
import json
import sys
from pathlib import Path
try:
    import browser_agent
    import playwright
    import tomli
    import tomli_w
    import torch
    value = {
        "ok": True,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_site": str(Path(torch.__file__).resolve().parents[1]),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "nccl_available": bool(torch.distributed.is_available() and torch.distributed.is_nccl_available()),
    }
except Exception as exc:
    value = {"ok": False, "error": repr(exc)}
print(json.dumps(value))
'''
    try:
        raw = capture([str(python), "-c", code])
        value = json.loads(raw)
    except Exception as exc:
        return False, f"health probe failed: {exc}"
    if not value.get("ok"):
        return False, str(value.get("error", "unknown import failure"))
    expected_python = str(config["environment"]["expected_python"])
    if value["python"] != expected_python:
        return False, f"Python {value['python']} != {expected_python}"
    if value["torch"] != torch_info["torch"]:
        return False, f"Torch {value['torch']} != {torch_info['torch']}"
    if str(value["torch_cuda"]) != str(torch_info["torch_cuda"]):
        return False, "CUDA runtime differs from verified system Torch"
    if not value["cuda_available"] or not value["nccl_available"]:
        return False, "CUDA or NCCL unavailable in project environment"
    if int(value["gpu_count"]) < int(config["environment"]["required_gpu_count"]):
        return False, "project environment cannot see all required GPUs"
    return True, "healthy"


def pip_install(
    python: Path,
    packages: list[str],
    config: dict[str, Any],
    *extra: str,
) -> None:
    index = str(config["environment"].get("pip_index", "")).strip()
    base = [str(python), "-m", "pip", "install", *extra, *packages]
    if index:
        result = run([*base, "-i", index], check=False)
        if result.returncode == 0:
            return
        print("[WARN] Configured pip mirror failed; retrying the default index")
    run(base)


def write_compatibility_shims(site: Path) -> None:
    (site / "sitecustomize.py").write_text(
        '''"""Python 3.10 compatibility for the Stage-1 runtime."""\n'
        'import datetime\n'
        'import enum\n'
        'import typing\n'
        'try:\n'
        '    import typing_extensions\n'
        'except ImportError:\n'
        '    typing_extensions = None\n'
        'if typing_extensions is not None:\n'
        '    for _name in ("Self", "LiteralString", "Never", "NotRequired", "Required", "TypeVarTuple", "Unpack", "override"):\n'
        '        if not hasattr(typing, _name) and hasattr(typing_extensions, _name):\n'
        '            setattr(typing, _name, getattr(typing_extensions, _name))\n'
        'if not hasattr(datetime, "UTC"):\n'
        '    datetime.UTC = datetime.timezone.utc\n'
        'if not hasattr(enum, "StrEnum"):\n'
        '    class StrEnum(str, enum.Enum):\n'
        '        def __str__(self):\n'
        '            return str(self.value)\n'
        '    enum.StrEnum = StrEnum\n',
        encoding="utf-8",
    )


def create_project_environment(
    config: dict[str, Any],
    torch_info: dict[str, Any],
) -> None:
    venv_dir = project_path(config, "venv_dir", ".venv")
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    run([system_python(config), "-m", "venv", str(venv_dir)])
    python = venv_python(config)
    site = Path(capture([
        str(python), "-c", "import site; print(site.getsitepackages()[0])",
    ]))
    (site / "00_dolphin_cuda_torch.pth").write_text(
        str(torch_info["torch_site"]) + "\n",
        encoding="utf-8",
    )
    write_compatibility_shims(site)

    pip_install(
        python,
        ["pip", "setuptools", "wheel"],
        config,
        "--upgrade",
    )
    pip_install(
        python,
        [
            "playwright>=1.60,<2",
            "pytest>=8,<10",
            "tomli>=2.2,<3",
            "tomli-w>=1.2,<2",
            "typing-extensions>=4.12,<5",
            "psutil>=5.9,<8",
            "nvidia-ml-py>=12.560",
        ],
        config,
    )
    run([
        str(python), "-m", "pip", "install", "--editable", str(ROOT),
        "--no-deps", "--no-build-isolation",
    ])

    healthy, reason = project_environment_health(config, torch_info)
    if not healthy:
        fail(f"New project environment failed validation: {reason}")

    runtime_dir = project_path(config, "runtime_dir", "runtime")
    write_json(runtime_dir / "environment_manifest.json", {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "server_toml_sha256": sha256(SERVER_TOML),
        "python": str(python),
        "torch": torch_info,
        "health": reason,
    })


def ensure_environment(config: dict[str, Any], argv: list[str]) -> None:
    torch_info = inspect_system_torch(config)
    healthy, reason = project_environment_health(config, torch_info)
    if not healthy:
        if not bool(config["project"]["auto_create_environment"]):
            fail(f"Project environment is unavailable: {reason}")
        if project_path(config, "venv_dir", ".venv").exists() and not bool(
            config["project"]["auto_repair_environment"]
        ):
            fail(f"Project environment is unhealthy and auto repair is disabled: {reason}")
        print(f"[SETUP] Rebuilding project environment: {reason}")
        create_project_environment(config, torch_info)

    if not current_is_venv(config):
        env = os.environ.copy()
        env["BROWSER_EASY_REEXEC"] = "1"
        os.execve(
            str(venv_python(config)),
            [str(venv_python(config)), str(Path(__file__).resolve()), *argv],
            env,
        )


def inspect_project_torch(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    value = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_full": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "nccl_available": bool(
            torch.distributed.is_available()
            and torch.distributed.is_nccl_available()
        ),
    }
    expected = inspect_system_torch(config)
    if value["torch"] != expected["torch"]:
        fail(f"Project venv replaced PyTorch: {value['torch']} != {expected['torch']}")
    if str(value["torch_cuda"]) != str(expected["torch_cuda"]):
        fail("Project venv CUDA runtime differs from verified system runtime")
    return value


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_block(
    host: str,
    start: int,
    count: int,
    limit: int,
    *,
    excluded: set[int] | None = None,
) -> int:
    if count <= 0:
        fail("Port block size must be positive")
    excluded = excluded or set()
    for candidate in range(start, limit - count + 2):
        block = {candidate + offset for offset in range(count)}
        if block & excluded:
            continue
        if all(port_is_free(host, port) for port in block):
            return candidate
    fail(f"No continuous block of {count} free ports found between {start} and {limit}")


@contextlib.contextmanager
def allocation_lock(config: dict[str, Any]) -> Iterator[None]:
    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / "port_allocation.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def resolve_ports(config: dict[str, Any]) -> dict[str, Any]:
    ports = config["ports"]
    hardware = config["hardware"]
    host = str(ports["host"])
    gpu_count = len(hardware["gpu_ids"])
    ports_per_gpu = int(ports["ports_per_gpu"])
    count = gpu_count * ports_per_gpu
    base = int(ports["base_port"])
    limit = int(ports["search_until_port"])
    master = int(ports["master_port"])

    if bool(ports["auto_find_free_block"]):
        base = find_free_block(host, base, count, limit, excluded={master})
    else:
        block = range(base, base + count)
        if master in block:
            fail("DDP master_port overlaps the browser task port block")
        if not all(port_is_free(host, port) for port in block):
            fail(f"Configured browser port block {base}-{base + count - 1} is occupied")

    browser_ports = set(range(base, base + count))
    if master in browser_ports or not port_is_free(host, master):
        if bool(ports["auto_find_master_port"]):
            master = find_free_block(
                host,
                max(int(ports["master_port"]), base + count),
                1,
                65535,
                excluded=browser_ports,
            )
        else:
            reason = "overlaps browser ports" if master in browser_ports else "is occupied"
            fail(f"Configured DDP master port {master} {reason}")

    return {
        "host": host,
        "base_port": base,
        "port_count": count,
        "last_port": base + count - 1,
        "master_port": master,
    }


def assert_ports_still_free(ports: dict[str, Any]) -> None:
    host = str(ports["host"])
    browser_ports = range(int(ports["base_port"]), int(ports["last_port"]) + 1)
    occupied = [port for port in browser_ports if not port_is_free(host, port)]
    if not port_is_free(host, int(ports["master_port"])):
        occupied.append(int(ports["master_port"]))
    if occupied:
        fail(f"Ports became occupied before launch: {sorted(set(occupied))}")


def discover_chromium(config: dict[str, Any]) -> str:
    environment = config["environment"]
    mode = str(environment["chromium_mode"]).lower()
    configured = str(environment["chromium_executable"]).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    env_candidate = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", "").strip()
    if env_candidate:
        candidates.append(Path(env_candidate).expanduser())
    candidates.extend(Path(path) for path in (
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ))
    if mode != "playwright":
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
    if mode == "system":
        fail("chromium_mode=system but no executable was found")
    if bool(environment["install_playwright_browser"]):
        run([sys.executable, "-m", "playwright", "install", "chromium"])
        return ""
    fail("Playwright Chromium is unavailable and automatic installation is disabled")


def flatten_paths(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    result: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            result.append(path)
            result.extend(flatten_paths(child, path))
    return result


def path_exists(data: dict[str, Any], dotted: str) -> bool:
    cursor: Any = data
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return False
        cursor = cursor[part]
    return True


def get_existing_path(data: dict[str, Any], dotted: str) -> Any:
    if not path_exists(data, dotted):
        fail(f"Base config target does not exist: {dotted}")
    cursor: Any = data
    for part in dotted.split("."):
        cursor = cursor[part]
    return cursor


def set_existing_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor: Any = data
    for part in parts[:-1]:
        if not isinstance(cursor, dict) or part not in cursor:
            fail(f"Base config target does not exist: {dotted}")
        cursor = cursor[part]
    if not isinstance(cursor, dict) or parts[-1] not in cursor:
        fail(f"Base config target does not exist: {dotted}")
    old = cursor[parts[-1]]
    numeric_pair = type(old) in {int, float} and type(value) in {int, float}
    if type(old) is not type(value) and not numeric_pair:
        fail(
            f"Type mismatch for {dotted}: base has {type(old).__name__}, "
            f"server.toml has {type(value).__name__}"
        )
    cursor[parts[-1]] = value


def resolve_target_path(
    effective: dict[str, Any],
    source: str,
    explicit_bindings: dict[str, Any],
) -> str:
    if source in explicit_bindings:
        target = str(explicit_bindings[source])
        if not path_exists(effective, target):
            fail(f"Explicit binding {source} -> {target} does not exist")
        return target

    all_paths = flatten_paths(effective)
    patterns = TARGET_PATTERNS[source]

    # First prefer an exact full-path match in declared order.
    for pattern in patterns:
        parts = tuple(pattern.split("."))
        exact = [path for path in all_paths if path == parts]
        if len(exact) == 1:
            return ".".join(exact[0])

    # Then permit one unique suffix match. Multiple matches are never guessed.
    matches: list[tuple[str, ...]] = []
    for pattern in patterns:
        parts = tuple(pattern.split("."))
        for path in all_paths:
            if len(path) >= len(parts) and path[-len(parts):] == parts:
                if path not in matches:
                    matches.append(path)
    if len(matches) == 1:
        return ".".join(matches[0])
    if not matches:
        fail(
            f"server.toml field {source} has no target in base config.toml. "
            "Add an exact [bindings] entry after confirming the base Schema."
        )
    fail(
        f"server.toml field {source} is ambiguous in base config.toml: "
        f"{['.'.join(path) for path in matches]}. Add an exact [bindings] entry."
    )


def source_value(config: dict[str, Any], source: str) -> Any:
    if source == "curriculum.previous_task_probability":
        if not bool(config["curriculum"]["replay_previous_tasks"]):
            return 0.0
    return get_path(config, source)


def prepare_effective_config(
    config: dict[str, Any],
    ports: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    try:
        import tomli_w
    except ImportError as exc:
        raise RuntimeError("tomli-w is required in the project environment") from exc

    base_config = project_path(config, "base_config", "config.toml")
    if not base_config.is_file():
        fail(f"Missing base training configuration: {base_config}")
    effective = load_toml(base_config)
    bindings = config.get("bindings", {})

    values: dict[str, Any] = {
        source: source_value(config, source)
        for source in TARGET_PATTERNS
        if source != "tasks.enabled" or bool(config["tasks"]["enabled"])
    }
    values["ports.base_port"] = int(ports["base_port"])

    report: dict[str, Any] = {
        "base_config": str(base_config),
        "base_config_sha256": sha256(base_config),
        "applied": {},
    }
    for source, value in values.items():
        target = resolve_target_path(effective, source, bindings)
        previous = get_existing_path(effective, target)
        set_existing_path(effective, target, value)
        report["applied"][source] = {
            "target": target,
            "previous": previous,
            "value": value,
        }

    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output = runtime_dir / "effective_config.toml"
    temporary = output.with_suffix(".toml.tmp")
    temporary.write_text(tomli_w.dumps(effective), encoding="utf-8")
    os.replace(temporary, output)
    # 写回后重新解析，确保生成的TOML有效。
    load_toml(output)
    report["effective_config"] = str(output)
    report["effective_config_sha256"] = sha256(output)
    write_json(runtime_dir / "binding_report.json", report)
    return output, report


def runtime_environment(
    config: dict[str, Any],
    ports: dict[str, Any],
    chromium: str,
) -> dict[str, str]:
    env = os.environ.copy()
    gpu_ids = [str(item) for item in config["hardware"]["gpu_ids"]]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["MASTER_ADDR"] = str(ports["host"])
    env["MASTER_PORT"] = str(ports["master_port"])
    env["BROWSER_AGENT_BASE_PORT"] = str(ports["base_port"])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["BROWSER_AGENT_DISTRIBUTED_BACKEND"] = str(
        config["hardware"]["distributed_backend"]
    )
    env["BROWSER_AGENT_MIXED_PRECISION"] = (
        "1" if config["hardware"]["mixed_precision"] else "0"
    )
    if chromium:
        env["PLAYWRIGHT_CHROMIUM_EXECUTABLE"] = chromium
    if bool(config["hardware"]["allow_tf32"]):
        env["NVIDIA_TF32_OVERRIDE"] = "1"
    else:
        env.pop("NVIDIA_TF32_OVERRIDE", None)
    return env


def choose_ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_port(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.1)
    fail(f"Local HTTP probe server did not start on port {port}")


def browser_probe(env: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        (root / "index.html").write_text(
            "<!doctype html><button id='ok'>Stage-1 browser probe</button>",
            encoding="utf-8",
        )
        port = choose_ephemeral_port()
        process = subprocess.Popen(
            [
                sys.executable, "-m", "http.server", str(port), "--bind",
                "127.0.0.1", "--directory", str(root),
            ],
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            wait_for_port(port)
            code = f'''
import os
from playwright.sync_api import sync_playwright
kwargs = {{"headless": True, "args": ["--disable-dev-shm-usage"]}}
if hasattr(os, "geteuid") and os.geteuid() == 0:
    kwargs["args"].append("--no-sandbox")
executable = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", "").strip()
if executable:
    kwargs["executable_path"] = executable
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(**kwargs)
    try:
        page = browser.new_page()
        response = page.goto("http://127.0.0.1:{port}/", wait_until="domcontentloaded", timeout=15000)
        assert response is not None and response.ok, response.status if response else None
        assert page.locator("#ok").inner_text() == "Stage-1 browser probe"
    finally:
        browser.close()
print("[PASS] Chromium local HTTP probe")
'''
            run([sys.executable, "-c", code], env=env)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def snapshot_runtime(
    config: dict[str, Any],
    torch_info: dict[str, Any],
    ports: dict[str, Any],
    chromium: str,
    effective_config: Path,
    binding_report: dict[str, Any],
) -> Path:
    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "server_toml": str(SERVER_TOML),
        "server_toml_sha256": sha256(SERVER_TOML),
        "effective_config": str(effective_config),
        "effective_config_sha256": sha256(effective_config),
        "binding_report": binding_report,
        "torch": torch_info,
        "ports": ports,
        "chromium": chromium or "playwright-managed",
        "gpu_ids": config["hardware"]["gpu_ids"],
    }
    output = runtime_dir / "runtime_manifest.json"
    write_json(output, value)
    return output


@contextlib.contextmanager
def activated_effective_config(
    config: dict[str, Any],
    effective_config: Path,
) -> Iterator[None]:
    base_config = project_path(config, "base_config", "config.toml")
    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    backup = runtime_dir / "original_config.toml"
    lock_path = runtime_dir / "config_swap.lock"

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if backup.exists():
            print("[RECOVERY] Restoring config.toml left by an interrupted run")
            shutil.copy2(backup, base_config)
            backup.unlink()
        shutil.copy2(base_config, backup)
        temporary = base_config.with_suffix(base_config.suffix + ".easy.tmp")
        shutil.copy2(effective_config, temporary)
        os.replace(temporary, base_config)
        try:
            yield
        finally:
            if backup.exists():
                shutil.copy2(backup, base_config)
                backup.unlink()
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def explicit_checkpoint(config: dict[str, Any]) -> Path | None:
    raw = str(config["training"]["checkpoint_path"]).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        fail(f"Configured checkpoint_path does not exist: {path}")
    if path.stat().st_size <= 0:
        fail(f"Configured checkpoint is empty: {path}")
    return path.resolve()


def latest_checkpoint(config: dict[str, Any]) -> Path | None:
    explicit = explicit_checkpoint(config)
    if explicit is not None:
        return explicit
    directory = project_path(config, "checkpoint_dir", "runs/checkpoints")
    candidates: list[Path] = []
    if directory.exists():
        for suffix in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(
                path for path in directory.rglob(suffix)
                if path.is_file() and path.stat().st_size > 0
            )
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def run_logged(
    command: list[str],
    env: dict[str, str],
    config: dict[str, Any],
    label: str,
) -> None:
    log_dir = project_path(config, "log_dir", "runs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{label}_{timestamp}.log"
    print(f"[LOG] {log_path}")
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        fail(f"Command failed with exit code {return_code}; see {log_path}")


def common_resolution(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, Path, dict[str, str]]:
    torch_info = inspect_project_torch(config)
    ports = resolve_ports(config)
    chromium = discover_chromium(config)
    effective_config, binding_report = prepare_effective_config(config, ports)
    env = runtime_environment(config, ports, chromium)
    snapshot_runtime(
        config,
        torch_info,
        ports,
        chromium,
        effective_config,
        binding_report,
    )
    return torch_info, ports, chromium, effective_config, env


def command_check(config: dict[str, Any]) -> None:
    with allocation_lock(config):
        torch_info, ports, chromium, effective_config, env = common_resolution(config)
        print(json.dumps(
            {"torch": torch_info, "ports": ports, "chromium": chromium or "managed"},
            ensure_ascii=False,
            indent=2,
        ))
        browser_probe(env)
        checks = config["check"]
        if checks["run_python_syntax"]:
            run([
                sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests",
            ], env=env)
        if checks["run_shell_syntax"]:
            for script in sorted(ROOT.rglob("*.sh")):
                if ".venv" not in script.parts and ".easy_bootstrap" not in script.parts:
                    run(["bash", "-n", str(script)], env=env)
        with activated_effective_config(config, effective_config):
            if checks["run_unit_tests"]:
                run([sys.executable, "-m", "pytest", "-q"], env=env)
            if checks["run_full_browser_smoke"]:
                script = ROOT / "run_smoke_linux.sh"
                if not script.is_file():
                    fail(f"Full browser smoke entry is missing: {script}")
                run_logged(["bash", str(script)], env, config, "smoke")
        print(f"[PASS] Unified check completed with {effective_config}")


def validate_training_entry() -> Path:
    script = ROOT / "run_train_8xa100.sh"
    if not script.is_file():
        fail(f"Missing original training entry: {script}")
    text = script.read_text(encoding="utf-8", errors="replace")
    if "--updates" not in text:
        fail("run_train_8xa100.sh does not support --updates")
    if "--resume" not in text:
        fail("run_train_8xa100.sh does not support --resume")
    return script


def training_command(
    config: dict[str, Any],
    updates: int,
    resume_policy: str,
) -> list[str]:
    script = validate_training_entry()
    command = ["bash", str(script), "--updates", str(updates)]
    checkpoint = latest_checkpoint(config)

    if resume_policy == "never":
        return command
    if resume_policy == "required":
        if checkpoint is None:
            fail("A checkpoint is required, but none was found")
        command.extend(["--resume", str(checkpoint)])
        return command
    if resume_policy == "auto":
        if checkpoint is not None:
            command.extend(["--resume", str(checkpoint)])
        else:
            print("[INFO] No checkpoint found; starting from random initialization")
        return command
    fail(f"Unknown resume policy: {resume_policy}")


def command_training(
    config: dict[str, Any],
    *,
    probe: bool,
    explicit_resume: bool,
) -> None:
    with allocation_lock(config):
        _, ports, _, effective_config, env = common_resolution(config)
        browser_probe(env)
        updates = int(
            config["training"]["probe_updates"]
            if probe else config["training"]["updates"]
        )
        if probe:
            resume_policy = "never"
            label = "ddp_probe"
        elif explicit_resume:
            resume_policy = "required"
            label = "resume"
        else:
            resume_policy = str(config["training"]["resume"])
            label = "train"
        command = training_command(config, updates, resume_policy)
        print(
            f"[PORTS] Browser tasks: {ports['base_port']}-{ports['last_port']}; "
            f"DDP: {ports['master_port']}"
        )
        # 在真正启动前再次检查；项目级锁持续持有，避免同项目并发抢占。
        assert_ports_still_free(ports)
        with activated_effective_config(config, effective_config):
            run_logged(command, env, config, label)


def command_status(config: dict[str, Any]) -> None:
    runtime_dir = project_path(config, "runtime_dir", "runtime")
    manifest = runtime_dir / "runtime_manifest.json"
    print("Browser Agent Stage 1 status")
    print("Root:", ROOT)
    print("Project environment:", venv_python(config))
    if manifest.is_file():
        print(manifest.read_text(encoding="utf-8"))
    else:
        print("Runtime manifest: not created yet")
    try:
        checkpoint = latest_checkpoint(config)
        print("Latest checkpoint:", checkpoint or "none")
    except RuntimeError as exc:
        print("Checkpoint configuration error:", exc)
    log_dir = project_path(config, "log_dir", "runs/logs")
    logs = (
        sorted(log_dir.glob("*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
        if log_dir.exists() else []
    )
    print("Latest log:", logs[0] if logs else "none")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser Agent Stage 1 unified fail-closed TOML launcher",
    )
    parser.add_argument(
        "command",
        choices=("setup", "check", "probe", "train", "resume", "status"),
        help="Only server.toml needs to be edited",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(actual_argv)
    config = server_config()

    if args.command != "status":
        ensure_environment(config, actual_argv)

    if args.command == "setup":
        with allocation_lock(config):
            torch_info, ports, chromium, effective_config, env = common_resolution(config)
            browser_probe(env)
            validate_training_entry()
            print(json.dumps({
                "torch": torch_info,
                "ports": ports,
                "chromium": chromium or "playwright-managed",
                "effective_config": str(effective_config),
            }, ensure_ascii=False, indent=2))
            print("[PASS] Setup completed")
    elif args.command == "check":
        command_check(config)
    elif args.command == "probe":
        command_training(config, probe=True, explicit_resume=False)
    elif args.command == "train":
        command_training(config, probe=False, explicit_resume=False)
    elif args.command == "resume":
        command_training(config, probe=False, explicit_resume=True)
    elif args.command == "status":
        command_status(config)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
