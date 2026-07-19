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
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
SERVER_TOML = ROOT / "server.toml"


def fail(message: str) -> "NoReturn":
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
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
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


def capture(command: list[str], *, cwd: Path = ROOT) -> str:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def server_config() -> dict[str, Any]:
    if not SERVER_TOML.is_file():
        fail(f"Missing unified configuration: {SERVER_TOML}")
    return load_toml(SERVER_TOML)


def project_path(config: dict[str, Any], key: str, default: str) -> Path:
    raw = str(config.get("project", {}).get(key, default))
    path = Path(raw).expanduser()
    return path if path.is_absolute() else ROOT / path


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

    env_cfg = config["environment"]
    expected_python = str(env_cfg.get("expected_python", "3.10"))
    expected_torch = str(env_cfg.get("expected_torch_prefix", "2.11."))
    expected_cuda = str(env_cfg.get("expected_cuda", "12.8"))
    required_gpus = int(env_cfg.get("required_gpu_count", 8))

    errors: list[str] = []
    if value["python"] != expected_python:
        errors.append(f"Python expected {expected_python}, got {value['python']}")
    if not str(value["torch"]).startswith(expected_torch):
        errors.append(f"PyTorch expected prefix {expected_torch}, got {value['torch']}")
    if str(value["torch_cuda"]) != expected_cuda:
        errors.append(f"CUDA expected {expected_cuda}, got {value['torch_cuda']}")
    if not value["cuda_available"]:
        errors.append("torch.cuda.is_available() is false")
    if int(value["gpu_count"]) < required_gpus:
        errors.append(f"Expected at least {required_gpus} GPUs, got {value['gpu_count']}")
    if not value["nccl_available"]:
        errors.append("NCCL backend is unavailable")
    if errors:
        fail("Server environment mismatch:\n- " + "\n- ".join(errors))
    return value


def venv_python(config: dict[str, Any]) -> Path:
    return project_path(config, "venv_dir", ".venv") / "bin" / "python"


def current_is_venv(config: dict[str, Any]) -> bool:
    target = venv_python(config)
    try:
        return Path(sys.executable).resolve() == target.resolve()
    except FileNotFoundError:
        return False


def ensure_environment(config: dict[str, Any], argv: list[str]) -> None:
    if current_is_venv(config):
        return

    torch_info = inspect_system_torch(config)
    venv_dir = project_path(config, "venv_dir", ".venv")
    python = system_python(config)
    auto_create = bool(config["project"].get("auto_create_environment", True))

    if not venv_python(config).is_file():
        if not auto_create:
            fail(f"Project environment is missing: {venv_dir}")
        print(f"[SETUP] Creating project environment: {venv_dir}")
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        run([python, "-m", "venv", str(venv_dir)])

        site = capture([
            str(venv_python(config)),
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ])
        pth = Path(site) / "00_dolphin_cuda_torch.pth"
        pth.write_text(str(torch_info["torch_site"]) + "\n", encoding="utf-8")

        pip_index = str(config["environment"].get("pip_index", "")).strip()
        index_args = ["-i", pip_index] if pip_index else []
        run([
            str(venv_python(config)), "-m", "pip", "install", "--upgrade",
            "pip", "setuptools", "wheel", *index_args,
        ])
        run([
            str(venv_python(config)), "-m", "pip", "install",
            "playwright>=1.60,<2", "pytest>=8,<10", "tomli>=2.2,<3",
            "tomli-w>=1.2,<2", "psutil>=5.9,<8", *index_args,
        ])
        run([
            str(venv_python(config)), "-m", "pip", "install",
            "--editable", str(ROOT), "--no-deps",
        ])

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
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nccl_available": bool(torch.distributed.is_available() and torch.distributed.is_nccl_available()),
    }
    expected = inspect_system_torch(config)
    if value["torch"] != expected["torch"]:
        fail(f"Project venv replaced PyTorch: {value['torch']} != {expected['torch']}")
    if str(value["torch_cuda"]) != str(expected["torch_cuda"]):
        fail("Project venv CUDA runtime differs from the verified system runtime")
    return value


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_block(host: str, start: int, count: int, limit: int) -> int:
    if count <= 0:
        fail("Port block size must be positive")
    for candidate in range(start, limit - count + 2):
        if all(port_is_free(host, candidate + offset) for offset in range(count)):
            return candidate
    fail(f"No continuous block of {count} free ports found between {start} and {limit}")


def resolve_ports(config: dict[str, Any]) -> dict[str, Any]:
    ports = config["ports"]
    hardware = config["hardware"]
    host = str(ports.get("host", "127.0.0.1"))
    gpu_count = len(hardware.get("gpu_ids", []))
    ports_per_gpu = int(ports.get("ports_per_gpu", 1))
    count = max(1, gpu_count * ports_per_gpu)
    base = int(ports.get("base_port", 18500))
    limit = int(ports.get("search_until_port", 25000))

    if bool(ports.get("auto_find_free_block", True)):
        base = find_free_block(host, base, count, limit)
    elif not all(port_is_free(host, base + i) for i in range(count)):
        fail(f"Configured browser port block {base}-{base + count - 1} is occupied")

    master = int(ports.get("master_port", 29510))
    if not port_is_free(host, master):
        if bool(ports.get("auto_find_master_port", True)):
            master = find_free_block(host, max(master, base + count), 1, 65535)
        else:
            fail(f"Configured DDP master port {master} is occupied")

    return {
        "host": host,
        "base_port": base,
        "port_count": count,
        "last_port": base + count - 1,
        "master_port": master,
    }


def discover_chromium(config: dict[str, Any]) -> str:
    env_cfg = config["environment"]
    mode = str(env_cfg.get("chromium_mode", "auto")).lower()
    configured = str(env_cfg.get("chromium_executable", "")).strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    env_candidate = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", "").strip()
    if env_candidate:
        candidates.append(Path(env_candidate).expanduser())
    candidates.extend(Path(p) for p in (
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    ))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())

    if mode == "system":
        fail("chromium_mode=system but no executable was found")
    if bool(env_cfg.get("install_playwright_browser", True)):
        run([sys.executable, "-m", "playwright", "install", "chromium"])
        return ""
    if mode == "playwright":
        fail("Playwright Chromium is unavailable and automatic installation is disabled")
    return ""


def recursive_paths(value: Any, key: str, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    found: list[tuple[str, ...]] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            path = (*prefix, str(child_key))
            if child_key == key:
                found.append(path)
            found.extend(recursive_paths(child_value, key, path))
    return found


def set_path(data: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cursor: dict[str, Any] = data
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def set_existing_key(
    data: dict[str, Any],
    keys: Iterable[str],
    value: Any,
) -> list[str]:
    changed: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for key in keys:
        for path in recursive_paths(data, key):
            if path in seen:
                continue
            cursor: Any = data
            for part in path[:-1]:
                cursor = cursor[part]
            cursor[path[-1]] = value
            seen.add(path)
            changed.append(".".join(path))
    return changed


def get_server_value(config: dict[str, Any], dotted: str) -> Any:
    cursor: Any = config
    for part in dotted.split("."):
        cursor = cursor[part]
    return cursor


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

    mappings: list[tuple[str, tuple[str, ...]]] = [
        ("training.seed", ("seed",)),
        ("training.rollout_steps", ("rollout_steps",)),
        ("training.ppo_epochs", ("ppo_epochs", "update_epochs")),
        ("training.minibatch_size", ("minibatch_size", "mini_batch_size")),
        ("training.learning_rate", ("learning_rate", "policy_learning_rate", "lr")),
        ("training.value_learning_rate", ("value_learning_rate", "value_lr")),
        ("training.gamma", ("gamma",)),
        ("training.gae_lambda", ("gae_lambda",)),
        ("training.clip_ratio", ("clip_ratio", "clip_coef")),
        ("training.value_clip_ratio", ("value_clip_ratio", "value_clip_coef")),
        ("training.entropy_coef", ("entropy_coef",)),
        ("training.value_coef", ("value_coef",)),
        ("training.text_aux_coef", ("text_aux_coef",)),
        ("training.max_grad_norm", ("max_grad_norm",)),
        ("training.target_kl", ("target_kl",)),
        ("training.checkpoint_every_updates", ("checkpoint_every_updates", "checkpoint_interval")),
        ("training.evaluation_every_updates", ("evaluation_every_updates", "eval_interval")),
        ("training.log_every_updates", ("log_every_updates", "log_interval")),
        ("hardware.environments_per_gpu", ("environments_per_gpu", "envs_per_rank", "num_envs")),
        ("browser.headless", ("headless",)),
        ("browser.page_width", ("page_width", "viewport_width")),
        ("browser.page_height", ("page_height", "viewport_height")),
        ("browser.navigation_timeout_ms", ("navigation_timeout_ms",)),
        ("browser.action_timeout_ms", ("action_timeout_ms",)),
        ("browser.episode_timeout_seconds", ("episode_timeout_seconds", "episode_timeout_s")),
        ("browser.disable_service_workers", ("disable_service_workers",)),
        ("browser.allow_downloads", ("allow_downloads",)),
        ("browser.restart_failed_environment", ("restart_failed_environment",)),
        ("browser.max_environment_restarts", ("max_environment_restarts",)),
        ("tasks.train_template_variants", ("train_template_variants",)),
        ("tasks.evaluation_template_variants", ("evaluation_template_variants", "eval_template_variants")),
        ("tasks.max_episode_steps", ("max_episode_steps", "max_steps")),
        ("tasks.max_elements", ("max_elements",)),
        ("tasks.max_arguments", ("max_arguments", "max_args")),
        ("tasks.max_generated_text_tokens", ("max_generated_text_tokens",)),
        ("curriculum.enabled", ("curriculum_enabled",)),
        ("curriculum.minimum_episodes_per_task", ("minimum_episodes_per_task", "min_episodes_per_task")),
        ("curriculum.success_rate_threshold", ("success_rate_threshold",)),
        ("curriculum.previous_task_probability", ("previous_task_probability",)),
        ("security.allow_external_network", ("allow_external_network",)),
        ("security.allowed_hosts", ("allowed_hosts",)),
        ("security.strict_origin_whitelist", ("strict_origin_whitelist",)),
    ]

    report: dict[str, Any] = {"base_config": str(base_config), "auto": {}, "explicit": {}}
    for source, candidates in mappings:
        value = get_server_value(config, source)
        changed = set_existing_key(effective, candidates, value)
        report["auto"][source] = {"value": value, "changed": changed}

    port_changed = set_existing_key(
        effective,
        ("base_port", "server_base_port", "task_server_port"),
        int(ports["base_port"]),
    )
    report["auto"]["ports.base_port"] = {
        "value": ports["base_port"],
        "changed": port_changed,
    }
    if not port_changed:
        print("[WARN] No base_port key was found in config.toml; MASTER_PORT is still configured.")

    enabled_tasks = config.get("tasks", {}).get("enabled", [])
    if enabled_tasks:
        changed = set_existing_key(effective, ("task_kinds", "enabled_tasks"), enabled_tasks)
        report["auto"]["tasks.enabled"] = {"value": enabled_tasks, "changed": changed}

    for dotted, value in config.get("overrides", {}).items():
        set_path(effective, str(dotted), value)
        report["explicit"][str(dotted)] = value

    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    output = runtime_dir / "effective_config.toml"
    temporary = output.with_suffix(".toml.tmp")
    temporary.write_text(tomli_w.dumps(effective), encoding="utf-8")
    os.replace(temporary, output)
    report["effective_config"] = str(output)
    report["effective_config_sha256"] = sha256(output)
    write_json(runtime_dir / "override_report.json", report)
    return output, report


def runtime_environment(
    config: dict[str, Any],
    ports: dict[str, Any],
    chromium: str,
) -> dict[str, str]:
    env = os.environ.copy()
    gpu_ids = [str(item) for item in config["hardware"].get("gpu_ids", [])]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["MASTER_ADDR"] = str(ports["host"])
    env["MASTER_PORT"] = str(ports["master_port"])
    env["BROWSER_AGENT_BASE_PORT"] = str(ports["base_port"])
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    if chromium:
        env["PLAYWRIGHT_CHROMIUM_EXECUTABLE"] = chromium
    if bool(config["hardware"].get("allow_tf32", True)):
        env["NVIDIA_TF32_OVERRIDE"] = "1"
    return env


def browser_probe(env: dict[str, str]) -> None:
    code = r'''
import os
from playwright.sync_api import sync_playwright
kw = {"headless": True, "args": ["--disable-dev-shm-usage"]}
exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE", "").strip()
if exe:
    kw["executable_path"] = exe
with sync_playwright() as p:
    browser = p.chromium.launch(**kw)
    try:
        page = browser.new_page()
        page.set_content("<button id='ok'>OK</button>")
        assert page.locator("#ok").inner_text() == "OK"
    finally:
        browser.close()
print("[PASS] Chromium probe")
'''
    run([sys.executable, "-c", code], env=env)


def snapshot_runtime(
    config: dict[str, Any],
    torch_info: dict[str, Any],
    ports: dict[str, Any],
    chromium: str,
    effective_config: Path,
) -> Path:
    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "server_toml": str(SERVER_TOML),
        "server_toml_sha256": sha256(SERVER_TOML),
        "effective_config": str(effective_config),
        "effective_config_sha256": sha256(effective_config),
        "torch": torch_info,
        "ports": ports,
        "chromium": chromium or "playwright-managed",
        "gpu_ids": config["hardware"].get("gpu_ids", []),
    }
    output = runtime_dir / "runtime_manifest.json"
    write_json(output, value)
    return output


@contextlib.contextmanager
def activated_effective_config(
    config: dict[str, Any],
    effective_config: Path,
):
    base_config = project_path(config, "base_config", "config.toml")
    runtime_dir = project_path(config, "runtime_dir", "runtime")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    backup = runtime_dir / "original_config.toml"
    lock_path = runtime_dir / "config_swap.lock"

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if backup.exists():
            print("[RECOVERY] Restoring config.toml left by an interrupted launcher run")
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


def latest_checkpoint(config: dict[str, Any]) -> Path | None:
    explicit = str(config["training"].get("checkpoint_path", "")).strip()
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        return path if path.is_file() else None
    directory = project_path(config, "checkpoint_dir", "runs/checkpoints")
    candidates: list[Path] = []
    if directory.exists():
        for suffix in ("*.pt", "*.pth", "*.ckpt"):
            candidates.extend(directory.rglob(suffix))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


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


def common_resolution(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str, Path, dict[str, str]]:
    torch_info = inspect_project_torch(config)
    ports = resolve_ports(config)
    chromium = discover_chromium(config)
    effective_config, _ = prepare_effective_config(config, ports)
    env = runtime_environment(config, ports, chromium)
    snapshot_runtime(config, torch_info, ports, chromium, effective_config)
    return torch_info, ports, chromium, effective_config, env


def command_check(config: dict[str, Any]) -> None:
    torch_info, ports, chromium, effective_config, env = common_resolution(config)
    print(json.dumps({"torch": torch_info, "ports": ports, "chromium": chromium or "managed"}, ensure_ascii=False, indent=2))
    browser_probe(env)

    checks = config.get("check", {})
    if bool(checks.get("run_python_syntax", True)):
        run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"], env=env)
    if bool(checks.get("run_shell_syntax", True)):
        scripts = sorted(str(path) for path in ROOT.glob("*.sh"))
        for script in scripts:
            run(["bash", "-n", script], env=env)
    if bool(checks.get("run_unit_tests", True)):
        run([sys.executable, "-m", "pytest", "-q"], env=env)
    if (ROOT / "run_smoke_linux.sh").is_file():
        run_logged(["bash", "run_smoke_linux.sh"], env, config, "smoke")
    else:
        print("[WARN] run_smoke_linux.sh is missing; only direct checks were run")
    print(f"[PASS] Unified check completed with effective config {effective_config}")


def training_command(
    config: dict[str, Any],
    updates: int,
    resume: bool,
) -> list[str]:
    script = ROOT / "run_train_8xa100.sh"
    if not script.is_file():
        fail(f"Missing original training entry: {script}")
    command = ["bash", str(script), "--updates", str(updates)]
    if resume:
        checkpoint = latest_checkpoint(config)
        mode = str(config["training"].get("resume", "auto")).lower()
        if checkpoint is not None:
            command.extend(["--resume", str(checkpoint)])
        elif mode == "required":
            fail("Resume was required but no checkpoint was found")
        elif mode == "auto":
            print("[INFO] No checkpoint found; starting from random initialization")
    return command


def command_training(config: dict[str, Any], *, probe: bool, resume: bool) -> None:
    _, ports, chromium, effective_config, env = common_resolution(config)
    browser_probe(env)
    updates = int(
        config["training"].get("probe_updates", 2)
        if probe
        else config["training"].get("updates", 10000)
    )
    label = "ddp_probe" if probe else "train"
    command = training_command(config, updates, resume)
    print(f"[PORTS] Browser tasks: {ports['base_port']}-{ports['last_port']}; DDP: {ports['master_port']}")
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
    checkpoint = latest_checkpoint(config)
    print("Latest checkpoint:", checkpoint or "none")
    log_dir = project_path(config, "log_dir", "runs/logs")
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True) if log_dir.exists() else []
    print("Latest log:", logs[0] if logs else "none")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browser Agent Stage 1 unified TOML launcher",
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
        torch_info, ports, chromium, effective_config, env = common_resolution(config)
        browser_probe(env)
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
        command_training(config, probe=True, resume=False)
    elif args.command == "train":
        resume_mode = str(config["training"].get("resume", "auto")).lower()
        command_training(config, probe=False, resume=resume_mode in {"auto", "required"})
    elif args.command == "resume":
        command_training(config, probe=False, resume=True)
    elif args.command == "status":
        command_status(config)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
