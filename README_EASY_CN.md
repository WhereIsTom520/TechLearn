# Browser Agent Stage 1 — 8×A100 傻瓜版

本版本基于 `browser_agent_stage1_v0.1.4`，只增加服务器环境、端口、配置和启动管理层，不修改已经审核过的浏览器动作、PPO、课程和任务核心。

## 目标

- 随机初始化，不加载预训练模型；
- DOM/ARIA优先，第一阶段不使用截图；
- 独立Playwright Chromium；
- 真实浏览器rollout、奖励、GAE和PPO；
- 8张A100通过NCCL训练；
- 训练、端口、GPU、Chromium、检查点和日志都由一个`server.toml`管理。

## 构建完整包

把以下两个文件放到同一个目录：

```text
browser_agent_stage1_v0.1.4.zip
browser_agent_stage1_easy_overlay_v0.1.5/
```

运行：

```bash
cd browser_agent_stage1_easy_overlay_v0.1.5
chmod +x apply_easy_overlay.sh
./apply_easy_overlay.sh ../browser_agent_stage1_v0.1.4.zip ..
```

会生成：

```text
browser_agent_stage1_easy_v0.1.5-easy8a100.zip
browser_agent_stage1_easy_v0.1.5-easy8a100.zip.sha256
```

## 服务器使用

```bash
unzip browser_agent_stage1_easy_v0.1.5-easy8a100.zip
cd browser_agent_stage1_easy_v0.1.5-easy8a100
```

只编辑：

```text
server.toml
```

首次准备环境：

```bash
./start.sh setup
```

完整冒烟检查：

```bash
./start.sh check
```

8卡两次更新验证：

```bash
./start.sh probe
```

正式训练：

```bash
./start.sh train
```

查看状态：

```bash
./start.sh status
```

强制恢复训练：

```bash
./start.sh resume
```

## 环境策略

程序会验证服务器现有环境：

```text
Python 3.10
PyTorch 2.11.x
CUDA 12.8
至少8张GPU
NCCL可用
```

随后创建项目内`.venv`，通过`.pth`复用已经验证的CUDA PyTorch，仅在项目环境内安装Playwright、pytest、tomli、tomli-w和psutil。不会重新安装或替换Torch。

## 端口策略

`server.toml`默认：

```toml
[ports]
base_port = 18500
master_port = 29510
auto_find_free_block = true
auto_find_master_port = true
```

程序启动前会寻找连续空闲端口，并把最终配置写入：

```text
runtime/effective_config.toml
runtime/override_report.json
runtime/runtime_manifest.json
```

## 配置覆盖

启动器会按键名把`server.toml`中的常用训练参数覆盖到原始`config.toml`副本中。原始配置不会永久修改；训练期间使用有效配置，退出后自动恢复。

如果原项目以后改变了内部TOML路径，可以通过：

```toml
[overrides]
"training.rollout_steps" = 128
```

添加精确覆盖。

## 正式训练门槛

只有以下命令都成功后再长训：

```bash
./start.sh check
./start.sh probe
```

第一阶段仍不包含截图视觉编码器、Canvas、closed Shadow DOM、验证码和真实支付操作；这些属于第二阶段或安全限制范围。
