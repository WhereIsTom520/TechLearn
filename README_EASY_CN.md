# Browser Agent Stage 1 — 8×A100 傻瓜版 v0.1.6

本版本基于 `browser_agent_stage1_v0.1.4`，增加服务器环境、端口、统一配置和启动管理层，不修改浏览器动作、PPO、课程和任务核心。

## 第一阶段目标

- 模型随机初始化，不加载预训练权重；
- DOM/ARIA优先，第一阶段不使用截图；
- 使用独立Playwright Chromium；
- 执行真实浏览器rollout、奖励、GAE和PPO；
- 8张A100通过NCCL训练；
- 训练参数、端口、GPU、Chromium和检查点由一个`server.toml`管理；
- 完整训练只允许本地HTTP任务服务器；
- 外部网络和浏览器下载能力失败封闭。

## v0.1.6修复内容

1. Python 3.10首次启动不再预先依赖`tomli`；`start.sh`会创建独立Bootstrap。
2. 残缺`.venv`会被检测并自动重建，不会因为存在`bin/python`就误判健康。
3. `server.toml`采用固定Schema；未知、缺失或类型错误字段会立即失败。
4. 配置覆盖不再递归修改所有同名键；每个参数必须精确绑定或唯一匹配。
5. 参数在基础`config.toml`中找不到或存在歧义时拒绝启动，不再静默忽略。
6. `./start.sh resume`必须找到非空检查点，否则失败，不会悄悄从零开始。
7. 明确校验`ports_per_gpu >= environments_per_gpu`。
8. DDP端口不得与浏览器端口块重叠；启动前执行二次端口检查。
9. 项目级端口锁和配置交换锁防止同一项目重复启动互相覆盖。
10. Chromium探针会真正访问本地HTTP页面，不再只测试`page.set_content()`。
11. 构建时强制执行Shell和Python语法检查；失败时不会生成ZIP。
12. 构建时检查原训练入口确实支持`--updates`和`--resume`。

## 构建完整包

把以下内容放到同一个目录：

```text
browser_agent_stage1_v0.1.4.zip
本傻瓜版覆盖层目录
```

运行：

```bash
chmod +x apply_easy_overlay.sh
./apply_easy_overlay.sh ./browser_agent_stage1_v0.1.4.zip .
```

成功后生成：

```text
browser_agent_stage1_easy_v0.1.6-easy8a100.zip
browser_agent_stage1_easy_v0.1.6-easy8a100.zip.sha256
```

构建器只有在以下检查全部通过后才会产包：

```text
基础包结构检查
训练入口参数检查
start.sh Shell语法
原8卡与冒烟脚本Shell语法
easy_launcher.py Python语法
Python 3.10与Torch 2.11元数据改造
内容SHA-256清单
```

## 服务器使用

```bash
unzip browser_agent_stage1_easy_v0.1.6-easy8a100.zip
cd browser_agent_stage1_easy_v0.1.6-easy8a100
```

只编辑：

```text
server.toml
```

依次运行：

```bash
./start.sh setup
./start.sh check
./start.sh probe
./start.sh train
```

### 命令含义

```text
setup   创建或修复项目环境，解析全部TOML绑定，验证本地HTTP Chromium
check   运行语法、单元测试和完整浏览器冒烟
probe   运行8卡两次真实PPO更新，不自动恢复旧检查点
train   按training.resume策略正式训练
resume  强制从现有检查点恢复；没有检查点立即失败
status  显示最后一次运行配置、端口、检查点和日志
```

## 环境策略

程序验证服务器现有环境：

```text
Python 3.10
PyTorch 2.11.x
CUDA 12.8
8张GPU
NCCL可用
```

项目内创建`.venv`，通过`.pth`复用服务器已经验证的CUDA PyTorch。Playwright、pytest、TOML解析器等安装在项目环境中，不重新安装Torch。

如果`.venv`存在但无法导入以下内容，会自动重建：

```text
torch
playwright
tomli
tomli_w
browser_agent
```

## 配置绑定原则

启动器只接受`server.toml`声明的字段。保留字段必须实际生效。

对于每个训练参数：

```text
server.toml源字段
→ 基础config.toml精确路径或唯一匹配
→ 类型兼容检查
→ runtime/effective_config.toml
```

出现以下任一情况都会停止：

```text
基础配置中找不到目标
发现多个可能目标
目标类型不兼容
server.toml存在未知字段
必要字段缺失
```

基础包未来改变Schema时，可在`server.toml`中写精确绑定：

```toml
[bindings]
"training.rollout_steps" = "ppo.rollout_steps"
```

实际绑定结果记录在：

```text
runtime/binding_report.json
runtime/effective_config.toml
runtime/runtime_manifest.json
```

## 端口策略

默认：

```toml
[ports]
host = "127.0.0.1"
base_port = 18500
ports_per_gpu = 1
master_port = 29510
auto_find_free_block = true
auto_find_master_port = true
```

8张GPU、每张1个环境时预留8个连续任务端口。若增加：

```toml
[hardware]
environments_per_gpu = 2
```

必须同步满足：

```toml
[ports]
ports_per_gpu = 2
```

程序会避免DDP端口与任务端口重叠，并在真正启动训练前再次检查端口。

## 检查点规则

```toml
[training]
resume = "auto" # auto | never | required
checkpoint_path = ""
```

- `auto`：找到检查点则恢复，否则从随机初始化开始；
- `never`：始终从随机初始化开始；
- `required`：找不到检查点就失败；
- `./start.sh resume`：无论配置是什么，都强制要求检查点存在；
- 指定的`checkpoint_path`不存在或为空文件时立即失败。

## 正式训练门槛

以下三项必须全部成功：

```bash
./start.sh setup
./start.sh check
./start.sh probe
```

之后才能运行：

```bash
./start.sh train
```

第一阶段仍不包含截图视觉编码器、Canvas、closed Shadow DOM、验证码和真实支付操作；这些属于第二阶段或明确的安全限制。
