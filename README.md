# OPD 机理实验：Blackwell 远程节点手册

这份手册给有 8 卡 B200/B300 机器的合作者用。从零搭好环境后，这台机器就是一个长期的**远程节点**：按项目维护的任务队列逐项跑实验，产出自动上传到 Hugging Face 或 ModelScope（魔搭），状态每 10 分钟发一次心跳。

**队列（固定优先级，一次只开放一项，上一项审过才开放下一项）：** E10b 七臂（1.7B 学生）→ E11b（E10b 的 16K 重测与 TWR/α）→ E12b（tier A 起点的 GRPO）→ 零训练补项。当前开放：**E10b**。

**不扩张：** 不加 seed，不扩到 4B，不加新的方法臂。除非 `tasks/queue.txt` 更新，否则节点只跑开放阶段里列出的任务。

> **状态：uv 路线已在 1× B200 上实测通过安装、单元测试、accept 和 3 步冒烟（见第 8 节），完整版本锁在 `requirements.lock`。** 1.7B 学生的完整 150 步、8 卡同时跑、Docker 与 conda 路线都还没有实测。我们在 H100（CUDA 12.6，torch 2.6 + vLLM 0.8.5）上用的是旧栈。请仍然先跑"验证"一节的三项检查，并把日志发回。

流程：**安装 → 验证 → 启动心跳 → 领任务**。

**问题反馈**：遇到问题时，把 `status/` 与 `$OPD_ROOT/logs/` 上传到同一个结果仓库（`python scripts/remote.py put $OPD_ROOT/logs logs/<hostname>`），项目方从那里读，然后通过任务队列或本页的更新回应。

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | B200 或 B300，单机 8 卡（一臂一卡，7 臂 + 1 张空闲） |
| 驱动 / CUDA | NVIDIA 驱动 ≥ 570，CUDA ≥ 12.8（`nvidia-smi` 右上角显示的 CUDA Version） |
| 系统 | Linux x86_64，Python 3.10 或 3.11 |
| 盘 | 每臂约 60 GB（见第 5 节），7 臂建议 ≥ 600 GB 可用；只保留上传后的 bf16 权重时约 300 GB |
| 网络 | 能访问 PyPI 与 GitHub；Hugging Face 或其镜像（下载模型与数据）；HF 或 ModelScope（上传结果） |
| 凭证 | 由项目负责人直接交给你：`HF_TOKEN` + `HF_REPO`，或 `MS_TOKEN` + `MS_REPO`。**只放在你自己的 shell 环境变量里，不要写进任何文件或提交到仓库。** |

网络受限时：
- 模型与数据下载：`export HF_ENDPOINT=https://hf-mirror.com`（公开模型和数据集可用，只读）。模型也可以从 ModelScope 下载（`USE_MODELSCOPE=1`，Qwen3 在魔搭上的 ID 与 HF 相同）。
- 结果上传：hf-mirror 只读，不能上传。用 ModelScope：`export UPLOAD_BACKEND=modelscope MS_TOKEN=... MS_REPO=<你的命名空间>/<仓库名>`。

## 2. 一键安装（三选一）

所有路径都在 `OPD_ROOT` 下（默认 `$HOME/opd-run`）。

**A. uv（推荐）**
```bash
git clone https://github.com/yang3121099/opd-blackwell-runbook.git && cd opd-blackwell-runbook
bash scripts/install_uv.sh
```

**B. conda**
```bash
bash scripts/install_conda.sh
```

**C. Docker**：以 verl 官方的 CUDA 12.8 基础镜像为底（`verlai/verl:base-verl0.6-cu128-cudnn9.8-torch2.8.0-fa2.7.4`），见 `Dockerfile`：
```bash
docker build -t opd-e10b .
docker run --gpus all --ipc=host --shm-size=64g -v $HOME/opd-run:/root/opd-run -e HF_TOKEN -e HF_REPO -it opd-e10b
```

三种装法装的是同一组版本：

| 组件 | 版本 | 说明 |
|---|---|---|
| torch | 2.8.0（PyPI 默认 wheel，CUDA 12.8 运行时） | 含 Blackwell（sm_100）支持 |
| vLLM | 0.10.2 | 它固定 `torch==2.8.0`、`transformers>=4.55.2` |
| flashinfer-python | 0.3.0 | vLLM 0.10.2 的配套版本；PyPI 上只有源码包，首次使用会即时编译 |
| transformers | 4.56.x | 需 ≥ 4.55.2 |
| verl | G-OPD 分支（github.com/RUCBM/G-OPD），固定在提交 `37371a4c31ad7947746200d234161769191f4748`（基于 verl 0.6.1），再打本仓库 `code/patches` 里的补丁 | 我们在 H100 上用的是同一个提交 |
| 注意力实现 | 默认 **SDPA**（训练侧）、**FlashInfer**（vLLM 侧）；flash-attn 不装 | flash-attn 对 Blackwell 的支持还不稳定。装上并能 `import flash_attn` 时，脚本会自动打开 verl 的 `use_remove_padding`（更省显存），否则关闭 |

**兼容性风险（如实说明）**：G-OPD 原装脚本用的是 vLLM 0.8.5 + torch 2.6。verl 0.6.1 与 vLLM 0.10.2 的组合我们还没有跑过。如果第 3 节的冒烟失败，先看第 7 节"常见错误"。

## 3. 验证（必须先做，日志发回）

```bash
bash scripts/verify.sh 2>&1 | tee verify.log
```

依次做三件事，每项都打印 `PASS` / `FAIL`：
1. **G2 单元测试**：`code/patches/test_opd_variants.py` 的 16 个测试（只用 CPU）。
2. **accept**：GPU 型号、驱动与 CUDA、bf16 矩阵乘吞吐、磁盘写速度、下载速度、vLLM 能否启动并生成 4 道 MATH500。
3. **3 步冒烟**：0.6B 学生 + 8B 教师的 P2 臂跑 3 步（关闭开训前验证，不存盘），记录每步耗时、显存峰值、`actor/pg_loss`、熵和回答长度。三步都有有限的 loss、没有 NaN/inf、显存在卡的容量以内，就算通过。

全程约 1 小时。把 `verify.log` 发给项目负责人，三项都 PASS 之后再启动心跳和领任务。

与 H100 的数值对照不在验证里单独做：E10b 第一个臂的前 20 步，会由项目方事后与同配置的 H100 日志比对（第 0 步验证值、第 1 步各项指标、单步时间），不占你的机时。

## 4. 启动心跳，然后领任务

**心跳**：每 10 分钟把本机状态写成 `status/<hostname>.json`，上传到你的结果仓库（HF 或 ModelScope）的 `status/` 目录。内容有当前任务、当前步、最近一次验证值、截断率、各 GPU 利用率、磁盘余量和时间戳。我们这边只读这个目录，不会碰你的机器。
```bash
setsid nohup bash scripts/heartbeat.sh > $OPD_ROOT/logs/heartbeat.log 2>&1 &
```

**领任务**：队列在本仓库的 `tasks/queue.txt`，带版本号，列出各阶段和它的状态（open / locked）。`run_tasks.sh` 每 30 分钟 `git pull` 一次，只启动 open 阶段里、且在脚本白名单内的任务 ID；其他内容一律忽略，所以改队列文件不会让你的机器执行任意命令。项目方审完一个阶段的结果后，才会推送下一个版本、开放下一阶段（新阶段需要的脚本随那个版本一起发布）：
```bash
setsid nohup bash scripts/run_tasks.sh > $OPD_ROOT/logs/run_tasks.log 2>&1 &
```
每个任务一张卡；完成后写 `$OPD_ROOT/runs/<任务>/.done` 并上传 metrics。脚本幂等：重跑同一条命令，已完成的跳过，没完成的从最近的 ckpt 续跑。

## 5. 当前开放阶段：E10b

E10b = E10 的 7 个准备臂换成 1.7B 学生：学生 Qwen/Qwen3-1.7B（非思考模式），教师 Qwen/Qwen3-8B，DAPO-Math 训练题。除学生外，配置与 0.6B 的 E10 完全一致：每步 128 题、每题 1 条、回答上限 4K、150 步、data.seed 42、不加 KL、每 10 步在 MATH500 前 100 题上验证（avg@2，4K）。

**学习率 3e-6**（P4-lowlr 为 6e-7，与 0.6B 相同）。依据：0.6B 上按预注册规则从 {1e-6, 3e-6, 1e-5} 中选出 3e-6；1.7B 在 3e-6 下的 50 步试跑没有崩溃，验证值从第 0 步的 74.0 升到第 50 步的 79.5（+5.5，单次运行，4K）。

| 任务 ID | 臂 | 做什么（advantage A_t = log q_T − log p_θ 的变体） | 单臂预计（B200，外推） |
|---|---|---|---|
| `E10b-P2` | P2 | 标准 OPD（K1） | 8–12 h |
| `E10b-P3` | P3 | 按学生 logp 分 10 档，只在同档内打乱教师 logp | 8–12 h |
| `E10b-P4lowlr` | P4-lowlr | 学生 logp 最低的 20% token 固定 A=−0.5，lr 6e-7 | 6–10 h |
| `E10b-P5` | P5 | 教师用思考模板打分 | 10–14 h（回答几乎全部顶到 4K） |
| `E10b-P6` | P6 | 只保留 \|A\| 最大的 20% token | 8–12 h |
| `E10b-P6r` | P6r | P6 的补集（其余 80%） | 8–12 h |
| `E10b-P2noEOS` | P2noEOS | P2，但最后一个 EOS 处的 advantage 置 0 | 8–12 h |

时长按 H100 实测外推：1.7B 学生 + 8B 教师在 H100 上每步 300–490 s，0.6B 为 110–120 s。B200 按 1.5–2 倍算，**未实测**，以第一臂的实际步时为准。8 卡并行时 7 臂的总时长约等于最慢的一臂。

单独跑一臂（不用任务清单）：
```bash
bash scripts/run_arm.sh <gpu> <任务 ID>      # 例：bash scripts/run_arm.sh 0 E10b-P2
```

**盘**：每 25 步存一次 ckpt（第 25、50……150 步，含 E10 的判定点 50/100/150）。1.7B 每份含 fp32 HF 权重约 6.8 GB，续跑状态（模型分片 + 优化器）约 20 GB，只在最新一份保留。上传的是 bf16 副本（约 3.4 GB）。设 `KEEP_LOCAL_HF=0` 时，本地 fp32 权重在 bf16 副本上传成功后删除。

## 6. 上传、状态板与续跑

`run_arm.sh` 自动在后台启动 `scripts/upload_sync.sh`：每份 ckpt 存好后，把 bf16 权重和 `train.log`、`config.yaml` 传到结果仓库；任务结束时再传完整日志和逐步指标。仓库里的目录与本地一致：

```
runs/<任务>/global_step_<N>/actor/huggingface/   每份 ckpt 的 bf16 权重
runs/<任务>/train.log, config.yaml, metrics.jsonl
runs/<任务>/.done                               任务完成标记
status/<hostname>.json                           心跳
```

**状态板** = 心跳（`status/`）+ 完成标记（`.done`）+ 归档的权重与日志。项目方只读这三样，判断是否开放下一阶段。

- 上传后端：`UPLOAD_BACKEND=hf`（默认，用 `HF_TOKEN` 与 `HF_REPO`）或 `UPLOAD_BACKEND=modelscope`（用 `MS_TOKEN` 与 `MS_REPO`；ModelScope 的上传接口我们还没实测，第一次请先看 `upload.log` 里有没有 `UP_OK`）。
- 脚本只从环境变量读凭证，不打印、不写盘。
- **续跑**：机器重启或任务中断后，重跑同一条 `run_arm.sh` 或 `run_tasks.sh` 即可。verl 的 `resume_mode=auto` 会从本地最近的 ckpt（模型 + 优化器 + 数据加载状态）接着跑。

## 7. 常见错误

| 现象 | 原因与处理 |
|---|---|
| `nvidia-smi` 的 CUDA Version < 12.8 | 驱动太旧，需要 ≥ 570。 |
| `CUDA error: no kernel image is available for execution on the device` | 某个包的 wheel 没编进 sm_100。先确认 `python -c "import torch;print(torch.cuda.get_arch_list())"` 里有 `sm_100`；若是 flash-attn，卸载它（脚本会自动改用 SDPA）。 |
| flashinfer 首次运行很慢或编译失败 | flashinfer 0.3.0 只有源码包，首次使用 JIT 编译需要几分钟和 `nvcc`。可改用 vLLM 自带的注意力后端：`export VLLM_ATTENTION_BACKEND=FLASH_ATTN`，或者用 Docker 路线。 |
| vLLM 初始化时 KV cache 不够 | 学生、教师和 vLLM 在同一张卡上。调低 `VLLM_UTIL`（默认 0.5）；B200 显存够时，可设 `TEACHER_OFFLOAD=0` 让教师常驻显存，速度更快。 |
| Ray 报 `pids` 或线程数超限 | 已默认 `RAY_prestart_worker_first_driver=0`、`RAY_num_workers_soft_limit=8`；同机多臂时让启动错开（`run_tasks.sh` 默认间隔 90 s）。 |
| 下载模型或数据失败 | 设 `HF_ENDPOINT=https://hf-mirror.com`，或 `USE_MODELSCOPE=1` 从魔搭下载模型。 |
| `hf auth whoami` 显示 "Not logged in" 但退出码是 0 | 这一版 CLI 的已知行为。脚本用 `HfApi().whoami()` 判断登录；只要设置了 `HF_TOKEN`，就不需要 `hf auth login`。 |

## 8. Blackwell 验证记录

**2026-10-16，在一台临时的 1× B200 上实测（uv 路线）。** Docker 路线和 conda 路线尚未实测。

| 项 | 结果 |
|---|---|
| 机器 | 1× NVIDIA B200 183 GB，驱动 580.126.09，`nvidia-smi` 显示 CUDA 13.0；Ubuntu，系统 Python 3.12（venv 用 uv 装的 3.10） |
| 版本 | torch 2.8.0+cu128（`get_arch_list()` 含 `sm_100`），vLLM 0.10.2，transformers 4.56.2，flashinfer-python 0.3.0，datasets 4.0.0，pyarrow 25.0.1，ray 2.59.0；G-OPD 提交 37371a4；flash-attn 未装（SDPA）。完整版本锁见 `requirements.lock`（225 个包），可用 `uv pip install -r requirements.lock` 复现 |
| 安装耗时 | 依赖约 40 s，G-OPD + 补丁 + 三个模型 + 数据约 1 分钟（美国机房，下载很快；国内网络会慢得多） |
| G2 单元测试 | 16 个全部通过 |
| accept | bf16 矩阵乘约 307 TFLOPS；磁盘写 5.5 GB/s；vLLM 能起并生成 |
| 3 步冒烟（0.6B 学生 + 8B 教师，P2） | 通过：三步都有有限的 loss（`actor/pg_loss` 0.0092 / 0.0081 / 0.0066），无 NaN，无 OOM。每步 161 s / 102 s / 105 s（第 1 步含教师首次加载；之后生成约 23 s、教师打分 26–28 s、actor 更新约 41 s）。熵 0.49–0.50，回答平均 1058–1363 token，`opd/adv_abs_mean` 0.33–0.35 |
| 显存 | verl 报告的 `perf/max_memory_allocated_gb` 为 194.8、`max_memory_reserved_gb` 为 244.3，都超过了卡的 179 GiB，所以不是单卡峰值的真实读数（多半把 vLLM 休眠时换出的内存也算了进去）。以"无 OOM"为准 |
| 验证中修掉的问题 | ① uv 建的 venv 里没有 pip，裸 `pip` 会装到系统 Python → 改用 `uv pip`；② uv 把 datasets 解析成 2.14.4，与 pyarrow 25 不兼容 → 固定 `datasets>=3.0,<4.1`；③ vLLM 检查从 stdin 运行时，spawn 子进程无法重新导入 → 改成脚本文件；④ 没有 flash-attn 时 16K 的 token 预算在 actor 更新时 OOM（要分配 30 GiB 的 logits）→ 改为 6144 |

作为参照，同配置（0.6B 学生 + 8B 教师，P2）在我们的 H100 上第 1 步 179 s，之后 110–120 s/步（那边用 flash-attn 去 padding）。数值对照（第 0 步验证值、第 1 步各项指标）会在你的 E10b 第一个臂跑到第 20 步后由项目方事后做。
