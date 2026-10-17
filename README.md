# OPD 机理实验：Blackwell 远程节点手册

> **已经克隆过本仓库的节点（v2 或更早）请先更新到最新版（v2.3.1，含 v2.1–v2.3 的修复）**：旧版的上传脚本会悄悄跳过所有分片保存的 1.7B ckpt，最后一个 ckpt 也可能漏传。步骤：`git pull` → 重跑 `verify.sh` 的第 4 项（UPLOAD_CHECK 应为 PASS）→ 按第 9 节"变更记录"的补传说明重启 `upload_sync.sh`，把本机已存的分片 ckpt 传上去。

> **B300 起夜重跑（v2.3.1）**：第一晚停在第 6 步的 vLLM 探测，原因已经查清，修复在 v2.3.1 里。只需要下面三行；如果原来那个 shell 还开着，第二行可以省掉：
> ```bash
> cd opd-blackwell-runbook && git pull
> export OPD_ROOT=...  HF_TOKEN=...  HF_REPO=...       # 和第一晚相同的值，只在当前 shell 里设
> setsid nohup bash scripts/overnight.sh > /dev/null 2>&1 &
> ```
> 脚本可以重跑：模型、数据和 venv 都会直接复用，几分钟内就会走到第 6 步。进度用 `grep STEP $OPD_ROOT/logs/overnight.log` 查看；同一时间只能运行一份。

这份手册给有 8 卡 B200/B300 机器的合作者用。从零搭好环境后，这台机器就是一个长期的**远程节点**：按项目维护的任务队列逐项跑实验，产出自动上传到 Hugging Face 或 ModelScope（魔搭），状态每 10 分钟发一次心跳。

**队列（版本 2，固定阶梯，前一阶段结束就自动开放下一阶段）：** E10b 七臂（1.7B 学生）→ E11b（16K 重测与 TWR/α）→ E12b tier A（含 1.7B 的 SFT 起点 P1）→ 零训练补项（E03、E02）→ E12b 其余起点 → 1.7B 第二组臂 → P2/P6r 的种子复制。项目方可以随时改顺序或暂停某一阶段。

**不扩张：** 不加 seed，不扩到 4B，不加新的方法臂。除非 `tasks/queue.txt` 更新，否则节点只跑开放阶段里列出的任务。

> **状态：uv 路线已在 1× B200 上实测通过安装、单元测试、accept 和 3 步冒烟（见第 8 节），完整版本锁在 `requirements.lock`。** 1.7B 学生的完整 150 步、8 卡同时跑、Docker 与 conda 路线都还没有实测。我们在 H100（CUDA 12.6，torch 2.6 + vLLM 0.8.5）上用的是旧栈。请仍然先跑"验证"一节的四项检查，并把日志发回。

流程：**安装 → 验证 → 启动心跳 → 领任务**。

**问题反馈（GitHub Issue，v2.3.1）**：在 https://github.com/yang3121099/opd-blackwell-runbook/issues 开一个 Issue。标题写主机名（`hostname` 的输出）和一句话描述现象；正文贴 `grep STEP $OPD_ROOT/logs/overnight.log` 的输出，以及日志在结果仓库里的位置：先用 `python scripts/remote.py put $OPD_ROOT/logs logs/$(hostname)` 上传，再贴 `logs/<hostname>/` 这个路径。**不要贴 token 或任何密钥**，本仓库是公开的。项目方大约每 10 分钟看一次 Issue，并在 Issue 里回复；需要在你机器上执行的改动，仍然通过本仓库的 `tasks/queue.txt` 和脚本更新下发，你 `git pull` 即可。

---

## 1. 前置条件

| 项 | 要求 |
|---|---|
| GPU | B200 或 B300，单机 8 卡（一臂一卡，7 臂 + 1 张空闲） |
| 驱动 / CUDA | NVIDIA 驱动 ≥ 570，CUDA ≥ 12.8（`nvidia-smi` 右上角显示的 CUDA Version） |
| 系统 | Linux x86_64，Python 3.10 或 3.11 |
| 盘 | 每臂约 60 GB（见第 5 节），7 臂建议 ≥ 600 GB 可用；只保留上传后的 bf16 权重时约 300 GB |
| 网络 | 能访问 PyPI 与 GitHub；Hugging Face 或其镜像（下载模型与数据）；HF 或 ModelScope（上传结果） |
| 凭证 | 由项目负责人直接交给你：`HF_TOKEN`（只对结果仓库有写权限）+ `HF_REPO`，或 `MS_TOKEN` + `MS_REPO`。**只放在你自己的 shell 环境变量里，不要写进任何文件或提交到仓库。** 结果仓库默认可以是公开的；需要私有时设 `REQUIRE_PRIVATE=1`，脚本会拒绝往公开仓库上传。 |

网络受限时：
- 模型与数据下载：`scripts/download_models.sh` 默认就走 hf-mirror（只读镜像），10 秒连不上会自动回退到 huggingface.co；也可以从 ModelScope 下载模型（`USE_MODELSCOPE=1`，Qwen3 在魔搭上的 ID 与 HF 相同）。步骤见第 2a 节。
- **`HF_ENDPOINT` 只影响下载**：结果上传固定走 huggingface.co（`scripts/remote.py` 显式指定，v2.2 起），所以全局设了 hf-mirror 也不会把上传打到只读镜像。上传慢就改用 ModelScope：`export UPLOAD_BACKEND=modelscope MS_TOKEN=... MS_REPO=<你的命名空间>/<仓库名>`。

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

## 2a. 下载模型与数据（单独一步，v2.2）

装好环境以后，从更新仓库开始依次执行：

```bash
cd opd-blackwell-runbook && git pull                        # 更新到最新版（v2.2）
bash scripts/scan_cache.sh 2>&1 | tee scan_cache.log         # 只读：列出本机已有的 Qwen3 和数据集缓存，以及每份是否完整
export HF_HUB_CACHE=<scan_cache.sh 末尾建议的路径>           # 本机已有完整缓存时才设，用来复用
bash scripts/download_models.sh 2>&1 | tee download.log      # 默认走 hf-mirror，10 秒连不上自动回退 huggingface.co
```

`scan_cache.sh` 什么都不写。它打印各缓存变量（HF_HOME、HF_HUB_CACHE、TRANSFORMERS_CACHE、MODELSCOPE_CACHE），列出常用缓存目录里的 Qwen3、MATH-500、DAPO 条目，再在 /m2v_intern2、/data、/mnt、/home、/root、/workspace 下查找别的 `models--Qwen--Qwen3-*` 或魔搭的 `Qwen/Qwen3-*` 副本（要多查的目录可以作为参数传给它）。每份模型都会核对权重是否齐全：0.6B 应有 1 个权重文件，1.7B 2 个，8B 5 个。最后打印可以复用的 `export` 行。

`download_models.sh` 下载 0.6B、1.7B、8B 三个模型和 MATH-500、DAPO-Math-17k 两个数据集。它只在自己的进程里设 `HF_ENDPOINT`，不影响你的 shell；`HF_MIRROR=0` 可以不走镜像。结束时打印五行 `OK <名称> -> <本地路径> (<权重文件数>)`，五行都 OK 才接着生成训练与验证用的 parquet 文件（`code/train/prep_data.py`，v2.2 起从安装脚本挪到这里），然后打印 `DOWNLOAD PASS` 并以 0 退出。三个模型的本地路径会写进 `$OPD_ROOT/models.env`（`STUDENT_06B`、`STUDENT_17B`、`TEACHER`），`env.sh` 会读取它，所以之后所有脚本用的都是这三份本地副本，学生和教师都包括在内。

**备选：魔搭（ModelScope）**。镜像和主站都慢时用：

```bash
USE_MODELSCOPE=1 bash scripts/download_models.sh 2>&1 | tee download.log   # 三个模型从魔搭下载，两个数据集仍走 HF（镜像）
```

魔搭的缓存默认在 `~/.cache/modelscope/hub`，可以用 `MODELSCOPE_CACHE` 改位置。下载完路径同样写进 `models.env`，后面的步骤不用改。魔搭这条路线我们这边没有实测过，请确认 `download.log` 里五行都是 OK。

## 2b. B300（Blackwell Ultra，sm_103）

上面的栈（torch 2.8.0 cu128）是在 B200 上实测的。它的 `torch.cuda.get_arch_list()` 含 `sm_100`、`sm_120`，不含 `sm_103`。按 CUDA 的兼容规则，为 sm_100 编译的代码可以在同一大版本、更高小版本的卡上运行（10.0 → 10.3），所以这套栈**预计**能在 B300 上跑，但**我们没有在 B300 上验证过**。请按下面的顺序来：

1. 先回报机器信息（写进 `verify.log` 一并发回）：
   ```bash
   nvidia-smi --query-gpu=name,driver_version --format=csv; nvidia-smi | grep "CUDA Version"
   python -c "import torch;print(torch.__version__, torch.cuda.get_device_capability(), torch.cuda.get_arch_list())"
   ```
2. 用默认的 cu128 栈跑 `bash scripts/verify.sh`。四项都 PASS，就什么都不用改。
3. **Triton 的 ptxas（v2.3，换 CUDA 13 之前先做这一步）。** 判据：`verify.log`（或 `overnight.sh` 写的 `logs/vllm_probe_*.log`）里出现 `PTXASError` 或 `Internal Triton PTX codegen error`（通常紧跟着 `torch._inductor.exc.InductorError`）。原因：torch 2.8 自带的 Triton 用的是 CUDA 12.8 的 ptxas，编不了 sm_103 的核。按这个顺序处理：12.8 原样（第 2 步）→ CUDA 12.9 的 ptxas（下面这几行）→ CUDA 13（第 4 步）。
   ```bash
   source $OPD_ROOT/venv/bin/activate
   uv pip install nvidia-cuda-nvcc-cu12==12.9.86     # conda 环境用 pip install；v2.3 的 install_uv.sh 在 10.3 的卡上已经装好
   P=$(find $OPD_ROOT/venv/ -path '*cuda_nvcc/bin/ptxas' | head -1); $P --version   # 应显示 release 12.9, V12.9.86
   echo "export TRITON_PTXAS_PATH=$P" > $OPD_ROOT/triton.env; rm -rf ~/.triton/cache
   bash scripts/verify.sh 2>&1 | tee verify.log
   ```
   换了 ptxas 之后，如果日志里出现 `nvcc fatal : Unsupported gpu architecture 'compute_103'`（前后有 `Ninja build failed`、`flashinfer/103/cached_ops/sampling`），说明 Triton 这一关已经过了，卡在 FlashInfer：vLLM 默认用 FlashInfer 做 top-p/top-k 采样，FlashInfer 首次运行时用系统的 nvcc 即时编译，而 CUDA 12.9 之前的 nvcc 不认 sm_103。这时改用 vLLM 自带的 PyTorch 采样，注意力固定为 FlashAttention（vLLM 0.10.2 在 10.3 的卡上本来就默认用它，是预编译的，不需要 nvcc）：
   ```bash
   printf '%s\n' "export TRITON_PTXAS_PATH=$P" "export VLLM_ATTENTION_BACKEND=FLASH_ATTN" "export VLLM_USE_FLASHINFER_SAMPLER=0" > $OPD_ROOT/triton.env
   bash scripts/verify.sh 2>&1 | tee verify.log
   ```
   如果这时报 `no kernel image is available`，把第二行换成 `export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1`（vLLM 的 Triton 注意力，同样走上面的 ptxas），再跑一次。

   `env.sh` 会读 `$OPD_ROOT/triton.env`，所以之后的所有脚本（verify、心跳、重启后的 `run_tasks.sh`，以及训练里 verl 起的 vLLM，它继承环境变量）都带上这些设置；`verify.log` 第 2 项会打印当前用的 ptxas、注意力后端和采样器。删掉这个文件，就回到默认设置。我们在 H200 上核过：这个包里的 ptxas 是 V12.9.86，能把 `sm_103a` 的 PTX 编成机器码。整条链在 B300 上能否跑通，以重跑 verify 的结果为准。`scripts/overnight.sh` 会自动按这几档依次尝试（见本节末尾）。
4. 只有在 `verify.sh` 报 `no kernel image is available for execution on the device`、accept 里的 bf16 矩阵乘跑不起来，或换了 12.9 的 ptxas 仍报 `PTXASError` 时，才换 CUDA 13 路线（**未测试**）。版本已与各项目公开的发布核对过：torch 2.9.0 + torchvision 0.24.0 的 cu130 构建（vLLM v0.11.2 的构建依赖固定 `torch==2.9.0`），vLLM v0.11.2 从源码编译，flashinfer-python 0.5.2。原来的环境保留为 `$OPD_ROOT/venv-cu128`。

   **A. 机器上有 nvcc ≥ 13.0**（`nvcc --version`）：
   ```bash
   bash scripts/install_cu130.sh 2>&1 | tee install_cu130.log   # 新 venv → torch 2.9.0/torchvision 0.24.0 (cu130) → 检查架构 → 源码编 vLLM v0.11.2 → 其余依赖 → install_gopd.sh
   bash scripts/verify.sh 2>&1 | tee verify_cu130.log
   ```
   编译 vLLM 需要较长时间。可以用 `MAX_JOBS` 控制并行度，内存不够时调小。脚本在新 venv 里依次执行：
   `uv pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130` →
   检查 `torch.cuda.get_arch_list()` 含本卡架构（没有 `sm_103` 时，至少要有 `sm_100`，否则停下）→
   `git clone --branch v0.11.2 https://github.com/vllm-project/vllm.git` 后 `python use_existing_torch.py`、`uv pip install -r requirements/build.txt`、`uv pip install --no-build-isolation -e .` →
   其余依赖（含 `flashinfer-python==0.5.2`、`transformers>=4.56,<5`）→ `install_gopd.sh`。`install_gopd.sh` 只装 G-OPD、补丁、模型和数据，不会动 torch 和 vLLM。

   **B. 没有 nvcc ≥ 13.0**：用 NVIDIA 的 NGC 容器（已带 CUDA 13 的 torch 和 nvcc；进去后先用 `python -c "import torch;print(torch.__version__, torch.version.cuda)"` 确认）：
   ```bash
   docker run --gpus all --ipc=host --shm-size=64g -v $HOME/opd-run:/root/opd-run -e HF_TOKEN -e HF_REPO -it nvcr.io/nvidia/pytorch:25.10-py3
   # 容器内：
   git clone https://github.com/yang3121099/opd-blackwell-runbook.git && cd opd-blackwell-runbook
   SKIP_TORCH=1 bash scripts/install_cu130.sh 2>&1 | tee install_cu130.log   # 用容器自带的 torch，只编 vLLM 并装其余依赖
   bash scripts/verify.sh 2>&1 | tee verify_cu130.log
   ```
   - 训练框架仍是 G-OPD 分支（基于 verl 0.6.1），它与 vLLM 0.11 的接口我们**没有验证**。冒烟失败时，请把 `verify_cu130.log` 发回，由项目方修改 `code/patches` 的锚点。`apply_patches.py` 按代码锚点打补丁，并检查每个锚点恰好出现一次，对不上就报错退出，不会悄悄打错：
     1. `dp_actor.py` 里策略损失调用前的 `entropy_coeff` / `_forward_micro_batch` 片段；
     2. `ref_input_utils.py` 里 `prepare_ref_model_inputs` 的 `apply_chat_template` 调用；
     3. `fsdp_sft_trainer.py` 里的 `attn_implementation="flash_attention_2"`。
5. 换栈之后重跑 `verify.sh`，并把 `uv pip freeze` 的结果和 `verify.log` 一起发回，我们据此在本仓库补一份 B300 的版本锁。

**无人值守一次跑完（v2.3，`scripts/overnight.sh`）。** 从 `git pull` 一直跑到领任务：后台下载三个模型（hf-mirror 与魔搭双源）→ cu128 环境（没有 venv 才装）→ 数据 → `models.env` → vLLM 探测，按顺序尝试（A：cu128 原样；B：加 CUDA 12.9 的 ptxas；B2：再改用 PyTorch 采样、固定 FlashAttention；B3：再把注意力换成 Triton；C：机器上有 nvcc ≥ 13 时跑 `install_cu130.sh`。每次探测用各自的空编译缓存，通过的那一档写进 `$OPD_ROOT/triton.env`）→ `verify.sh` → 上传 `verify.log` → 心跳 → 等 1.7B 下载完 → `run_tasks.sh`。每一步在 `$OPD_ROOT/logs/overnight.log` 写一行 `STEP n ...`；失败时写 `FAIL` 并上传日志后停下，不会盲目重试。同一时间只能跑一份（`$OPD_ROOT/overnight.lock`），第二次启动会直接退出并说明原因。可以重跑：已下载的模型、数据和 venv 会直接复用，重跑会很快走到第 6 步的探测。
```bash
cd opd-blackwell-runbook && git pull
export OPD_ROOT=...  HF_TOKEN=...  HF_REPO=...       # 只在当前 shell 里设，不写进任何文件；OPD_ROOT 默认是 ~/opd-run
setsid nohup bash scripts/overnight.sh > /dev/null 2>&1 &
```

## 3. 验证（必须先做，日志发回）

```bash
bash scripts/verify.sh 2>&1 | tee verify.log
```

依次做四件事，每项都打印 `PASS` / `FAIL`：
1. **G2 单元测试**：`code/patches/test_opd_variants.py` 的 16 个测试（只用 CPU）。
2. **accept**：GPU 型号、驱动与 CUDA、bf16 矩阵乘吞吐、磁盘写速度、下载速度、vLLM 能否启动并生成 4 道 MATH500。
3. **3 步冒烟**：0.6B 学生 + 8B 教师的 P2 臂跑 3 步（关闭开训前验证，不存盘），记录每步耗时、显存峰值、`actor/pg_loss`、熵和回答长度。三步都有有限的 loss、没有 NaN/inf、显存在卡的容量以内，就算通过。

4. **上传路径（UPLOAD_CHECK）**：把 Qwen3-1.7B 按 verl 的方式存成 fp32 分片 ckpt（两个 `model-0000k-of-00002.safetensors` 加 index），用 `UPLOAD_DRY=1` 走一遍 `upload_sync.sh`（不需要凭据，不上传），再用 `scripts/check_upload.py` 对照 `reference/upload_manifest_qwen3-1.7b_bf16.json` 检查 bf16 产物：文件齐全、分片大小、311 个张量全是 bf16、index 的 total_size。

通过标志（`verify.log` 里逐行出现，都在行首）：

| 项 | 通过 | 未通过 |
|---|---|---|
| 1. G2 单元测试 | `G2 PASS` | `G2 FAIL` |
| 2. accept / vLLM | `VLLM PASS` | `VLLM FAIL` |
| 3. 3 步冒烟 | `SMOKE PASS (<秒数>s; 3 steps with a finite loss, no NaN)` | `SMOKE FAIL (...)` |
| 4. 上传路径 | `UPLOAD_CHECK PASS (<秒数>s)`（`check_upload.py` 返回 0） | `UPLOAD_CHECK FAIL`，后面的 `UPLOAD_CHECK_BAD ...` 行说明哪一项不对 |

全程约 1 小时。把 `verify.log` 发给项目负责人，四项都 PASS 之后再启动心跳和领任务。

**首臂比对（自动，不占额外机时）**：E10b-P2 跑到第 10 步后，心跳脚本会把它第 0–10 步的指标（第 0 步验证值、`actor/pg_loss`、熵、回答长度、截断率、奖励、单步时间），与我们在 H100 上同配置跑的参照（`reference/E10b-P2_reference_H100_first10.json`：同样的 1.7B 学生、8B 教师、lr 3e-6、4K、seed 42；第 0 步验证 74.0，单步 256–513 s）并列写进 `status/compare_<hostname>.json` 并上传一次（`scripts/compare_first_arm.py`）。预期：数据顺序相同（seed 42），所以每一步的题目相同；但 rollout 和验证都是采样（T=1.0 / 0.7），两边的注意力实现也不同（这里 SDPA，那边 flash-attn），所以数值接近、不会逐位相同。第 0 步验证值应在采样噪声内（100 题 × 2 条，约 ±7 分），第 1 步各项应在同一量级，之后允许分叉。

## 4. 启动心跳，然后领任务（之后自动运行）

**心跳**：每 10 分钟把本机状态写成 `status/<hostname>.json`，上传到结果仓库（HF 或 ModelScope）的 `status/` 目录。内容有队列版本、每个任务的状态（queued / running / done / retrying / failed）和当前步、最近一次验证值、截断率、失败次数和失败日志的最后几行、各 GPU 利用率、磁盘余量、时间戳。项目方只读这个目录，不会登录你的机器。
```bash
setsid nohup bash scripts/heartbeat.sh > $OPD_ROOT/logs/heartbeat.log 2>&1 &
```

**领任务**：
```bash
setsid nohup bash scripts/run_tasks.sh > $OPD_ROOT/logs/run_tasks.log 2>&1 &
DRY=1 bash scripts/run_tasks.sh      # 只打印"现在会在哪张卡上启动什么"，不启动
```
`run_tasks.sh` 每 5 分钟看一次空闲的卡（没有计算进程、显存 ≤ 2 GB），一卡一任务，严格按 `tasks/queue.txt` 的顺序领任务；每 30 分钟 `git pull` 一次本仓库。规则：

- **阶梯自动推进**：队列分阶段，状态是 `open`（现在可跑）、`auto`（前面所有阶段的任务都结束后自动开放）或 `locked`（不开放，也挡住后面所有阶段）。"结束"指 `.done`（本地或结果仓库里出现均可），或失败满 3 次。项目方可以随时 `git push` 修改顺序或把某阶段改成 `locked`；你这边不用做任何事。
- **依赖**：任务可以带依赖（`E12b-P1:SFT-P1` 表示等 `SFT-P1` 完成）。依赖永久失败时，这个任务会被跳过并在日志里注明。
- **白名单**：只有 `scripts/task.sh` 认得的任务 ID 会运行，其他内容一律忽略。改队列文件不会让你的机器执行任意命令。
- **快速失败**：每个任务脚本遇错即停（`set -e`）。任务退出时若没有 `.done`，就写 `runs/<任务>/.failed`（失败次数 + 日志末尾），由心跳上报，并自动重试；最多重试 2 次，第 3 次失败后不再重试。
- **不扩张**：队列文件以外的任务一律不跑（不加 seed、不扩到 4B 训练、不加新的方法臂），除非新版本的队列加入。
- **需要人工干预的只有一种情况**：调整策略时手动结束任务（`pkill -f "task.sh [0-9]* <任务>"`）。其余时间让它一直跑，把卡用满。

## 5. 任务阶梯（队列版本 2）

除注明外，配置与 0.6B 的 E10/E11/E12 完全一致，只把学生换成 Qwen3-1.7B。"状态"一列说明这一阶段的脚本验证到什么程度。

| 阶段 | 任务 | 做什么 | 单任务预计（B200，外推） | 状态 |
|---|---|---|---|---|
| E10b（open） | `E10b-{P2,P3,P4lowlr,P5,P6,P6r,P2noEOS}` + `E11b-P0` | 7 个准备臂，各 150 步（见下）；第 8 张卡评测基座 1.7B（TWR 的参照） | 8–14 h；E11b-P0 约 1 h | 训练路径在 B200 上冒烟通过（0.6B 学生 3 步） |
| E11b（auto） | `E11b-<臂>` ×7 | 每个臂第 150 步的 16K 重测：MATH500 avg@4，T=0.6 与 T=1.0（非思考）；再算 TWR/α（判别器 Qwen3-1.7B，参照为基座在 T=0.6 下的生成） | 约 1–2 h | 只做过语法检查；评测代码与 H100 上相同 |
| E12bA（auto） | `SFT-P1`、`E12b-{P0,P2,P5,P3,P1}` | tier A 的 GRPO，各 100 步。P1 = 1.7B 在教师正确轨迹上的 SFT（lr 1e-5，与 P2 的学生 token 数对齐为 49.2M，每 epoch 存一次）；若 `$OPD_ROOT/data/p1_traces/` 里没有教师轨迹，就先用 Qwen3-8B 生成（约 16.7K 题） | SFT 约 4–8 h（含生成轨迹）；GRPO 约 4–8 h | 只做过语法检查；GRPO 配置与 0.6B 在 H100/H200 上相同；SFT 在没有 flash-attn 时靠补丁 3 改用 SDPA，未实测 |
| zero（auto） | `E03-MI-l{0.0,0.3,0.5,0.7,1.0}`、`E02-opsa17-c{0.5,1.5}`、`E02-opsa4-c{-1,0.5,1.5}` | 零训练：Qwen3-4B Thinking-2507 与 Instruct-2507 的权重插值在 MATH500 上的 avg@4（思考模板，32K）；OPSA 的权重增量缩放 Base + c·ΔW 在 AIME24 上的 avg@16（非思考，32K） | 约 1–3 h | 只做过语法检查；权重运算与评测代码与 A100/H100 上相同 |
| E12bRest（auto） | `E12b-{P4lowlr,P6,P6r,P2noEOS}` | 其余起点的 GRPO，压缩版 60 步 | 约 3–5 h | 同 E12bA |
| E10b2（auto） | `E10b-{Pstop,P6rand,P6top5,P4EOSpen}` | 1.7B 的第二组臂：只保留终止附近的 advantage、随机 20%、只留 \|A\| 最大的 5%、把第一个终止标记强制加入 P4 的惩罚集合 | 8–12 h | 同 E10b |
| seeds（auto） | `E10b-P2-s43`、`E10b-P2-s44`、`E10b-P6r-s43`、`E10b-P6r-s44` | P2 与 P6r 的数据顺序种子复制（data.seed 43、44） | 8–12 h | 同 E10b |

**E10b 七臂**（1.7B 学生，Qwen3-8B 教师，DAPO-Math 训练题；每步 128 题、每题 1 条、回答上限 4K、150 步、data.seed 42、不加 KL、每 10 步在 MATH500 前 100 题上验证 avg@2）：

| 任务 ID | 臂 | 做什么（advantage A_t = log q_T − log p_θ 的变体） |
|---|---|---|
| `E10b-P2` | P2 | 标准 OPD（K1） |
| `E10b-P3` | P3 | 按学生 logp 分 10 档，只在同档内打乱教师 logp |
| `E10b-P4lowlr` | P4-lowlr | 学生 logp 最低的 20% token 固定 A=−0.5，lr 6e-7 |
| `E10b-P5` | P5 | 教师用思考模板打分 |
| `E10b-P6` | P6 | 只保留 \|A\| 最大的 20% token |
| `E10b-P6r` | P6r | P6 的补集（其余 80%） |
| `E10b-P2noEOS` | P2noEOS | P2，但最后一个 EOS 处的 advantage 置 0 |

**学习率 3e-6**（P4-lowlr 为 6e-7，与 0.6B 相同）。依据：0.6B 上按预注册规则从 {1e-6, 3e-6, 1e-5} 中选出 3e-6；1.7B 在 3e-6 下的 50 步试跑没有崩溃，验证值从第 0 步的 74.0 升到第 50 步的 79.5（+5.5，单次运行，4K）。

时长按 H100 实测外推：1.7B 学生 + 8B 教师在 H100 上每步 300–490 s，0.6B 为 110–120 s；B200 的 0.6B 冒烟为 102–105 s/步。**未实测**，以第一臂的实际步时为准。8 卡并行时，每个阶段的总时长约等于该阶段最慢的任务。

单独跑一个任务（不用队列）：
```bash
bash scripts/task.sh <gpu> <任务 ID>      # 例：bash scripts/task.sh 0 E10b-P2
```

**盘**：训练任务每 25 步（E10b）或 20 步（E12b）存一次 ckpt。1.7B 每份含 fp32 HF 权重约 6.8 GB，续跑状态约 20 GB，只在最新一份保留。上传的是 bf16 副本（约 3.4 GB）。设 `KEEP_LOCAL_HF=0` 时，本地 fp32 权重在 bf16 副本上传成功后删除。整个阶梯建议 ≥ 1 TB 可用，或打开 `KEEP_LOCAL_HF=0`。

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

## 6b. 远程控制与读数（v2.4）

**全部停下之后用 v2.4 重新起跑**（2026-10-17 B300：v2.3.1 每 25 步全量存盘会写满磁盘）：在 runbook 目录里，`OPD_ROOT`、`HF_TOKEN`、`HF_REPO` 已 export 的同一个 shell 中运行。它会拉取新版本，确认旧的调度、心跳、上传循环都已停掉，删掉没有任何检查点的 `runs/E10b-*` 半成品（这样失败计数也一并清零），然后启动 v2.4 的 `run_tasks.sh` 和心跳。不需要重跑 `overnight.sh`（`triton.env` 和模型都保留）：

```bash
git pull -q --ff-only && grep -q CKPT_ROOT scripts/env.sh || { echo "STOP: runbook is older than v2.4 after git pull"; false; } && { pkill -f "scripts/run_tasks.s[h]"; pkill -f "scripts/heartbeat.s[h]"; pkill -f "scripts/upload_sync.s[h]"; sleep 2; echo "GPU processes left: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; CK=$(bash -c 'source scripts/env.sh > /dev/null 2>&1; echo $CKPT_ROOT'); echo "CKPT_ROOT=$CK ($(df -h $CK | awk 'NR==2{print $4}') free)"; [ -n "$OPD_ROOT" ] && [ -d "$OPD_ROOT/runs" ] && rm -rf "$OPD_ROOT"/runs/E10b-*; [ -n "$CK" ] && rm -rf "$CK"/E10b-*; setsid nohup bash scripts/run_tasks.sh >> $OPD_ROOT/logs/run_tasks.log 2>&1 < /dev/null & sleep 2; setsid nohup bash scripts/heartbeat.sh >> $OPD_ROOT/logs/heartbeat.log 2>&1 < /dev/null & sleep 5; tail -3 $OPD_ROOT/logs/run_tasks.log; python scripts/control.py node; }
```

它也会删掉 `$OPD_ROOT/runs/E10b-*` 里 v2.3.1 留下的检查点残留，以及 `$CKPT_ROOT` 下的 `E10b-*`。如果 `GPU processes left` 不是 0，先把 `nvidia-smi` 的输出贴到 Issue 里；仍被占着的卡，新的调度不会去用。

**你这边只需做一次**：`git pull` 之后重启一次 `run_tasks.sh`（旧版本不会自己重启）。之后它每 60 秒 `git pull` 一次（仓库没变时只是一次很小的请求）；如果拉下来的 `run_tasks.sh` 或 `env.sh` 有变化、并且新版本能通过 `bash -n` 检查，它会自己重启（`exec`）。正在跑的任务是独立的进程，不受影响。

**项目方怎么下指令**：在本仓库的 `control/<节点号>.yaml` 里追加带 `seq` 的条目。节点号由 `python scripts/control.py node` 打印，默认是主机名经 sha256 后的前 8 位，所以公开仓库里不会出现你的主机名；心跳里也会报这个节点号。每条指令只执行一次，严格按 `seq` 顺序，结果写在 `$OPD_ROOT/control/log.jsonl`，并随心跳上传。执行之前会核对 `git remote origin` 确实是本仓库，否则一条都不执行。只有下面这些动作，**没有任意 shell**：参数按空格拆分，含 `; & | $ \` ( )` 等字符的整条拒绝。

| 动作 | 作用 |
|---|---|
| `pause <任务>` / `resume <任务>` | 不再启动 / 恢复启动这个任务（正在跑的不受影响） |
| `kill <任务>` | 停掉这个任务（先 SIGTERM，30 秒后 SIGKILL），连同它那张卡上残留的 Ray 进程，并核对该卡空出来了；同时 pause，这次退出不算失败。可续跑的任务 resume 后从最新 ckpt 继续 |
| `prio <任务> <n>` | 同一阶段内先启动 n 小的（默认 100），不越过阶段和依赖 |
| `restore <任务> runs/<任务>/global_step_<N>` | 从结果仓库下载这一步的检查点，逐个文件核对大小和 sha256，全部一致才放进 `runs/<任务>/ckpt/`；任务正在跑时拒绝 |
| `rerun_verify` | 重跑 `verify.sh` 并上传日志；有任务在跑时拒绝 |
| `setenv KEY=VAL` | 写进 `$OPD_ROOT/control.env`（`env.sh` 最后读它），只影响之后启动的任务；只允许 `VLLM_*`、`TRITON_*`、`TORCHINDUCTOR_*`、`NCCL_*` 和几个本项目的开关，`PATH`、`LD_*`、各种 token 一律拒绝 |
| `snapshot <路径>` / `snapshot nvidia-smi` | 把 `$OPD_ROOT` 下的一个文件或目录（只传小文本文件：单个 ≤ 5 MB，共 ≤ 20 MB，最多 50 个；不传权重和数据，不传名字像 token/secret 的文件，解析符号链接后不能跑出 `$OPD_ROOT`）上传到结果仓库的 `logs/<节点号>/requests/<seq>/`，文本里的 token 形态一律替换成 `***` |

心跳（`status/<主机名>.json`）新增：每张卡的利用率和显存；每个任务的步数、总步数、最近 5 步平均步时、预计剩余时间、日志末 20 行（已脱敏）；已执行到的 `seq` 和最近 5 条结果；当前的 `control.env`；最近一次 `git pull` 的结果；runbook 的提交号。如果某次 `setenv` 之后 10 分钟内有任务失败，且日志里有 vLLM/Triton 初始化错误，心跳会标出 `control_env_suspect`，只报告，不自动回滚。

## 7. 常见错误

| 现象 | 原因与处理 |
|---|---|
| `nvidia-smi` 的 CUDA Version < 12.8 | 驱动太旧，需要 ≥ 570。 |
| `CUDA error: no kernel image is available for execution on the device` | 某个包的 wheel 没编进 sm_100。先确认 `python -c "import torch;print(torch.cuda.get_arch_list())"` 里有 `sm_100`；若是 flash-attn，卸载它（脚本会自动改用 SDPA）。 |
| flashinfer 首次运行很慢或编译失败 | flashinfer 0.3.0 只有源码包，首次使用 JIT 编译需要几分钟和 `nvcc`。可改用 vLLM 自带的注意力后端：`export VLLM_ATTENTION_BACKEND=FLASH_ATTN`，或者用 Docker 路线。 |
| vLLM 初始化时 KV cache 不够 | 学生、教师和 vLLM 在同一张卡上。调低 `VLLM_UTIL`（默认 0.5）；B200 显存够时，可设 `TEACHER_OFFLOAD=0` 让教师常驻显存，速度更快。 |
| Ray 报 `pids` 或线程数超限 | 已默认 `RAY_prestart_worker_first_driver=0`、`RAY_num_workers_soft_limit=8`；同机多臂时让启动错开（`run_tasks.sh` 默认间隔 90 s）。 |
| 下载模型或数据失败 | 先跑 `scripts/scan_cache.sh` 看本机有没有现成的缓存；`download_models.sh` 默认走 hf-mirror，失败时用 `USE_MODELSCOPE=1` 从魔搭下载模型（第 2a 节）。 |
| `PTXASError` 或 `Internal Triton PTX codegen error`（B300） | Triton 自带的 CUDA 12.8 ptxas 编不了 sm_103。按第 2b 节第 3 步换 CUDA 12.9 的 ptxas（写 `$OPD_ROOT/triton.env`）；仍失败再走 CUDA 13 路线。 |
| `nvcc fatal : Unsupported gpu architecture 'compute_103'`（`flashinfer/103/cached_ops`、`Ninja build failed`） | FlashInfer 的采样核要用系统 nvcc 即时编译，CUDA 12.9 之前的 nvcc 不认 sm_103。在 `triton.env` 里加 `VLLM_USE_FLASHINFER_SAMPLER=0` 和 `VLLM_ATTENTION_BACKEND=FLASH_ATTN`（第 2b 节第 3 步）。 |
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

## 9. 变更记录

- **v2.1（2026-10-16）**：修复上传脚本 `scripts/upload_sync.sh` 的两个问题。一是只认单文件 `model.safetensors`：1.7B 的 ckpt 是分片保存的，修复前会被一律跳过，而且没有任何报错。二是一看到 `finished:` 就退出：最后一个 ckpt 往往还不满 120 秒，修复前会漏传。现在分片逐个转 bf16，index 的 total_size 改成 bf16 大小（`to_bf16.py`）；所有 ckpt 都传完才退出；`verify.sh` 新增第 4 项上传路径检查（`check_upload.py` + `reference/upload_manifest_qwen3-1.7b_bf16.json`）。**v2 或更早的节点**：请更新到 v2.1，重启 `upload_sync.sh`（已传的 ckpt 有 `.uploaded` 标记，不会重传；被跳过的分片 ckpt 会补传），然后重跑 `verify.sh` 的第 4 项。
- **v2.2（2026-10-16）**：下载改成单独一步（第 2a 节）：新增只读的 `scripts/scan_cache.sh`（查本机缓存，核对每个模型的权重是否齐全）；`download_models.sh` 默认走 hf-mirror，10 秒连不上自动回退，结束时打印每项的本地路径和权重文件数，五行 OK 才以 0 退出；模型的本地路径写进 `$OPD_ROOT/models.env`，由 `env.sh` 读取，`run_arm/run_sft/run_e12/run_eval` 的教师和学生都从 `TEACHER`、`STUDENT_17B`、`STUDENT_06B` 取，魔搭下载的模型因此也能直接用（修复前教师在 `run_arm.sh` 里写死成 `Qwen/Qwen3-8B`）；`install_gopd.sh` 不再顺带下载。另外，`scripts/remote.py` 对结果仓库固定使用 `https://huggingface.co`（可用 `HF_UPLOAD_ENDPOINT` 覆盖）。修复前，为了加速下载全局设置 `HF_ENDPOINT=https://hf-mirror.com` 后，令牌检查和所有上传都会打到只读镜像而失败。下载仍按 `HF_ENDPOINT` 走镜像；`hf-transfer` 在各安装路线和 `requirements.lock` 里都已安装（0.1.9）。
- **v2.3（2026-10-17）**：代码审阅后的修复。① `run_e12.sh`、`run_eval.sh`、`run_sft.sh` 判断检查点是否存在时只认单文件 `model.safetensors`，而 1.7B 的检查点都是分片的：修复前，E10b 之后的 E11b、E12b、SFT-P1 会全部失败（SFT-P1 要在训练完以后才失败，并重训三次）。现在统一用 `env.sh` 里的 `has_weights`，认分片。② `run_tasks.sh`：依赖已失败三次的任务记为 `runs/<task>/.skipped`，算作完成，不再让所在阶段永远结束不了、卡住后面所有阶段；在结果仓库里查到的 `.done` 缓存到本地，不再每 5 分钟对每个任务问一次。③ `run_eval.sh`、`run_zero.sh` 先上传，上传成功才写 `.done`；修复前，上传失败的结果会被当作已完成而只留在本机。`run_zero.sh` 重试时跳过已完成的生成。④ `run_arm.sh`、`run_e12.sh` 只有跑满步数才写 `finished:`，提前退出写 `stopped:`，`upload_sync.sh` 因此会继续为重试上传检查点。⑤ 心跳：验证分数改为读任意 `mean@n`（E12b 是 mean@4，修复前一直为空），并报告各阶段的状态和新增的 `skipped` 任务状态。注意：`run_tasks.sh` 是常驻进程，`git pull` 之后要重启它，②才生效；正在跑的任务不受影响，新起的任务自动用新脚本。⑥ B300 的 Triton ptxas：判据和三档顺序见第 2b 节第 3 步；`env.sh` 读 `$OPD_ROOT/triton.env`；`install_uv.sh`、`install_conda.sh` 在 compute capability 10.3 的卡上预装 `nvidia-cuda-nvcc-cu12==12.9.86`，但不自动启用；`verify.sh` 打印当前用的 ptxas。⑦ 新增 `scripts/overnight.sh`（由项目审阅方编写），无人值守地从 `git pull` 跑到领任务，vLLM 按三档依次探测。
- **v2.3.1（2026-10-17）**：B300 第一晚的结果。cu128 原样跑会报 `ptxas fatal : Value 'sm_103a' is not defined`；换成 12.9 的 ptxas 之后，Triton 编译通过，又卡在 FlashInfer 的即时编译采样核：系统 nvcc 早于 12.9，报 `Unsupported gpu architecture 'compute_103'`。`overnight.sh` 因此新增 B2（PyTorch 采样 + FlashAttention）和 B3（Triton 注意力）两档；通过那一档的全部变量写进 `$OPD_ROOT/triton.env`，`env.sh` 会读它。每次探测用各自的空编译缓存，避免前一档或前一晚编好的核让后一档在没有修复的情况下误判通过。同一时间只允许一份 `overnight.sh`（`$OPD_ROOT/overnight.lock`）。STEP 6 的报错改为完整的行（之前的正则把 `Command` 截成了 `Comma`）。重跑时不会再起第二份心跳或 `run_tasks.sh`。`verify.sh` 打印注意力后端和采样器设置。第 2b 节第 3 步和第 7 节补充了相应的判据。
- **v2.4（2026-10-17）**：**存盘规则（Runming）**：专机不存续跑态，只在预注册的读数步存 HF 权重（E10b 只在第 150 步，即 E11b 和 E12b 读的那一份；E12b 每 20 步），`upload_sync.sh` 转成 bf16 上传，本地删掉 fp32，只留 bf16 给本机评测（E12b 不留）。E11b、E12b 都读这份 bf16，E12b 读完、E11b 也完成后删掉。1.7B 每次存盘只需临时约 11 GB，原来是 28 GB，且每 25 步一份全留。`SPOT=1` 的机器另存续跑态，只留最新一份。心跳增加 `disk_alert`（剩余 < 30 GB）。**检查点放在大盘**：`CKPT_ROOT` 在第一次使用时选定，取 `/home` 或 `$HOME` 下所在文件系统剩余空间最大的可写目录，再加 `/opd-ckpt`，写进 `$OPD_ROOT/paths.env`；shell 里已设的值优先。`runs/<任务>/ckpt` 是指向 `$CKPT_ROOT/<任务>/ckpt` 的软链，其余脚本不变；心跳另报这块盘的剩余（`ckpt_disk_free_gb`、`ckpt_disk_alert`，< 100 GB）。`UPLOAD_WEIGHTS`：哪几步的权重传到结果仓库，默认只传最后一步（E10b 是 s150）；其余步不做 bf16、删掉 fp32，标记为 `.not_uploaded`。E12b 默认也只在最后一步存权重。新单测：`scripts/test_upload_sync.sh`、`scripts/test_ckpt_root.sh`、`scripts/test_done_throttle.sh`（Linux）。**E12b 从 bf16 权重起步**（加载时升为 fp32），与实验室 0.6B 的 fp32 起点相比只差权重舍入，已记为偏离。`run_tasks.sh` 每 60 秒一轮，结果仓库里"是否已完成"的查询每个任务最多 10 分钟一次。远程控制与读数（第 6b 节）。新增 `scripts/control.py`（白名单动作、按 seq 只执行一次、核对仓库来源、统一脱敏）及其单测 `scripts/test_control.py`；`run_tasks.sh` 每 5 分钟拉取一次，支持 pause/prio、`kill` 不计入失败、远程 verify 期间不启动新任务，自身有变化时自动重启（新版本先过 `bash -n`）；`heartbeat.sh` 增加每卡、每任务和控制面的读数；`env.sh` 读 `$OPD_ROOT/control.env`。新增 G2 测试 `code/patches/test_variant_dispatch.py`：`task.sh` 白名单里的每个臂都必须真的走到对应的优势计算，否则测试失败（防止新臂因为补丁名单漏项而静默地按普通 OPD 训练）。**升级时要手动重启一次 `run_tasks.sh`**，之后就不用了。
