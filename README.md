# OPD 远程节点手册（v3.11，GitHub 模式）

这个仓库让一台别人的 8 卡机器变成本项目的**远程节点**：机器旁的人只粘贴一行命令、输入两个 token，之后的安装、检查、修复、跑任务都由项目方远程完成。项目方不登录机器；节点对本仓库只读（`git pull`），可选的第二个 token 只用来向私有的心跳仓库推送。

- 机器旁的人看 **[COLDSTART.md](COLDSTART.md)**（那一行命令和它会做什么）。
- 本页给项目方：控制通道怎么工作、新节点上线的指令顺序、队列格式、加一条新臂要改哪里、怎么测。
- v2 时期的长文档（三种手工安装路线、10-16 的七臂阶梯、逐版变更记录）在标签 `record-z172` 里。脚本注释里的 "README section N" 指那一版。

## 1. 工作方式

```
项目方 (Mac)                         GitHub / HF 镜像                     节点
ctl.py <节点> <动作> --reason …  ──►  control/<节点>.yaml  ──(每 60 s git pull)──►  control.py apply
tasks/queue_<名>.txt             ──►  main                 ──────────────────────►  run_tasks.sh → task.sh <卡> <任务>
读 status / pulse / 回执          ◄──  HF 结果仓库（公开：权重、读数、脉搏；私有：状态、日志、回执）  ◄──  heartbeat.sh / uploader.sh
                                 ◄──  私有 GitHub 心跳仓库（HF 不通时的后备）                           ◄──  gh_heartbeat.sh
```

- **下行只有 git**：指令写在 `control/<节点号>.yaml`，带 `seq`，每条只执行一次、按序、默认 2 h 或 24 h 过期；每条都有回执（拒绝的也有）。
- **没有任意 shell**：动作是白名单（`scripts/control_rules.py`，节点与项目方共用），参数含 shell 元字符整条拒绝。规则文件每次改动必须在 `control/CHANGES.md` 留一行 `rules <版本>`，否则节点退回上一提交并上报 `rules_refused`。
- **任务也是白名单**：只有 `scripts/task.sh` 认得的任务名会运行；改队列不能让节点执行别的东西。
- **节点号**是主机名 sha256 的前 8 位，公开仓库里不出现主机名、路径、用户名。
- **一个节点一份队列（v3.0）**：`tasks/queue.txt` 不开放任何阶段。节点收到 `queue <名>` 后才读 `tasks/queue_<名>.txt`。没被分配队列的节点（复活的旧机器、多克隆的一份）什么都不会启动。
- 时延：一条指令从推送到回执约 1–1.5 分钟。

## 2. 新节点上线（T0 的指令顺序）

机器旁的人粘贴 COLDSTART.md 的那一行后，节点打印 `node <id> online`，此时**调度是冻结的**。之后全部远程：

| 顺序 | 指令 | 作用 |
|---|---|---|
| 1 | `env-probe` | 卡、驱动、磁盘、网络，写进状态；同时扫描这台机器的磁盘上有没有我们以前的安装（v3.7，见下） |
| 2 | `queue <名>` | 指定这台节点的队列文件（之后的预取、清单都按它来） |
| 3 | `setenv K=V` | 与档位（`profiles/*.env`）不同的设置 |
| 4 | `setup all` | 后台装环境、下基座模型、探测 vLLM、跑 verify；每步进状态，失败给出日志末尾 |
| 5 | `run FETCH-all` | 冻结期间预取队列 `#fetch` 行的权重（不占卡，32 连接） |
| 6 | `run PROBE-TAIL` | 金丝雀：主环境装好后在本机跑一次 3 步冒烟 |
| 7 | `dispatch on` | 只有主环境可用且金丝雀在本机通过才接受；之后按队列占满空卡 |

**借用机器（v3.8–v3.10）另加三步**，因为它们会在没有真实训练时被收回，而且外网很慢：

| 何时 | 指令 | 作用 |
|---|---|---|
| 第 2 步之后立刻 | `setenv SFT_WARM_GPUS=1,2,…,7` 然后 `run SFT-WARM` | 基座模型的全参数 SFT，每张卡一个进程；不等主环境（用历史扫描找到的旧环境的包和模型），冻结时也启动。0 号卡留给安装自检和金丝雀（全参数训练每卡约 130 GB）。真实训练，但不是登记实验。换卡：改 `SFT_WARM_GPUS` 后 `kill` + `run`；正式臂要卡时 `kill SFT-WARM` |
| 权重预取完成后 | `store` | 把本机完整的基座模型和权重存进共享盘上的长期仓库（`scripts/store_sync.sh`，只增不删） |
| `setup all` 通过后 | `store` | 再存一次：环境包和 flash-attn 安装包。下一台机器的 `setup` 先从仓库 / 旧环境复制到本地，再补缺的包 |

**历史目录扫描（v3.7，`scripts/history_scan.py`）**：每次 `env-probe` 和每次安装开始时，节点自己找这台机器磁盘上我们以前的安装目录，尽量复用，少走网络。线索来自本机：GPU 上进程的工作目录和脚本路径（机器主人的填充程序通常就在他的代码目录里）、存下的填充命令、`$HOME`，以及同一相对目录在其他共享盘上的位置；在线索下面按层搜索（4 层、6000 个目录、60 秒为上限）。结果只写在节点本地（`control/history_secret.json` / `.env`，权限 600），路径不进任何控制文件；状态里 `env_probe.history` 只给个数、模型名和有没有长期仓库，不含路径。用法：基座模型完整就**拷贝**到本地盘（不依赖共享盘运行）；我们自己的权重由 `fetch_model.py` 先从旧目录拷贝、仍按 Hub 的清单核对大小和 sha256；数据文件和旧环境只报告、不自动沿用。旧目录里的查看和清理仍用 `du` / `rm-ours`。

**借用机器上的复用与容灾（v3.11）**。前提：机器可能两小时就被收回，本地盘随之丢失；共享盘（固定盘）上有我们以前的安装。
- **环境**：先限时拷到本地盘（`SEED_COPY_MAX_S`，默认 300 秒；先拷到临时目录，完成后才移入，不会留半成品）。超时就改为"就地使用"：本地环境里放一行 `.pth` 指向共享盘上的包目录。之后只向网络要缺的包。
- **基座模型**：不下载。依次找：本机 `models/`、长期仓库和旧安装（限时拷贝 `MODEL_COPY_MAX_S` 600 秒，超时就地使用）、共享盘上别处的完整副本（别人的 pretrained 目录、Hugging Face 缓存）。最后一类必须先通过 `scripts/model_verify.py`：每个权重文件的 sha256 与官方发布的一致才用，同名目录里可能是别的权重。都没有才下载。
- **我们自己的权重**：从 Hub 下到本机（`run FETCH-all`），之后由仓库同步存到固定盘。
- **容灾**：长期仓库同时是 DR 根目录（`DR_ROOT`）。每条在跑的训练每 `DR_EVERY` 步（`profiles/h200.env` 设为 5，能整除登记步 25 / 50）存一次续训状态，`dr_sync.sh` 镜像到 `<仓库>/dr/<任务>/`；下一台机器启动同名任务时自动拷回接着训（`dr_restore`）。
- **固定盘空间有限时的取舍**（`scripts/store_sync.sh`，每 20 分钟自动跑一次，也可发 `store`）：续训状态优先，仓库始终为它留 `STORE_RESERVE_GB`（100）；其余按"重新获得的代价"排序——flash-attn 安装包、环境包（环境只是就地使用时不存）、我们的权重、固定盘上还没有的基座模型。只增不删。

修复用 `tail <日志>`、`setup <步号>`、`restart-loops`、`verify`。目标：5 分钟内有状态，10 分钟内开始下权重，2.5 小时内第一条臂起训。

## 3. 队列格式（`tasks/queue_<名>.txt`）

```
version: 2
<阶段> open|auto|locked  <任务>[:<依赖>[+<依赖>…]] …     # open 现在可跑；auto 前面阶段都结束后开放；locked 不开放并挡住后面
#gpus <任务> <n>            多卡任务的卡数（唯一来源；队列没有这一行时看 tasks/board_arms.txt）
#fetch <来源>               开跑前要在本机备好的权重 / 数据（result:runs/…、Qwen/…、dataset:…）
#eval <任务> <tag> <模型> <基准> <n> <温度> [上限]      一次读数；#pair / #budget / #forced / #term 是它的配对、预算曲线、强制停、终止探针
#needs verify <任务>        verify 四项通过后才启动
#steptime <任务> <秒>       多卡臂的预期步时
```

- 档位文件由 `scripts/queue_profiles.py` 生成（改生成器，不改生成物；`--check` 核对）。行的来源是 `tasks/rows_*.txt`，它们不在 `queue_` 命名下，不能被分配给节点。
- 新增队列行的提交要写"部署说明"：新任务名、每台节点保留哪些；节点下一次 pull 就会按新行调度。
- 起训前先过预检：在节点上 `bash scripts/preflight.sh <卡> <任务>`（卡数、门、输入、存盘设置、续跑），或全量 `python3 scripts/preflight_all.py`。

## 3b. GPU 利用率规则（v3.1，借用机器）

机器方的规则：每张卡利用率不低于 85%（平时约 90%），否则两小时后作业被结束。节点的做法分两半：

- **空闲卡跑机器主人自己的填充程序**（`scripts/gpu_fill.sh`）。命令只存在节点本地（`bootstrap.sh --fill-cmd '<命令>'` 写进 `control/gpu_fill_secret.cmd`，权限 600，不进本仓库、不进状态、不上传）。开关：`setenv GPU_FILL=0|1`（`profiles/h200.env` 默认开）。
  - 它是一个独立的循环，每 20 秒看一次：一张卡连续两次为空（没有任务占着、没有进程、显存 ≤ 2 GB）就起一个填充实例（`CUDA_VISIBLE_DEVICES=<卡>`，独立会话）。循环的状态放在内存盘 `/dev/shm`，不依赖调度循环，也不依赖磁盘：任务因为盘满退出、调度器写不了文件时，卡照样被填上。
  - 只跑着填充程序的卡对队列来说是空闲的。无论是我们起的，还是机器主人事先起的（按命令行里的脚本名认）。
  - 任务启动前，先把它要用的卡上的填充程序清掉（先 TERM，20 秒后 KILL -9），等显存释放再起。只清填充程序、只清这几张卡；不用 `fuser -k /dev/nvidia*`，那会把我们在别的卡上正在跑的任务一起杀掉。
  - 安装阶段不清任何卡（v3.3）。机器主人的填充程序常常是"一个进程占全部卡"：结束它之前，节点先用学到的命令在一张卡上并排试跑 30 秒（v3.6，`control/gpu_fill_selftest`），活着才算"能重启"；没证明之前只结束我们自己起的实例，任务与主人的填充程序共用那张卡。证明之后，任务要的卡被清空的同时，其余卡立刻各起一个实例。
  - 命令从正在跑的填充进程学来：解释器用真实路径，带上它自己的 `PATH` / `LD_LIBRARY_PATH` / conda 变量（v3.6）。状态里 `gpu_util.gpus[i].apps` 给出每张卡上进程的程序名和脚本名（只有文件名），`gpu_util.fill` 给出是否开启、命令是否已知、试跑结果。
- **节点自己记账**（`scripts/gpu_monitor.py`）。每 10 秒采样，每张卡给出 1 分钟、10 分钟、30 分钟、2 小时均值，过去 2 小时达标时间占比，以及卡上是任务、填充程序还是空的。写进状态的 `gpu_util`；30 分钟均值低于线发事件 `gpu_util_warn`（比规则生效早 90 分钟），2 小时均值低于线发 `gpu_util_low`。线用 `setenv GPU_UTIL_FLOOR=<50–99>` 改，默认 85。

任务运行期间不做补足：真实任务自己的利用率是多少，由上面的记账给出。

## 4. 控制动作

| 类别 | 动作 |
|---|---|
| 调度 | `queue` `dispatch on\|off` `pause` `resume` `prio` `run` `rerun` `kill` `cancel` |
| 查看 | `ps` `logs` `tail` `snapshot` `du` `mounts` `inventory` `env-probe` `heartbeat` |
| 环境 | `set` / `setenv`（白名单键）`setup` `verify` `rerun_verify` `fa-build` `update` `restart-loops` |
| 进程 | `signal <pid> TERM\|KILL\|STOP\|CONT` `kill-pgid` `reap`（pid 必须在最近发布的进程表里且启动时间、命令行哈希一致） |
| 数据 | `eval` `upload` `restore` `publish` `republish` `publish-redacted` `pend-weight` `move-ckpt` `rm-ours` |
| 容灾（节点在共享盘集群上时） | `dr-keep` `persist-env` `rebuild-drill`（见 REBUILD.md）`leakscan` `scrubdiff` |

绝不碰：其他用户的进程（哪怕占着我们的卡）、pid 1 / sshd / 容器服务、登录 shell / tmux / jupyter、`$OPD_ROOT` 与 `$CKPT_ROOT` 之外的文件、名字像 token 的文件、PATH / LD_* / 代理 / 各种 token。

## 5. 加一条新臂要改哪里

当前每条臂是手写的分支，一条新臂涉及：`task.sh`（白名单）、`run_arm.sh` 或 `run_e12.sh`（配方、门、上传步）、`probe_*.sh`（门）、`control_rules.py` 与 `control.py`（`run` / `rerun` 认名字，需升规则版本并在 CHANGES.md 签字）、`tasks/board_arms.txt` 或队列的 `#gpus` 行、队列行、测试。`scripts/test_preflight_enum.py` 会让每条多卡臂和板上的臂在 CPU 上走一遍真实的 `task.sh` 直到发射点；漏了任何一处都会在这里失败，而不是在机器上。

## 6. 测试

```bash
python3 -m pytest -q scripts code --continue-on-collection-errors      # 约 6 分钟
for t in scripts/test_*.sh; do bash $t > /dev/null 2>&1 || echo "FAIL $t"; done   # 约 14 分钟
```

没有 torch 的机器（Mac）上，`code/patches/test_opd_variants.py`、`test_variant_dispatch.py`、`code/eval/test_bench_v2.py`、`scripts/test_probe_wdiff.sh` 跑不了，要在节点上跑。推送前看退出码，不要 `pytest | tail && git push`。

## 7. 仓库结构

| 路径 | 内容 |
|---|---|
| `COLDSTART.md` `scripts/bootstrap.sh` `cold_start.sh` `overnight.sh` | 上线：一行命令 → 控制通道 → 远程安装 |
| `scripts/run_tasks.sh` `heartbeat.sh` `loops.sh` `watchdog.sh` `uploader.sh` | 节点的常驻循环 |
| `scripts/control.py` `control_rules.py` `remote.py` | 控制面与上传 |
| `scripts/task.sh` `run_arm.sh` `run_e12.sh` `run_queue_eval.sh` `probe_*.sh` | 任务白名单、训练 / 读数 / 门 |
| `scripts/dryrun_json.py` `preflight*.py/sh` `deps.py` | 不占卡的预检与行为对照 |
| `code/` | 训练补丁、评测、分析 |
| `control/` `tasks/` `profiles/` | 每节点的指令文件、队列、机型档位 |
