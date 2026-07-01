# 在 NUS Vanda（A40）上跑项目三 —— 主路径

**Vanda HPC 是项目三的唯一主路径**（和项目二同一套：PBS + Singularity + 用户态隔离包）。
Colab 仅作 stage A 的兜底（见文末）。

> **状态（按项目记忆）**：项目二的 **MACE baseline 已在 Vanda A40 实跑过**（50ps × 600/800/1000K，
> σ300≈29.6 mS/cm、Ea≈0.198 eV，过预测实验 ~9.4×=未微调预期内）。所以 stage B（MACE 在 A40 上弛豫、
> 复用 `~/macepkg`）走的是**已验证路径**。**新的只有 stage A 的 MatterGen**——它在 Vanda 上从未跑过，
> 是首次在 Vanda 运行时的检验点（见文末风险）。未经脚本佐证的具体数字（配额 GB、scratch 路径等）不写死，以 Vanda 实际为准。

## 环境事实（已核实）

- **GPU = NVIDIA A40**（Ampere, compute capability 8.6, 48GB VRAM）。
- 镜像 `pytorch_2.5_cuda_12.4_unsloth.sif`（py3.10 + torch2.5 + **CUDA 12.4**）→ A40 零兼容问题（非 Blackwell，不需 CUDA≥12.8）。
- MLIP/生成都走 **FP32**；A40 单精度强、FP64 弱（本工作流不碰 FP64）。容器里只有 `python3`（无 `python`）。
- ⚠️ **torch 的 CUDA build 要 ≤ A40 驱动（12.4）**：12.4 驱动能向后跑任何**更旧**的 toolkit（cu118/cu121/cu124 都行），
  但跑不动**更新**的（如 `cu130`）——那才会在 A40 上**静默回落 CPU**（已知坑）。MatterGen 钉的是 `torch2.2.1+cu118`，
  在 12.4 驱动上没问题；`setup_mattergen.sh` 的断言只拦「比驱动新」或 `cpu` 版，看到 `TORCH_CUDA_OK` 即过。
  **别手动改 venv 里 torch 的 cu 版**——它的编译扩展（`torch_scatter` 等）按 cu118 编译，换了 torch 的 cu 版会找不到 `libcudart` 而崩。stage B 复用 macepkg 的容器 torch，不受影响。

两阶段、两套包（同镜像、互不污染）：
- **A 生成**：MatterGen 独立 uv venv `~/mattergen/.venv`（它自己解析的 torch，不碰容器的 2.5）。
- **B 筛选+打分**：**复用项目二已装的 `~/macepkg`**（mace+ase），只把 catboost+pymatgen+mp-api 叠进 `~/p3pkg`。

## 0) 前置：代码上 Vanda（并排放）

`score.py` 向上找 `../../project1_screening/`，所以项目一、三必须并排：

```
~/AI4SSB/project1_screening/     # 含 catboost_model.cbm + src/featurize.py + 04_screen_mp.py
~/AI4SSB/project3_generative/     # 本项目
```

`data/ref_structures.json` 一并带上 → 建凸包直接用缓存的 96 个 MP 参考相，**连 MP_API_KEY 都不用配**
（这是可移植的结构缓存，跨 pymatgen 版本都能读；不要用本机 pickle 的 `ref_entries.pkl`，新版 pymatgen 解不开）。

## 1) 一次性装环境（登录节点，有网）

> 计算节点离线，所以装包 + 预下权重都必须在**登录节点**做（它有外网）。

```bash
module load singularity
IMG=/app1/common/singularity-img/vanda/pytorch_2.5_cuda_12.4_unsloth.sif

# A 阶段：建 MatterGen venv + 预下 chemical_system 权重到 ~/hf_cache
singularity exec "$IMG" bash ~/AI4SSB/project3_generative/setup_mattergen.sh

# B 阶段：catboost+pymatgen 叠进 ~/p3pkg（复用 ~/macepkg）+ 预下 MACE-MP-0 权重到 ~/.cache/mace
singularity exec "$IMG" bash ~/AI4SSB/project3_generative/setup_screen.sh
```

看到 `MATTERGEN_WEIGHTS_OK` 和 `mace/catboost/pymatgen ok` + `MACE_WEIGHTS_OK` 即装好
（后者也顺带验证了「复用 macepkg + 叠 p3pkg」这条假设成立）。

## 2) 提交作业（计算节点，离线）

一个 GPU 作业、两个 `singularity exec`（A→B）。**注意 NUS 上不加 `--nv`**（容器自动暴露 GPU，加了反而 CUDA 不可见）。

```bash
cd ~/AI4SSB/project3_generative
# 先 smoke（几分钟，验证两阶段管路）
qsub -v BATCH=4,NBATCH=1,EHULL=0.1,TOP=5 run_p3.pbs
# 通了再正式跑（~64 个候选）
qsub -v BATCH=16,NBATCH=4,EHULL=0.1,TOP=5 run_p3.pbs
```

参数（`qsub -v`，逗号分隔变量；本作业的值里都没有逗号，故不需要项目二 TEMPS 那种 `_`→`,` 技巧）：
`BATCH` 每批结构数、`NBATCH` 批数（总数=两者乘积）、`GUIDANCE` 扩散引导强度、
`CHEMSYS` 化学体系（默认 Li-P-S）、`EHULL` 稳定性阈值 eV/atom、`TOP` shortlist 大小。
个人免费配额跑，**不加 `-P`**。

## 3) 提交 / 监控 / 取结果（可从 Mac 异步操作）

```bash
# 一行提交（需先配好 passwordless ssh 到 vanda；校外先连 NUS nVPN）
ssh vanda "cd ~/AI4SSB/project3_generative && qsub -v BATCH=4,NBATCH=1 run_p3.pbs"

# 监控
qstat -u $USER         # 我的全部作业
qstat -f <jobid>       # 单个作业详情
qdel  <jobid>          # 取消

# 取回结果到本机
rsync -az vanda:~/AI4SSB/project3_generative/{data,figures}/ ./
```

> 作业一旦 `qsub` 进队列就在服务器端独立跑，**中途断网/关 Mac 不影响**。
> GPU 队列每个作业上限 **2 GPU + 2 节点**（不是每用户）；要并行多个化学体系/参数就**多投几个作业**。
> 无 `-P` = 个人免费配额、优先级较低，GPU 忙时可能排队。

## 4) 产物

- `data/candidates_final.csv` — 全部候选，按预测 log₁₀σ 排序，含 e_above_hull / S.U.N. 标记。
- `figures/01_landscape.png` — 稳定性 vs 电导率 landscape。
- `figures/02_shortlist.png` — 最终 S.U.N. shortlist。

把 shortlist 头部喂给项目二 MLIP-MD 算真实 σ/Eₐ（W11），再回填项目三 README 的结果表。

## 风险与兜底

| 环节 | 状态 |
|---|---|
| B 阶段（MACE 筛 + 打分 + 排序） | 复用**项目二已在 A40 跑通的** MACE 路径（baseline σ300≈29.6 mS/cm）+ 已装 macepkg；本机 CPU 管路也验过 → 低风险 |
| A 阶段（MatterGen 装 + 跑） | 命令按官方 README 核准；MatterGen 为本项目新引入，首次在 Vanda 运行要确认三点（见下） |

**A 阶段首次运行要盯三点**：
1. **权重路径**：`setup_mattergen.sh` 用 `snapshot_download(microsoft/mattergen, checkpoints/chemical_system/*)`
   预下到 `~/hf_cache`，PBS 把同一 `HF_HOME` 传进容器。若上游改了 checkpoint 路径，报错就在这一行；改 `allow_patterns` 即可。
2. **uv 建 venv 要联网**：必须在**登录节点**跑 `setup_mattergen.sh`（计算节点离线）；venv 与权重都落在 `$HOME`，计算节点能直接用。
3. **venv torch 的 CUDA build ≤ 12.4**（已知坑）：`setup_mattergen.sh` 的断言拦「比驱动新」(`cu130`) 或 `cpu` 版
   （会在 A40 上静默回落 CPU）；MatterGen 的 `cu118` 在 12.4 驱动上 OK。看到 `TORCH_CUDA_OK` 才算过。
   ⚠️ 别手动改 venv 里 torch 的 cu 版——`torch_scatter` 等扩展按 cu118 编译，换了会崩（`libcudart.so.11.0` 找不到）。

**兜底**：万一 A 阶段在 Vanda 装不顺，stage A 改用免费 Colab T4 跑（notebook `01_mattergen_pipeline.ipynb` 阶段 A，已就绪），
把 `generated_crystals_cif.zip` 下载下来放进 `data/mattergen_run/`，Vanda 上只提交 stage B
（把 `run_p3.pbs` 里 stage A 那行 `singularity exec ... gen_in_container.sh` 注释掉即可）。

> 可选升级：官方还有 `chemical_system_energy_above_hull` 多属性模型，能同时按「体系 + 稳定性」条件生成
> （`--properties_to_condition_on="{'energy_above_hull': 0.05, 'chemical_system': 'Li-P-S'}"`），候选更靠近凸包。
> 想要更强信号时换这个 `--pretrained-name` 即可。
