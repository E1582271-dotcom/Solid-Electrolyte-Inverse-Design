"""Generate 01_mattergen_pipeline.ipynb (valid nbformat-4 JSON, no nbformat dep).

Colab-first orchestration of Project 3's inverse-design demo. Two stages:
  A) MatterGen conditional generation in an isolated uv/py3.10 env (CLI subprocess);
  B) stability screen (MACE hull) + project-1 conductivity score in the kernel env.
Re-run this script to regenerate the notebook after editing the cell sources below.
"""
import json
import os

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n").splitlines(keepends=True)}

cells = [
md("""# 03 inverse design — MatterGen → MLIP 稳定性筛 → 项目一电导率打分 (W10, 彩蛋)

**定位：概念验证（concept-validation）**，展示 inverse-design 范式 + 串起 *generate → screen → validate* 闭环。
**不主张可合成性** —— MatterGen 原论文仅做一例合成验证，这里把它当主菜反而信号弱。

串联叙事中的位置：项目一(筛选) → **项目三(生成新候选)** → 项目二(高精度 MLIP-MD 验证) → 未来电化学实验闭环。

---
**两阶段**（MatterGen 需独立 Python 3.10 环境，与筛选环境隔离）：
- **A 生成**：在 uv 建的 py3.10 venv 里跑官方 `mattergen-generate` CLI（条件 = 化学体系 Li-P-S）→ `generated_crystals_cif.zip`。
- **B 筛选+打分**：回到 kernel 环境，用 MACE-MP-0 自洽凸包算 `e_above_hull` 筛稳定 → 项目一 CatBoost 模型给电导率打分 → 排序出图。

> **Vanda OnDemand 备选**（用户首选 HPC）：阶段 A 在 Vanda 容器里用 `PYTHONUSERBASE=~/mgenpkg` + `pip install --user mattergen` 装，CLI 同样可跑；阶段 B 复用已验证的 `~/macepkg` MACE 装法。命令同此 notebook，只是把 `!`-shell 换成容器内执行。"""),

md("""## 0) 环境 + 取代码

项目三的打分步骤 (`src/score.py`) 会向上找 `../project1_screening/`（复用项目一训练好的 `catboost_model.cbm`），
所以 **project1_screening 与 project3_generative 必须并排放在同一父目录**（即本机 `~/Code/AI4SSB/` 的布局）。

下面两选一：A) 填 `REPO_URL` git clone（若已把 AI4SSB 推成一个 repo）；B) 否则上传 `ai4ssb.zip`
（内含 `project1_screening/` + `project3_generative/` 两个文件夹）自动解压。"""),

code("""# GPU 自检
import torch, sys
print("python", sys.version.split()[0], "| torch", torch.__version__, "| CUDA", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only (生成会很慢)")"""),

code("""# 取代码：A) git clone  或  B) 上传 ai4ssb.zip（含 project1_screening + project3_generative）
import os, glob, zipfile, subprocess
REPO_URL = ""   # 例: https://github.com/<user>/AI4SSB.git ；留空则走上传 zip 分支
ROOT = "/content/AI4SSB"

if REPO_URL:
    if not os.path.isdir(ROOT):
        subprocess.run(["git", "clone", REPO_URL, ROOT], check=True)
else:
    if not os.path.isdir(ROOT):
        from google.colab import files          # 上传含两个 project 文件夹的 zip
        up = files.upload()
        zname = next(k for k in up if k.endswith(".zip"))
        os.makedirs(ROOT, exist_ok=True)
        with zipfile.ZipFile(zname) as zf:
            zf.extractall(ROOT)
        # 若 zip 里多包了一层（AI4SSB/AI4SSB/...）则下探一层
        if not os.path.isdir(os.path.join(ROOT, "project3_generative")):
            inner = [d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d)]
            if len(inner) == 1:
                ROOT = inner[0]

P3 = os.path.join(ROOT, "project3_generative")
assert os.path.isfile(os.path.join(ROOT, "project1_screening", "catboost_model.cbm")), \
    "找不到项目一模型 catboost_model.cbm —— 请确保 project1_screening 与 project3_generative 并排"
os.chdir(P3)
print("工作目录:", os.getcwd())"""),

code("""# MP_API_KEY（阶段 B 拉参考相建凸包用）—— 存 Colab Secret 名为 MP_API_KEY
import os
try:
    from google.colab import userdata
    os.environ["MP_API_KEY"] = userdata.get("MP_API_KEY")
    print("MP_API_KEY 已从 Colab Secret 载入")
except Exception as e:
    print("未取到 Secret（可改用 project1_screening/mp_api_key.txt）:", e)"""),

md("""## A) 生成 —— MatterGen 条件生成 Li-P-S 候选

在隔离的 uv/py3.10 venv 里装 MatterGen 并跑 CLI（避免与 kernel 的 torch/e3nn 冲突）。
**安装/调用命令以 [microsoft/mattergen](https://github.com/microsoft/mattergen) 官方 README 为准**（版本会变）。
`chemical_system` 预训练权重在 HuggingFace，首次自动下载。"""),

code("""# A1) 隔离环境装 MatterGen（git clone + uv，py3.10）。首次约 5–10 分钟。
import os, subprocess
MG = "/content/mattergen"
if not os.path.isdir(MG):
    subprocess.run(["git", "clone", "https://github.com/microsoft/mattergen.git", MG], check=True)
subprocess.run(["pip", "install", "-q", "uv"], check=True)
# 在 mattergen 目录建 py3.10 venv 并可编辑安装；后续直接调 .venv/bin 里的可执行文件
subprocess.run(["uv", "venv", ".venv", "--python", "3.10"], cwd=MG, check=True)
subprocess.run(["uv", "pip", "install", "-e", ".", "--python", ".venv/bin/python"], cwd=MG, check=True)
print("MatterGen 环境就绪:", MG)"""),

code("""# A2) 条件生成：以化学体系 Li-P-S 为条件采样候选结构
#    薄跑：batch_size=16, num_batches=2（共 ~32 个）。要更多就加 num_batches。
import os, shutil, subprocess
MG = "/content/mattergen"
RESULTS = os.path.join(MG, "results", "chemical_system")
cmd = [".venv/bin/mattergen-generate", RESULTS,
       "--pretrained-name=chemical_system",
       "--batch_size=16", "--num_batches=2",
       "--properties_to_condition_on={'chemical_system': 'Li-P-S'}",
       "--diffusion_guidance_factor=2.0"]
print("$", " ".join(cmd))
subprocess.run(cmd, cwd=MG, check=True)

# 把 MatterGen 输出搬到项目三 data/ 下，供 from-results 读取
import os
DST = os.path.join(os.getcwd(), "data", "mattergen_run")
os.makedirs(DST, exist_ok=True)
for fn in ("generated_crystals_cif.zip", "generated_crystals.extxyz"):
    src = os.path.join(RESULTS, fn)
    if os.path.exists(src):
        shutil.copy(src, DST)
print("已复制 MatterGen 输出到", DST, "->", os.listdir(DST))"""),

code("""# A3) 归一化为统一 manifest + 单结构 CIF（纯 pymatgen，跑在 kernel 里）
!python 01_generate.py --source from-results --results-path data/mattergen_run"""),

md("""## B) 筛选 + 打分 —— MACE 自洽凸包稳定性筛 + 项目一电导率打分

- **稳定性**：参考相和候选都用**同一** MACE-MP-0 弛豫，凸包由这些 MLIP 能量自洽建立（避免把 MLIP 候选能量与 PBE+U 参考能量混用而系统性偏差；见 `src/stability.py`）。参考相能量按 material_id 缓存，重跑很快。
- **S.U.N.**：Stable（`e_above_hull` ≤ cutoff）+ Unique（候选间去重）+ Novel（不匹配任何 MP 已知相）—— MatterGen 自身报告的标准透镜。
- **打分**：复用项目一 OBELiX-CatBoost 模型给存活候选一个**粗电导率先验（排序用，非定量 σ）**。"""),

code("""# B1) 装筛选/打分重包到 kernel（MACE + pymatgen 栈）。约 2–4 分钟。
!pip install -q mace-torch chgnet pymatgen pymatgen-analysis-diffusion mp-api catboost"""),

code("""# B2) 稳定性筛：MACE 自洽凸包 → e_above_hull + 去重 + 新颖性
#     首次会下载 MACE-MP-0 权重；参考相弛豫能量缓存到 data/hull_energy_cache.json
!python 02_screen_stability.py --calc mace --ehull-cutoff 0.1 --fmax 0.05 --steps 200"""),

code("""# B3) 用项目一模型给存活候选打电导率先验分（只打稳定的）
!python 03_score_conductivity.py --stable-only"""),

code("""# B4) 合并 + 排序 + 出图（landscape + 最终 shortlist）
!python 04_rank_candidates.py --top 5"""),

code("""# B5) 展示结果图
from IPython.display import Image, display
for f in ["figures/01_landscape.png", "figures/02_shortlist.png"]:
    print(f); display(Image(f))"""),

md("""## 跑通之后 / 诚实披露（写进 README + 技术报告）

- **生成结构大多不稳定/不可合成** → 必须过 MLIP 稳定性筛；这是 demo 的核心信号点。
- **e_above_hull 来自 universal MLIP**（未针对硫化物微调），是相对稳定性指标，非 DFT 级定量；自洽凸包消除了参考态不一致，但 MLIP 本身的偏差仍在。
- **电导率分是粗排序先验**（项目一模型），生成的新化学计量比 `Family='unknown'`、对称性常为 P1，分数只用于排序、交给项目二 MD 验证。
- **彩蛋定位**：申请材料里**不过度承诺**，明确写「概念验证 + 闭环演示」。
- **下一步**：把 S.U.N. shortlist 的头部喂给项目二 MLIP-MD 算真实 σ/Eₐ；W11 画统一 pipeline 流程图。"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_mattergen_pipeline.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", out, "with", len(cells), "cells")
