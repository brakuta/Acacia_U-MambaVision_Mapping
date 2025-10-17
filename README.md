# Acacia tortilis Mapping from Large-Scale UAV Imagery

**U-Shape-MambaVision** — A custom U-Net–style decoder with a **MambaVision** backbone for semantic segmentation of *Acacia tortilis* crowns in ultra‑high‑resolution UAV images. Built on **MMSegmentation**.

> **Highlights**
> - Plug‑in backbone via Hugging Face (`nvidia/MambaVision-*-1K`)
> - Simple U‑Net decoder over 4 feature pyramid levels
> - Reproducible configs for **Tiny / Small / Base**
> - Two test splits: **test** (in‑distribution) and **Generalizability** (OOD)

---

## Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Data Layout](#data-layout)
- [Configs](#configs)
- [Training](#training)
- [Evaluation (two test sets)](#evaluation-two-test-sets)
- [Inference](#inference)
- [Reproducibility & Environment](#reproducibility--environment)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Quick Start

> Quick Start uses the **copy‑in** workflow (no submodules) to keep things simple.

```bash
# 1) Clone MMSeg and install editable
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
cd ..

# 2) Clone this repo (sibling to mmsegmentation/)
git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git

# 3) Copy custom code + configs into your MMSeg clone
cp -r Acacia_U-MambaVision_Mapping/mmseg mmsegmentation/
cp -r Acacia_U-MambaVision_Mapping/configs mmsegmentation/

# 4) Install dependencies (install PyTorch CUDA wheels first)
cd mmsegmentation
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118   --index-url https://download.pytorch.org/whl/cu118
pip install -r ../Acacia_U-MambaVision_Mapping/requirements.txt
```

> Prefer submodules? See [Installation](#installation) for the **git submodule** option.

---

## Installation

You can use **either** workflow:

### Option A — Copy‑in (same as Quick Start)
1. Clone and `pip install -e` **mmsegmentation**
2. Clone **this repo**
3. Copy `mmseg/` and `configs/` into your `mmsegmentation/`
4. Install dependencies (PyTorch CUDA wheels first)

### Option B — Submodule (more reproducible)
```bash
# From your new project root
git init
git submodule add https://github.com/open-mmlab/mmsegmentation.git mmsegmentation

# Install MMSeg editable
cd mmsegmentation && pip install -v -e . && cd ..

# Bring in this repo's custom code (or keep it in your project and point to it)
git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git tmp_repo
cp -r tmp_repo/mmseg mmsegmentation/
cp -r tmp_repo/configs mmsegmentation/
rm -rf tmp_repo

# Dependencies
cd mmsegmentation
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118   --index-url https://download.pytorch.org/whl/cu118
pip install -r ../Acacia_U-MambaVision_Mapping/requirements.txt
```

---

## Data Layout

Organize your dataset as follows (indexed masks; basenames must match):  

```
uav_acacia/
├── img_dir/
│   ├── train/
│   ├── val/
│   ├── test/                 # in-distribution test
│   └── Generalizability/     # OOD test
└── ann_dir/
    ├── train/
    ├── val/
    ├── test/
    └── Generalizability/
```

- **Images**: `.tif` or common image formats readable by MMSeg.  
- **Masks**: **indexed** `.png` (preferred) or `.tif` with per‑pixel class IDs (not RGB).  
- **Classes (binary)**: `0 = background`, `1 = acacia` (adjust if needed).

In your configs, set (example):
```python
data_root = 'uav_acacia'
classes = ('background', 'acacia')
palette = [[0, 0, 0], [0, 255, 0]]
```

---

## Configs

After copying, configs live under:
```
mmsegmentation/configs/mambavision/
  U-MV-tiny.py
  U-MV-small.py
  U-MV-base.py
```
Each config:
- Imports the **backbone** from `mmseg/custom_models/mamba_vision.py`
- Imports the **U‑Net head** from `mmseg/custom_models/generic_unet_head.py`
- Uses the **custom dataset base** (not ADE20K) and defines your `classes/palette`
- Provides two test dataloaders (`test` and `Generalizability`) or you can override paths via `--cfg-options`

> If your files live under `mmseg/custom_models/`, include in the config:
> ```python
> custom_imports = dict(
>     imports=[
>         'mmseg.custom_models.mamba_vision',
>         'mmseg.custom_models.generic_unet_head',
>     ],
>     allow_failed_imports=False
> )
> ```

---

## Training

From inside **mmsegmentation/**:
```bash
# Tiny
python tools/train.py configs/mambavision/U-MV-tiny.py

# Small
python tools/train.py configs/mambavision/U-MV-small.py

# Base
python tools/train.py configs/mambavision/U-MV-base.py
```

Tips:
- Set `work_dir` in each config to control where checkpoints/logs are saved.
- For determinism: `env_cfg = dict(cudnn_benchmark=False)` and fix seeds as needed.

---

## Evaluation (two test sets)

If your config defines **two** `test_dataloader` entries for `test` and `Generalizability`, a single command evaluates both:

```bash
python tools/test.py   configs/mambavision/U-MV-small.py   work_dirs/U-MV-small/latest.pth --eval mIoU mFscore
```

MMSeg prints separate metrics with prefixes (e.g., `test/mIoU` for in‑distribution, `ood/mIoU` for Generalizability).

Evaluate a specific split by overriding paths:
```bash
# OOD only
python tools/test.py   configs/mambavision/U-MV-small.py   work_dirs/U-MV-small/latest.pth --eval mIoU mFscore   --cfg-options   test_dataloader.dataset.data_prefix.img_path=img_dir/Generalizability   test_dataloader.dataset.data_prefix.seg_map_path=ann_dir/Generalizability
```

---

## Inference

Single image or folder (writes overlays to `--out-dir`):

```bash
# Standard MMSeg image demo
python demo/image_demo.py   path/to/image_or_folder   configs/mambavision/U-MV-small.py   --device cuda:0   --opacity 0.0   --out-dir outputs/vis
```

> If you provide a custom script such as `tools/geospatial_inference.py`, document its arguments in `docs/usage.md`.

---

## Reproducibility & Environment

You can include exact environment exports used during training (optional but reviewer-friendly). From your container (replace `mvmmseg` with your name/id):

```bash
docker exec mvmmseg pip freeze > requirements_full.txt
docker exec mvmmseg conda env export --no-builds > environment.yml || true
docker exec mvmmseg bash -lc 'nvcc --version > cuda.txt || true'
docker exec mvmmseg python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cudnn', torch.backends.cudnn.version())
PY
```

Commit these files for transparency; keep the top-level `requirements.txt` minimal and pip-installable.

---

## Troubleshooting

- **Module not found** (`mmseg.custom_models...`)  
  Ensure you ran `pip install -v -e .` **inside `mmsegmentation/`** and that `custom_models/__init__.py` exists.

- **CUDA mismatch**  
  Install the PyTorch wheel that matches your CUDA/driver using the correct `--index-url`.

- **Mask format errors**  
  Masks must be **indexed** (class IDs per pixel), not RGB color masks.

- **Tiny channels differ**  
  If backbone variant changes, update `encoder_channels` in the config to match the backbone’s feature dims.

---

## Citation

If you use this repository, please cite:

> **Regional-Scale Acacia tortilis Crown Mapping from UAV Remote Sensing Using Semi‑Automated Annotation and a Lightweight Hybrid Segmentation Framework**  
> Mohamed Barakat, *et al.*, 2025.

```bibtex
@article{Barakat2025AcaciaUAVMambaVision,
  title   = {Regional-Scale Acacia tortilis Crown Mapping from UAV Remote Sensing Using Semi-Automated Annotation and a Lightweight Hybrid Segmentation Framework},
  author  = {Barakat, Mohamed and Others},
  year    = {2025},
  journal = {TBD}
}
```

---

## License

Specify your license (e.g., **Apache-2.0**) in `LICENSE`.

---

## Acknowledgments

Built with **MMSegmentation** and MambaVision backbones via **Hugging Face Transformers**. Thanks to the OpenMMLab community.
