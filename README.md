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

```bash
# 1) Clone MMSeg and install editable
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .

# 2) Clone this repo (in the same parent dir as mmsegmentation)
cd ..
git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git
cd Acacia_U-MambaVision_Mapping

# 3) Copy custom code + configs into your MMSeg clone
cp -r mmseg ../mmsegmentation/
cp -r configs ../mmsegmentation/

# 4) Install dependencies (install PyTorch CUDA wheels first)
cd ../mmsegmentation
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r ../Acacia_U-MambaVision_Mapping/requirements.txt
```

---

## Installation

> If you prefer, you can keep `mmsegmentation` as a **git submodule** in this project. The steps below match a simple “copy‑in” workflow.

### 1) MMSegmentation
```bash
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation && pip install -v -e . && cd ..
```

### 2) This repository
```bash
git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git
cd Acacia_U-MambaVision_Mapping
cp -r mmseg ../mmsegmentation/
cp -r configs ../mmsegmentation/
```

### 3) Python dependencies
```bash
cd ../mmsegmentation
# Install CUDA 11.8 builds (adjust if your CUDA differs)
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Then the rest
pip install -r ../Acacia_U-MambaVision_Mapping/requirements.txt
```

---

## Data Layout

Organize your dataset as follows (indexed PNG masks; basenames must match):  

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
- **Masks**: **indexed** `.png` with per‑pixel class IDs (not RGB).  
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
- Sets `num_classes=2` and wires **two test sets** (`test` and `Generalizability`)

> If your files are inside `mmseg/custom_models/`, the config must include:
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

If your config defines **two** `test_dataloader` entries (list‑of‑dicts) for `test` and `Generalizability`, a single command evaluates both:

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
python demo/image_demo.py   path/to/image_or_folder   configs/mambavision/U-MV-small.py   --device cuda:0   --opacity 0.0   --out-dir outputs/vis
```

---

## Reproducibility & Environment

You can include exact environment exports used during training (optional but reviewer‑friendly). From your container (replace `mvmmseg` with your name/id):

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

Commit these files for transparency; keep the top‑level `requirements.txt` minimal and pip‑installable.

---

## Troubleshooting

- **Module not found** (`mmseg.custom_models...`)  
  Ensure you ran `pip install -v -e .` **inside `mmsegmentation/`** and that `custom_models/__init__.py` exists.

- **CUDA mismatch**  
  Install the PyTorch wheel that matches your CUDA/driver using the correct `--index-url`.

- **Mask format errors**  
  Masks must be **indexed PNGs** with integer class IDs (not RGB colors).

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
