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
- [Evaluation](#evaluation)
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

## Evaluation

If your config defines **two** `test_dataloader` entries for `test` and `Generalizability`, a single command evaluates both:

```bash
python tools/test.py   configs/mambavision/U-MV-small.py   work_dirs/U-MV-small/latest.pth --eval mIoU mFscore
```

---

## Inference

This project includes **two geospatial inference utilities** that post-process predictions into GIS vector formats (GeoPackage / Shapefile). They assume **georeferenced** source imagery (e.g., GeoTIFF) and require geospatial libs (see `extras/geo-requirements.txt` if provided).

### A) Single image
```bash
python tools/geospatial_inference.py  
```
**What it does**
1. Runs tile-wise inference over large imagery (sliding window if needed).  
2. Reassembles a full-size raster mask in the image CRS.  
3. Vectorizes polygons (e.g., crowns), optionally filters small objects.  
4. Writes a **GeoPackage** (`.gpkg`) or **ESRI Shapefile** (`.shp`).

### B) Batch processing (folders of images)
```bash
python tools/Batch_processing_geospatial_inference.py   
```
**Notes**
- The script walks subfolders and processes each georeferenced image.  
- Output vector files mirror the input folder structure.  
---

## Reproducibility & Environment

Export your exact environment (optional but reviewer‑friendly). From your container (replace `mvmmseg` with your name/id):

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

Commit these files for transparency; keep the top-level `requirements.txt` minimal and pip‑installable.

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

- Built on **MMSegmentation** by OpenMMLab: <https://github.com/open-mmlab/mmsegmentation>  
- Uses **MambaVision** backbones by NVLabs: <https://github.com/NVlabs/MambaVision>  
- Model weights are loaded via **Hugging Face Transformers**.
