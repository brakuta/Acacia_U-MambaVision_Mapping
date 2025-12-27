readme_content = """
# Regional-Scale *Acacia tortilis* Crown Mapping from UAV Imagery (U-MV)

**U-MV (U-Shape-MambaVision)** — A lightweight U-shaped semantic segmentation framework (U-Net–style decoder + **MambaVision** backbone) for delineating *Acacia tortilis* crowns in ultra-high-resolution UAV imagery. Built on **MMSegmentation**.

This repository provides the official implementation for the paper:
**"Regional-Scale *Acacia tortilis* Crown Mapping from UAV Remote Sensing Using Semi-Automated Annotation and a Lightweight Hybrid Segmentation Framework"** by Barakat *et al.*, 2025.

## 🌟 Highlights
- **Novel Architecture:** Integrates **MambaVision** backbones (from Hugging Face: `nvidia/MambaVision-*-1K`) with a U-Net–style decoder over four feature pyramid levels.
- **Reproducible Configurations:** Provides ready-to-use configurations for **Tiny, Small, and Base** model variants.
- **Robust Evaluation:** Includes two distinct evaluation splits: **test** (in-distribution) and **Generalizability** (out-of-distribution) to assess model performance and adaptability.
- **Geospatial Utilities:** Features specialized inference tools to convert model predictions into GIS-compatible vector formats (GeoPackage/Shapefile) for direct use in geospatial analyses.

## 💾 Data Availability
The UAV orthomosaic imagery used for this research is **restricted** due to sensitive land-use information and cannot be publicly shared. However, the derived vector layers used for model development and evaluation (e.g., ground truth polygons) are available upon reasonable request from the corresponding author (see `DATA_AVAILABILITY.md` for details and contact information).

## 📝 Table of Contents
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Layout](#data-layout)
- [Configuration Files](#configuration-files)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🚀 Quick Start

This project utilizes a **copy-in** workflow, integrating custom components into an MMSegmentation environment. Follow these steps to get started rapidly:

1.  **Clone MMSegmentation and install in editable mode:**
    ```bash
    git clone https://github.com/open-mmlab/mmsegmentation.git
    cd mmsegmentation
    pip install -v -e .
    cd ..
    ```

2.  **Clone this repository:**
    ```bash
    git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git
    ```

3.  **Copy custom code and configurations:**
    ```bash
    cp -r Acacia_U-MambaVision_Mapping/mmseg mmsegmentation/
    cp -r Acacia_U-MambaVision_Mapping/configs mmsegmentation/
    ```

4.  **Install dependencies (ensure PyTorch CUDA wheels are installed first):**
    ```bash
    cd mmsegmentation
    pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
      --index-url https://download.pytorch.org/whl/cu118
    pip install -r ../Acacia_U-MambaVision_Mapping/requirements.txt
    ```

## 🛠️ Installation

You can choose between two primary installation workflows:

### Option A — Copy-in (Recommended for quick setup)
This is the same workflow as described in [Quick Start](#quick-start). It's straightforward and suitable for most users.

1.  Clone and `pip install -e` **MMSegmentation**.
2.  Clone **this repository**.
3.  Copy `mmseg/` and `configs/` from this repo into your `mmsegmentation/` directory.
4.  Install dependencies (PyTorch CUDA wheels first) as shown above.

### Option B — Submodule (Recommended for version control and reproducibility)
This method integrates MMSegmentation as a Git submodule, which can be beneficial for managing dependencies in larger projects or ensuring specific MMSegmentation versions.

```bash
# From your new project root
git init
git submodule add https://github.com/open-mmlab/mmsegmentation.git mmsegmentation

# Install MMSegmentation in editable mode
cd mmsegmentation && pip install -v -e . && cd ..
    
# Bring in this repo's custom code and configurations
git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git tmp_repo
cp -r tmp_repo/mmseg mmsegmentation/
cp -r tmp_repo/configs mmsegmentation/
rm -rf tmp_repo

# Install dependencies (PyTorch CUDA wheels first)
cd mmsegmentation
pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 torchaudio==2.6.0+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
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
- Model weights are loaded via **Hugging Face Transformers**: <https://huggingface.co/nvidia>
