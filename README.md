
# Regional-Scale *Acacia tortilis* Crown Mapping from UAV Imagery (U-MV)

**U-MV (U-Shape-MambaVision)** — A lightweight U-shaped semantic segmentation framework (U-Net–style decoder + **MambaVision** backbone) for delineating *Acacia tortilis* crowns in ultra-high-resolution UAV imagery. Built on **MMSegmentation**.

This repository provides the official implementation for the paper:
**"Regional-Scale *Acacia tortilis* Crown Mapping from UAV Remote Sensing Using Semi-Automated Annotation and a Lightweight Hybrid Segmentation Framework"** by Barakat *et al.*, 2025.

## 🌟 Highlights
- **Architecture:** Integrates **MambaVision** backbones (from Hugging Face: `nvidia/MambaVision-*-1K`) with a U-Net–style decoder over four feature pyramid levels.
- **Reproducible Configurations:** Provides ready-to-use configurations for **Tiny, Small, and Base** model variants.
- **Geospatial Utilities:** Features specialized inference tools to convert model predictions into GIS-compatible vector formats (GeoPackage/Shapefile) for direct use in geospatial analyses.


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
## 📂 Project Structure

This project extends MMSegmentation by adding custom modules and configuration files. After the installation steps, your mmsegmentation/ directory will contain:  

```
mmsegmentation/
├── mmseg/
│   ├── custom_models/        # Our custom MambaVision backbone and U-Net head
│   │   ├── mamba_vision.py
│   │   └── generic_unet_head.py
│   └── ... (MMSegmentation's original files)
├── configs/
│   ├── mambavision/          # Our specific model configurations
│   │   ├── U-MV-tiny.py
│   │   ├── U-MV-small.py
│   │   └── U-MV-base.py
│   └── ... (MMSegmentation's original configs)
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── geospatial_inference.py           # Our custom single image inference script
│   └── Batch_processing_geospatial_inference.py # Our custom batch inference script
├── requirements.txt          # Project-specific dependencies
└── ... (Other MMSegmentation files)
```

---
## 🧠 Model Architecture

The U-MV framework combines a MambaVision backbone for robust feature extraction with a U-Net style decoder for precise semantic segmentation. The overall architecture is illustrated below:

![Model Architecture](Assets/mamba_vision_architecture.png)


---
## 🌳 Data Layout

Organize your dataset with separate directories for images (img_dir) and corresponding indexed segmentation masks (ann_dir). Basenames of image and annotation files within corresponding subdirectories (e.g., train, val, test, Generalizability) must match.:  

```
UAV_Acacia_Data/
├── img_dir/
│   ├── train/                 # Training images
│   ├── val/                   # Validation images
│   ├── test/                  # In-distribution test images
│   └── Generalizability/      # Out-of-distribution (OOD) test images
└── ann_dir/
    ├── train/                 # Training masks
    ├── val/                   # Validation masks
    ├── test/                  # In-distribution test masks
    └── Generalizability/      # Out-of-distribution (OOD) test masks
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
## ⚙️ Configuration Files

Our custom model configurations are located under mmsegmentation/configs/mambavision/:

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

> If your custom files are placed in mmseg/custom_models/, you must include the following custom_imports section in your configuration file to ensure they are correctly loaded by MMSegmentation:
> 
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

## 🏋️ Training

To train a model, navigate to the mmsegmentation/ directory and run the tools/train.py script with your desired configuration file::
```bash
# From inside the mmsegmentation/ directory

# Train the Tiny model
python tools/train.py configs/mambavision/U-MV-tiny.py

# Train the Small model
python tools/train.py configs/mambavision/U-MV-small.py

# Train the Base model
python tools/train.py configs/mambavision/U-MV-base.py
```

Training Tips:
- Set `work_dir` in each config to control where checkpoints/logs are saved.
- For determinism: `env_cfg = dict(cudnn_benchmark=False)` and fix seeds as needed.

---

## 📊 Evaluation

If your config defines **two** `test_dataloader` entries for `test` and `Generalizability`, a single command evaluates both:

```bash
python tools/test.py   configs/mambavision/U-MV-small.py   work_dirs/U-MV-small/latest.pth --eval mIoU mFscore
```
- Replace `configs/mambavision/U-MV-small.py` with the path to your desired model configuration.

- Replace `work_dirs/U-MV-small/latest.pth` with the path to your trained model checkpoint (e.g., latest.pth for the most recent or a specific epoch checkpoint).

- The `--eval mIoU mFscore` flags specify the evaluation metrics to compute (Mean IoU and Mean F-score are standard for segmentation).
  
---

## 🌍 Inference

This project includes **two geospatial inference utilities** that post-process predictions into GIS vector formats (GeoPackage / Shapefile). They assume **georeferenced** source imagery (e.g., GeoTIFF) and require geospatial libs (see `extras/geo-requirements.txt` if provided).

### A) Single image
```bash
python tools/geospatial_inference.py  
```
**What it does**
1. Performs tile-wise inference over large input imagery using a sliding window approach to handle memory constraints.  
2. Reassembles the tile predictions into a full-size raster mask in the original image's Coordinate Reference System (CRS).  
3. Vectorizes the predicted polygons (e.g., individual tree crowns) from the raster mask.
4. Optionally filters out small, spurious objects based on a minimum area threshold.
5. Writes a **GeoPackage** (`.gpkg`) or **ESRI Shapefile** (`.shp`)format, preserving spatial information..

### B) Batch processing (folders of images)
This script is designed for automated processing of multiple georeferenced images organized within folders.
```bash
python tools/Batch_processing_geospatial_inference.py   
```
**Notes**
- The script recursively walks through subfolders within the '--input-folder' and processes each georeferenced image found.  
- Output vector files mirror the input folder structure.  


---

## Citation

If you find this repository or the U-MV framework useful in your research, please cite our paper::

> **Regional-Scale Acacia tortilis Crown Mapping from UAV Remote Sensing Using Semi‑Automated Annotation and a Lightweight Hybrid Segmentation Framework**  
> Mohamed Barakat, *et al.*, 2025.

```bibtex
@article{Barakat2025AcaciaUAVUMV,
  title   = {Regional-Scale \textit{Acacia tortilis} Crown Mapping from UAV Remote Sensing Using Semi-Automated Annotation and a Lightweight Hybrid Segmentation Framework},
  author  = {Gibril, M.B.A. and others},
  year    = {2025},
  journal = {In review / in press}
}
```

---

## License

Specify your license (e.g., **Apache-2.0**) in `LICENSE`.

---

## 🙏Acknowledgments


We gratefully acknowledge the foundational work of:

- **MMSegmentation (OpenMMLab):** the semantic segmentation toolbox on which this project is built.  
  https://github.com/open-mmlab/mmsegmentation

- **MambaVision (NVLabs):** for providing the MambaVision backbones used in our framework.  
  https://github.com/NVlabs/MambaVision

- **Hugging Face:** for hosting and enabling access to the pretrained MambaVision model weights.  
  https://huggingface.co/nvidia

