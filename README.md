# Acacia tortilis Mapping from Large-scale UAV-based Images
A code for Mapping Acacia tortilis trees from UAV images using a custom U-Shape-MambaVision model

## Overview

This project presents a lightweight, U-Shape hybrid framework for the semantic segmentation of *Acacia tortilis* trees from ultra-high-resolution UAV imagery. Our model, **U-MambaVision (U-MV)**, pairs a powerful **MambaVision** backbone with a custom U-Net style decoder, all built upon the MMSegmentation framework.

This repository provides all the necessary components to reproduce our results, including:
*   Custom model source code for the U-MV architecture.
*   Configuration files for three model variants: Tiny, Small, and Base.
*   An optimized inference script for processing large-scale geospatial rasters.


## Installation

**Prerequisites:**
*   An NVIDIA GPU with CUDA 11.8 installed.
*   Python 3.8+ and Git.

1. Clone Repositories:
Create a main project folder and clone the official MMSegmentation repository and this repository side-by-side.
```bash
# Example: Create a folder named 'Projects'
mkdir Projects && cd Projects

# Clone the two required repositories
git clone https://github.com/open-mmlab/mmsegmentation.git
git clone https://github.com/brakuta/Acacia_U-MambaVision_Mapping.git



2. Set Up Python Environment:
Navigate into the mmsegmentation folder and create a Python virtual environment.

cd mmsegmentation
python -m venv venv
# Activate on Windows: .\venv\Scripts\activate
# Activate on Linux/macOS: source venv/bin/activate


3. Install Dependencies:
First, install MMSegmentation itself. Then, install the specific packages from our requirements.txt file.

# Install mmsegmentation in editable mode
pip install -e .

# Install all other packages from our requirements file
# The --index-url is critical for getting the correct PyTorch version
pip install -r ../Acacia_U-MambaVision_Mapping/requirements.txt --index-url https://download.pytorch.org/whl/cu118






#Usage
Data Preparation

1. Organize Your Data Folders:
Your dataset should be organized in the following structure:

/path/to/your/data/
├── img_dir/
│   ├── train/                # training images
│   ├── val/                  # validation images
│   └── Generalizability/     # test images
└── ann_dir/
    ├── train/                # training masks
    ├── val/                  # validation masks
    └── Generalizability/     # test masks




2. Configure the Data Path:
Open the dataset configuration file: configs/_base_/datasets/uav_acacia_dataset.py. Find the data_root variable at the top and change the path to point to your main data folder:

# IMPORTANT: Users must edit this path to point to their dataset location.
data_root = '/path/to/your/data'


Training

All commands should be run from the root of the mmsegmentation folder.

- Train the U-MV-Tiny model:

python tools/train.py configs/mambavision/unet_mambavision_tiny.py


- Train the U-MV-Small model:
python tools/train.py configs/mambavision/unet_mambavision_small.py


- Train the U-MV-Base model:

python tools/train.py configs/mambavision/unet_mambavision_base.py




Inference on Large-Scale UAV Images

We provide an optimized script for running seamless inference on large GeoTIFFs, which generates vector outputs (GeoPackage or Shapefile).





#Basic Example:
Download a pre-trained model from the Releases section. Then, run the following command from the root of the mmsegmentation folder:


python tools/geospatial_inference.py \
    configs/mambavision/unet_mambavision_small.py \
    /path/to/your/downloaded_small_model.pth \
    /path/to/large_uav_image.tif \
    /path/to/output/folder/prediction_results \
    --gpkg

Use python tools/geospatial_inference.py --help to see all available options, such as changing the tile size, overlap, or vectorization threshold.
