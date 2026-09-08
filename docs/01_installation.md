# 1. Installation

Two routes are supported. The Docker route reproduces the environment in which
the published experiments were executed and is the recommended option for
hand-over to new team members. The manual route documents every dependency for
users who cannot run containers.

## 1.1 Software stack and version rationale

| Component | Version | Reason |
|---|---|---|
| Python | 3.11 | Version of the original container (`docs/reference/environment.frozen.yml`). |
| PyTorch / torchvision | 2.6.0 + cu118 / 0.21.0 | Original experiments; CUDA 11.8 wheels run on driver ≥ 520 (TITAN RTX, RTX A5000). |
| MMEngine | 0.10.7 | Handles `torch.load(weights_only=True)` introduced in PyTorch 2.6. |
| MMCV | 2.1.0 (full build) | Latest release accepted by MMSegmentation 1.2.2 (`mmcv>=2.0.0rc4,<2.2.0`). No prebuilt wheel exists for PyTorch 2.6, so it is compiled from source. |
| MMSegmentation | 1.2.2 | Last stable 1.x release; equivalent to the `main` branch used originally. |
| transformers / timm | 4.50.0 / 1.0.15 | Versions with which the MambaVision remote code (`trust_remote_code=True`) is known to import (`timm.models.registry`, `timm.models.layers` shims). |
| mamba-ssm | 2.2.4 | Provides `selective_scan_fn`, the CUDA kernel used by the MambaVision mixer. Requires `nvcc`. |
| GDAL / rasterio / geopandas | 3.10 / 1.4 / 1.0 (conda-forge) | GDAL `Polygonize` is used for streaming vectorisation of gigapixel probability rasters. |

MambaVision does **not** call `causal_conv1d`; that package is optional.

## 1.2 Docker route (recommended)

### Prerequisites on Windows 11 / WSL2

1. Install an NVIDIA driver ≥ 520 on Windows (the driver is shared with WSL2; do not install a driver inside WSL2).
2. Install Docker Desktop with the WSL2 backend and enable the Ubuntu distribution under *Settings → Resources → WSL integration*.
3. Verify GPU visibility from WSL2:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```
4. Grant WSL2 enough memory for the 1024 × 1024 training tiles (in `%USERPROFILE%\.wslconfig`):
   ```ini
   [wsl2]
   memory=48GB
   swap=16GB
   ```
   then `wsl --shutdown` from PowerShell.

### Build

```bash
git clone https://github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping.git
cd U-MV-Acacia-tortilis-Crown-Mapping
git lfs install && git lfs pull            # released checkpoints (~815 MB), optional
docker compose -f docker/docker-compose.yml build
```

The build compiles MMCV and mamba-ssm and typically takes 30–60 min. The
`TORCH_CUDA_ARCH_LIST` variable in `docker/Dockerfile` covers Turing (TITAN RTX,
7.5), Ampere (RTX A5000, 8.6; A100, 8.0), Ada (8.9) and Hopper (9.0); extend it
for other GPUs before building.

### Run

```bash
export DATA_DIR=/mnt/h/UAV_Data           # host folder mounted at /data
docker compose -f docker/docker-compose.yml run --rm umv
# inside the container
python tools/verify_install.py --variant small
```

The compose file bind-mounts the repository at `/workspace/U-MV`, the dataset at
`/data`, a named volume for the Hugging Face cache and `work_dirs/` for training
outputs. `docker/run.sh` offers the same without compose.

### Offline use

The MambaVision code and ImageNet weights are downloaded once from the
Hugging Face Hub into the `hf_cache` volume:

```bash
python tools/download_backbones.py --variants tiny small base
export HF_HUB_OFFLINE=1      # subsequent runs need no network
```

## 1.3 Manual route (conda)

Install in this order; later steps compile against earlier ones.

```bash
conda create -n umv python=3.11 -y && conda activate umv
# 1. PyTorch (CUDA 11.8 wheels)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
# 2. MMEngine and MMCV (compiled; requires nvcc from CUDA 11.8 toolkit on PATH)
pip install mmengine==0.10.7
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="7.5;8.6" pip install --no-build-isolation mmcv==2.1.0
# 3. Mamba kernels
pip install packaging ninja && pip install --no-build-isolation -r requirements/mamba.txt
# 4. Geospatial stack (GDAL bindings from conda-forge)
conda install -c conda-forge gdal=3.10 rasterio=1.4 geopandas=1.0 shapely=2.1 pyproj=3.7 pyogrio -y
# 5. Project
pip install -r requirements/core.txt
pip install --no-deps -e .
python tools/verify_install.py
```

`pip install -e .` makes `umv` importable from any working directory; the
scripts in `tools/` also work without installation because they add the
repository root to `sys.path`.

## 1.4 Verification

```bash
python tools/verify_install.py                    # imports, CUDA, mamba_ssm, mmcv.ops
python tools/verify_install.py --variant small    # builds U-MV-small, forward pass, peak VRAM
python -m pytest tests -q                         # tiling / vectorisation unit tests (CPU)
```

Expected: `RESULT: OK`, and for the second command a forward pass of a
512 × 512 tensor producing logits of shape `(1, 2, 512, 512)`.
