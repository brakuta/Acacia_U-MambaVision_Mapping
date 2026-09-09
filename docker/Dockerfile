# ---------------------------------------------------------------------------
# U-MV (U-shaped MambaVision) - reproducible GPU environment
#
# Base image: the official PyTorch 2.6.0 / CUDA 11.8 / cuDNN 9 *devel* image.
# The devel variant ships nvcc, which is required to compile mmcv (full build
# with CUDA ops) and mamba-ssm (selective-scan kernels).  This mirrors the
# environment in which the published experiments were run
# (docs/reference/environment.frozen.yml: torch 2.6.0+cu118, Python 3.11).
#
# Build (from the repository root, expect 30-60 min for the compiled parts):
#   docker build -t umv:latest -f docker/Dockerfile .
# Run:
#   docker compose -f docker/docker-compose.yml run --rm umv
# ---------------------------------------------------------------------------
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Dubai \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    # architectures: Turing (TITAN RTX 7.5), Ampere (A5000 8.6, A100 8.0), Ada (8.9), Hopper (9.0)
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    MAX_JOBS=8 \
    HF_HOME=/workspace/hf_cache

# --- system packages -------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential ninja-build ca-certificates curl \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- geospatial stack via conda-forge -------------------------------------
# GDAL Python bindings are not reliably pip-installable; conda-forge provides a
# consistent GDAL/PROJ/GEOS/rasterio/geopandas set (versions as in the frozen env).
RUN conda install -y --override-channels -c conda-forge \
        "python=3.11" "gdal=3.10" "rasterio=1.4" "geopandas=1.0" "shapely=2.1" \
        "pyproj=3.7" "pyogrio" "numpy=2.2" \
    && conda clean -afy

# --- OpenMMLab: mmengine (pure python) + mmcv (compiled from source) --------
RUN pip install mmengine==0.10.7 && \
    pip install --no-build-isolation -v mmcv==2.1.0 && \
    python -c "import mmcv, mmcv.ops; print('mmcv', mmcv.__version__, 'ops OK')"

# --- Mamba selective-scan kernels ------------------------------------------
RUN pip install packaging ninja && \
    pip install --no-build-isolation mamba-ssm==2.2.4 && \
    python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('mamba_ssm OK')"

# --- project dependencies ---------------------------------------------------
WORKDIR /workspace/U-MV
COPY requirements ./requirements
RUN pip install -r requirements/core.txt

# --- project code (editable) -----------------------------------------------
COPY . .
RUN pip install --no-deps -e . && python -c "import umv; print('umv', umv.__version__)"

RUN mkdir -p /data /workspace/hf_cache /workspace/U-MV/work_dirs
VOLUME ["/data", "/workspace/hf_cache", "/workspace/U-MV/work_dirs"]
CMD ["/bin/bash"]
