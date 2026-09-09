# 2. The team environment

## 2.1 The standard setup

All segmentation work in the team runs inside one Docker image on Windows 11
workstations with WSL2 (Ubuntu) and an NVIDIA GPU (TITAN RTX or RTX A5000,
24 GB). The image is defined in `docker/Dockerfile` of this tutorial and
contains the complete stack:

| Layer | Content |
|---|---|
| base | `pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel` (PyTorch, CUDA 11.8, nvcc) |
| OpenMMLab | MMEngine 0.10.7, MMCV 2.1.0 (compiled), MMSegmentation 1.2.2 |
| encoders | the MMSegmentation zoo; timm for `TIMMBackbone` |
| geospatial | GDAL 3.10, rasterio, geopandas, shapely, pyproj (conda-forge) |
| tutorial code | the `geoseg` package (multi-band loader, tiling and inference helpers) and `tools/` |

Working inside a container has one purpose: everyone runs the same versions,
so a config that trains on one workstation trains identically on another, and
results can be compared across people and years.

## 2.2 One-time installation

1. Windows: install the NVIDIA driver (≥ 520), Docker Desktop with the WSL2
   backend, and enable the Ubuntu distribution in *Settings → Resources → WSL
   integration*. Set `memory=48GB` in `%USERPROFILE%\.wslconfig` and run
   `wsl --shutdown` once.
2. Ubuntu (WSL2) terminal:

```bash
cd ~
git clone <tutorial repository URL> mmseg-geospatial-tutorial
cd mmseg-geospatial-tutorial
cp docker/.env.example docker/.env      # edit DATA_DIR and WEIGHTS_DIR (next section)
docker compose --env-file docker/.env -f docker/docker-compose.yml build   # 30-50 min once
```

3. Verify:

```bash
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm mmseg
python tools/verify_install.py --config templates/project/configs/segformer_b2.py
python -m pytest tests -q                 # all passed
```

**Checkpoint.** `verify_install.py` prints `CUDA available: True` with the GPU
name, `mmcv.ops ... ok`, a forward pass `(1, 3, 512, 512) -> (1, 2, 512, 512)`
and `RESULT: OK`.

Without Docker (conda, Linux or WSL2), install in this order: PyTorch 2.6.0
CUDA 11.8 wheels; `mmengine==0.10.7`; `mmcv==2.1.0` compiled from source
(`MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install --no-build-isolation mmcv==2.1.0`,
needs `nvcc`); `conda install -c conda-forge gdal rasterio geopandas`;
`pip install -r requirements.txt`; `pip install --no-deps -e .`. A CPU-only
laptop can run the tests, config checks and small inference jobs but not
training.

## 2.3 Where data and projects live

The container sees three locations:

| Inside the container | Host (set in `docker/.env`) | Purpose |
|---|---|---|
| `/workspace/tutorial` | the tutorial clone | code, `projects/`, `work_dirs/` (training outputs) |
| `/data` | `DATA_DIR` | datasets: one sub-folder per project, e.g. `/data/acacia`, `/data/buildings` |
| `/weights` (read-only) | `WEIGHTS_DIR` | shared pretrained encoders and archived checkpoints |

Recommended host layout on the local SSD (never train from a network share):

```
D:\DL_Data\                        <- DATA_DIR = /mnt/d/DL_Data
├── acacia\{img_dir, ann_dir}\{train, val, test2, Generalizability}
├── buildings\{img_dir, ann_dir}\{train, val, test}
└── lulc_s2\{img_dir, ann_dir}\{train, val, test}
D:\DL_Weights\                     <- WEIGHTS_DIR = /mnt/d/DL_Weights
├── pretrained\resnet50_v1c-2cccc1ad.pth, mit_b2_20220624-66e8bf70.pth
└── acacia\segformer_b2\best_mIoU_iter_36000.pth
```

Project configs live under `projects/<name>/` (created from
`templates/project`), and outputs under `work_dirs/<name>/<model>/`. Commit
`projects/*/configs`; never commit `work_dirs/` or data.

## 2.4 Daily use

```bash
cd ~/mmseg-geospatial-tutorial
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm mmseg
# ... work ...
exit
```

Long trainings should run inside `tmux` so that closing the terminal does not
stop them: `tmux new -s train`, start the job, detach with Ctrl+B then D,
re-attach with `tmux attach -t train`. A second shell into a running container:
`docker exec -it <container id> bash` (`docker ps` lists ids).

Useful environment variables (set in `docker/.env` or before a command):

| Variable | Effect |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | select the GPU on multi-GPU hosts |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | reduces fragmentation-related OOM (set by compose) |

## 2.5 Hardware budget

| Configuration | Typical VRAM | Notes |
|---|---|---|
| DeepLabV3+ R50, 512², batch 4 | ~10 GB | fp32 |
| SegFormer-B2, 512², batch 4 | ~8 GB | |
| UPerNet R50, 1024², batch 2 | ~18 GB | |
| SegFormer-B2, 1024², batch 2 | ~11 GB | |

`--amp` (automatic mixed precision) roughly halves activation memory at no
accuracy cost for these models. RAM: keep `num_workers × tile size` in mind;
8 workers on 1024² RGB tiles use ~4 GB.

**Pitfall (WSL2).** When GPU memory is exhausted, the NVIDIA driver may fall
back to system memory and freeze WSL2. In the NVIDIA Control Panel set *CUDA –
Sysmem Fallback Policy* to *Prefer No Sysmem Fallback*, and reduce batch size
or use `--amp` instead of relying on fallback.
