# 8. Hand-over checklist

Follow the steps in order; each has a verifiable outcome.

| # | Step | Command / action | Expected outcome |
|---|---|---|---|
| 1 | Clone the repository | `git clone https://github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping.git` | `configs/`, `umv/`, `tools/`, `docker/` present |
| 2 | Fetch the released weights | `git lfs install && git lfs pull` | Files in `Pretrained_Weights/` are 159/234/422 MB, not 134 bytes |
| 3 | Check any local checkpoint | `python tools/inspect_checkpoint.py <file>.pth` (works in any Python with torch) | Prints `variant: small` (or tiny/base) and the iteration |
| 4 | Enable GPU in Docker Desktop (WSL2) | `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` | GPU listed |
| 5 | Build the image | `docker compose -f docker/docker-compose.yml build` (30–60 min) | `umv:latest` in `docker images` |
| 6 | Start a shell | `DATA_DIR=/mnt/h/UAV_Data docker compose -f docker/docker-compose.yml run --rm umv` | Prompt in `/workspace/U-MV` |
| 7 | Verify the stack | `python tools/verify_install.py --variant small` | `RESULT: OK`, forward pass shape `(1, 2, 512, 512)` |
| 8 | Cache backbones for offline use | `python tools/download_backbones.py` | Three entries under `/workspace/hf_cache` |
| 9 | Unit tests | `python -m pytest tests -q` | `14 passed` |
| 9b | Validate the dataset copy | `python tools/check_dataset.py /data --splits train val test2 Generalizability` | `RESULT: OK`; pair counts 26 615 / 2 400 / 3 125 / 2 165 |
| 10 | Evaluate a released checkpoint | `python tools/test.py configs/mambavision/U-MV-small.py Pretrained_Weights/U-MV-small_latest.pth` | mIoU/mFscore within ±0.1 pp of the paper |
| 11 | Map one orthomosaic | `python tools/geospatial_inference.py ... --save-prob` (see `docs/05_inference.md`) | `.gpkg` with `area`, `mean_prob`; opens in QGIS/ArcGIS Pro in the orthomosaic CRS |
| 12 | Batch mapping | `python tools/batch_geospatial_inference.py ... --skip-existing` | Mirrored folder of `.gpkg` files and `batch_summary.json` |
| 13 | (Optional) Retrain | `python tools/train.py configs/mambavision/U-MV-small.py` | `work_dirs/U-MV-small/best_mIoU_iter_*.pth` after ~12 h on a TITAN RTX |

## Original training work directories

The folders `A.tortilis Models\Pretrained weights\mambavision-{t,s,b}_generic-unet_acacia*`
are the MMSegmentation work directories of the published runs. Each contains
the exact training config (`mambavision-*_generic-unet_acacia.py`), timestamped
logs with `vis_data/scalars.json`, `best_mIoU_iter_*.pth` (checkpoint used for
the paper's test results) and `iter_100000.pth` (last iteration). Keep them as
the archival record; for hand-over use `best_mIoU_iter_*.pth` and confirm with
`tools/inspect_checkpoint.py` which iteration a given file holds.

Contacts: corresponding author M. B. A. Gibril (mbgibril@sharjah.ac.ae).
Data are not publicly shareable (paper, Data availability); request access
through the corresponding author.
