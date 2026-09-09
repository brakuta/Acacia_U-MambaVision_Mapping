# 8. Hand-over guide for the receiving team member

This chapter is self-contained. Follow it from top to bottom; every step
names a command and the outcome that confirms success. The receiving team
member has read access to the project share (`Z:`) where the archive
`A.tortilis_Data & Model` resides.

## What you receive

| Item | Where | What it is |
|---|---|---|
| Code | https://github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping | Model, configs, Docker build, tools, tests and this guide (`docs/`). |
| Dataset | `Z:\...\A.tortilis_Data & Model\Data used to build the model` | `img_dir/` and `ann_dir/` with `train`, `val`, `test2`, `Generalizability` (1024 × 1024 tiles, ~32 GB). |
| Trained models | `Z:\...\A.tortilis_Data & Model\A.tortilis Models\Pretrained weights` | Three MMSegmentation work directories with the checkpoints, training configs and logs (~2.5 GB). |

The archive also contains `A.tortilis Models\MMsegmentation Folder`; that is
the superseded container workspace and is **not** needed.

Layout of the two folders to copy:

```
Data used to build the model/                <- DATA_DIR
├── img_dir/{train, val, test2, Generalizability}/*.tif   RGB tiles
└── ann_dir/{train, val, test2, Generalizability}/*.tif   masks: 0 = background, 1 = acacia

Pretrained weights/                          <- WEIGHTS_DIR
├── mambavision-t_generic-unet_acacia/       best_mIoU_iter_100000.pth  (U-MV-tiny)
├── mambavision-s_generic-unet_acacia-88/    best_mIoU_iter_95000.pth   (U-MV-small)
└── mambavision-b_generic-unet_acacia/       best_mIoU_iter_60000.pth   (U-MV-base)
    each with: mambavision-*_generic-unet_acacia.py (training config), iter_100000.pth,
               <timestamp>/ (logs, vis_data/scalars.json)
```

Facts worth knowing before starting:

* **Which weights.** Use `best_mIoU_iter_*.pth` (checkpoint selected on
  validation mIoU; used for the paper's test results). The files on GitHub
  (`Pretrained_Weights/U-MV-*_latest.pth`, Git LFS) are byte-identical copies,
  so `git lfs pull` is unnecessary when the archive is available. Every tool
  accepts the work-directory folder in place of a file and selects the best
  checkpoint automatically.
* **Which model.** Start with U-MV-small: it produced the regional maps and
  has the best generalisability figures. Tiny is fastest; base is largest.
* **Dataset.** Tile counts in the archive are 4 893 (train), 2 407 (val),
  3 123 (test2), 2 162 (Generalizability). The published models were trained
  on 26 615 tiles; the training folder was pruned after training. Evaluation
  and inference are therefore exact; retraining reproduces the recipe but not
  the published model.
* **Split names.** The in-distribution test split is `test2`; commands pass
  `--test-split test2`.
* **Confidentiality.** The dataset is not publicly shareable. Do not
  redistribute it.

## Prerequisites

* Windows 11 with WSL2 (Ubuntu) and Docker Desktop with the WSL2 backend and
  GPU support, or a Linux host with Docker and the NVIDIA container toolkit.
* NVIDIA GPU with ≥ 16 GB memory (24 GB used for the published runs) and a
  driver ≥ 520.
* ~40 GB free on a local SSD for the two folders, ~25 GB for the Docker image.
* Internet access during the image build and the first model build (the
  MambaVision backbone is downloaded once from the Hugging Face Hub).

## Procedure

Windows paths seen from WSL2 have the form `/mnt/<drive>/...`; paths with
spaces must be quoted. Replace `H:` with your local disk.

### Copy the two folders to a local disk (PowerShell)

Reading tiles over the network share is too slow for training and
evaluation. Copy once, excluding ArcGIS side-car files. Run the
commands one at a time (some consoles reorder multi-line pastes):

```powershell
$Z = "Z:\Final Geodatabase\Vegetation_Geodatabase\3_Mapping Acacia tortilis Trees\A.tortilis_Data & Model"
$H = "H:\A.tortilis_Data & Model"
$D = "Data used to build the model"; $W = "A.tortilis Models\Pretrained weights"
robocopy "$Z\$D" "$H\$D" /E /XF *.aux.xml *.ovr *.xml /MT:16 /R:2 /W:5
robocopy "$Z\$W" "$H\$W" /E /MT:16 /R:2 /W:5
```

Expected: in each summary, *Failed* = 0 and *Copied* + *Skipped* = *Total*.

### Get the code and point it at the folders (WSL2 terminal)

```bash
cd ~
git clone https://github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping.git
cd U-MV-Acacia-tortilis-Crown-Mapping
cp docker/.env.example docker/.env
nano docker/.env
```

Set the two lines (keep the quotes), save with Ctrl+O, Enter, exit with Ctrl+X:

```
DATA_DIR="/mnt/h/A.tortilis_Data & Model/Data used to build the model"
WEIGHTS_DIR="/mnt/h/A.tortilis_Data & Model/A.tortilis Models/Pretrained weights"
```

### Build and start the container

```bash
# 1. GPU visible to Docker?
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# 2. build the image (30-60 min), keeping a log
docker compose --env-file docker/.env -f docker/docker-compose.yml build 2>&1 | tee ~/umv_build.log
# 3. start an interactive container
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm umv
```

The build compiles MMCV and mamba-ssm (30–60 min). Expected after `run`: a
prompt in `/workspace/U-MV`; `ls /data` shows `ann_dir img_dir`; `ls /weights`
shows the three `mambavision-*` folders. All remaining commands run inside the
container.

### Verify, validate, evaluate, map

| # | Step | Command | Expected outcome |
|---|---|---|---|
| 1 | Software stack | `python tools/verify_install.py --variant small` | `RESULT: OK`; forward pass `(1, 3, 512, 512) -> (1, 2, 512, 512)` |
| 2 | Cache backbones for offline use | `python tools/download_backbones.py` | three cache paths printed; later `export HF_HUB_OFFLINE=1` |
| 3 | Unit tests | `python -m pytest tests -q` | all passed |
| 4 | Dataset | `python tools/check_dataset.py /data --splits train val test2 Generalizability` | `RESULT: OK`; pairs 4 893 / 2 407 / 3 123 / 2 162; mask values `[0, 1]` |
| 5 | Checkpoint | `python tools/inspect_checkpoint.py "/weights/mambavision-s_generic-unet_acacia-88"` | resolves `best_mIoU_iter_95000.pth`; `variant: small`; `iteration: 95000` |
| 6 | Test split | `python tools/test.py configs/mambavision/U-MV-small.py "/weights/mambavision-s_generic-unet_acacia-88" --test-split test2` | mIoU ≈ 85.4 %, mFscore ≈ 91.6 % (paper: 85.38 / 91.58) |
| 7 | Generalisability split | same with `--test-split Generalizability` | mIoU ≈ 89.5 %, mFscore ≈ 94.2 % (paper: 89.48 / 94.17) |
| 8 | Other variants | repeat 5–7 with `U-MV-tiny.py` + `mambavision-t_generic-unet_acacia`, `U-MV-base.py` + `mambavision-b_generic-unet_acacia` | tiny 85.44 / 91.61; base 85.30 / 91.52 on `test2` |
| 9 | One orthomosaic | see command below | `.gpkg` with `area`, `mean_prob`; opens in ArcGIS Pro / QGIS in the orthomosaic CRS |
| 10 | Many orthomosaics | `python tools/batch_geospatial_inference.py ... --skip-existing` (see the inference chapter and Appendix C) | mirrored folder of `.gpkg` files + `batch_summary.json` |

Step 9 in full. Place the orthomosaic (GeoTIFF, RGB) under the dataset
folder so that it is visible at `/data`, e.g. `.../Data used to build the
model/orthos/<name>.tif`, then:

```bash
CKPT="/weights/mambavision-s_generic-unet_acacia-88"
python tools/geospatial_inference.py \
  --config configs/mambavision/U-MV-small.py --checkpoint "$CKPT" \
  --input "/data/orthos/<name>.tif" \
  --output "/data/predictions/<name>_crowns.gpkg" \
  --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob
```

Each evaluation run takes 10–20 min on a 24 GB GPU; one 1 km² orthomosaic at
2.5 cm takes a few minutes.

### Optional: retrain

```bash
python tools/train.py configs/mambavision/U-MV-small.py
```

Outputs go to `work_dirs/U-MV-small/` (bind-mounted, persists after exit).
Because the archived training split is a subset, expect validation mIoU below
the published 88.0 %.

## If something fails

Consult the troubleshooting chapter first. When reporting a problem, include
the exact command, the last 50 lines of output, and the result of
`python tools/verify_install.py`.

Contact: Mohamed Barakat A. Gibril (mbgibril@sharjah.ac.ae).
