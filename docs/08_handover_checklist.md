# 8. Hand-over guide for the receiving team member

You receive one folder, `A.tortilis_Data & Model`, and this repository. The
folder can be copied to any location; nothing in the code depends on its
path. It contains:

```
A.tortilis_Data & Model/
├── Data used to build the model/          <- DATA_DIR
│   ├── img_dir/{train, val, test2, Generalizability}/   RGB tiles 1024 x 1024 (.tif)
│   └── ann_dir/{train, val, test2, Generalizability}/   masks 0 = background, 1 = acacia
└── A.tortilis Models/Pretrained weights/  <- WEIGHTS_DIR
    ├── mambavision-t_generic-unet_acacia/
    ├── mambavision-s_generic-unet_acacia-88/
    └── mambavision-b_generic-unet_acacia/
        ├── mambavision-*_generic-unet_acacia.py   exact training config of the published run
        ├── best_mIoU_iter_*.pth                   checkpoint used for the paper's test results
        ├── iter_100000.pth                        last iteration
        └── <timestamp>/                           logs, vis_data/scalars.json
```

Practical notes on the folder:

* Copy it to a **local disk** (SSD) rather than reading it from a network
  share; training reads ~30 000 tiles per epoch. Excluding ArcGIS side-cars
  (`*.tif.aux.xml`, `*.tif.ovr`) during the copy saves space; the code ignores
  them either way.
* The in-distribution test split is named `test2`. Nothing needs renaming;
  evaluation commands pass `--test-split test2`.
* The archived `.py` configs reference the previous code layout. They load
  unchanged through this repository (`umv.compat.load_config`, used by every
  tool), which maps the old module and dataset names to the new package.

## Step-by-step

Paths below assume Windows 11 with WSL2 and Docker Desktop; a Windows folder
`D:\UAV\A.tortilis_Data & Model` is visible in WSL2 as
`/mnt/d/UAV/A.tortilis_Data & Model`. Linux hosts use the path directly.

| # | Step | Command / action | Expected outcome |
|---|---|---|---|
| 1 | Clone the repository | `git clone https://github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping.git && cd U-MV-Acacia-tortilis-Crown-Mapping` | `configs/`, `umv/`, `tools/`, `docker/` present |
| 2 | Tell the container where the folder is | `cp docker/.env.example docker/.env`, then set `DATA_DIR` and `WEIGHTS_DIR` (quoted; see the example file) | — |
| 3 | GPU visible in Docker (WSL2) | `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi` | GPU listed |
| 4 | Build the image (30–60 min) | `docker compose --env-file docker/.env -f docker/docker-compose.yml build` | `umv:latest` in `docker images` |
| 5 | Start a shell | `docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm umv` | Prompt in `/workspace/U-MV`; `/data` and `/weights` mounted |
| 6 | Verify the software stack | `python tools/verify_install.py --variant small` | `RESULT: OK`; forward pass `(1, 2, 512, 512)` |
| 7 | Validate the dataset | `python tools/check_dataset.py /data --splits train val test2 Generalizability` | `RESULT: OK`; pairs 26 615 / 2 400 / 3 125 / 2 165 |
| 8 | Identify a checkpoint | `python tools/inspect_checkpoint.py "/weights/mambavision-s_generic-unet_acacia-88/best_mIoU_iter_95000.pth"` | `variant: small`, `iteration: 95000` |
| 9 | Unit tests | `python -m pytest tests -q` | all passed |
| 10 | Reproduce the published test metrics | `python tools/test.py configs/mambavision/U-MV-small.py "/weights/mambavision-s_generic-unet_acacia-88/best_mIoU_iter_95000.pth" --test-split test2` | mIoU ≈ 85.38 %, mF-score ≈ 91.58 % (Table 1) |
| 11 | Generalisability split | same command with `--test-split Generalizability` | mIoU ≈ 89.48 %, mF-score ≈ 94.17 % |
| 12 | Map one orthomosaic | `python tools/geospatial_inference.py --config configs/mambavision/U-MV-small.py --checkpoint "/weights/.../best_mIoU_iter_95000.pth" --input /data/<ortho>.tif --output /data/predictions/<ortho>_crowns.gpkg --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob` | `.gpkg` with `area`, `mean_prob`, opens in QGIS/ArcGIS Pro in the orthomosaic CRS |
| 13 | Batch mapping | `python tools/batch_geospatial_inference.py ... --skip-existing` (`docs/05_inference.md`) | mirrored folder of `.gpkg` + `batch_summary.json` |
| 14 | (Optional) retrain or resume | `python tools/train.py configs/mambavision/U-MV-small.py` or `python tools/train.py "/weights/mambavision-s_generic-unet_acacia-88/mambavision-s_generic-unet_acacia.py" --work-dir work_dirs/resume_s --resume` | `work_dirs/.../best_mIoU_iter_*.pth` |

Steps 6–13 can also be run without Docker after the manual installation in
`docs/01_installation.md`; replace `/data` and `/weights` with the local paths.

## Which checkpoint to use

`best_mIoU_iter_*.pth` is the checkpoint selected on validation mIoU and used
for the paper's test results; `iter_100000.pth` is the final iteration. Use
the former unless a study specifically requires the last iteration. The
`-88` suffix of the small model's folder denotes its validation mIoU (88.02 %).

## Where results go

Training outputs are written to `work_dirs/` in the repository (bind-mounted;
they persist after the container exits). Inference outputs go wherever
`--output` / `--output-dir` points, normally under `/data`.

Contact: Mohamed Barakat A. Gibril (mbgibril@sharjah.ac.ae). The dataset is
not publicly shareable; do not redistribute it outside the project.
