# 12. Team workflow

## 12.1 Project structure

```
U-MV-Acacia-tortilis-Crown-Mapping/          (the shared repository)
├── projects/<name>/                         one folder per project, committed
│   ├── configs/_base_/{dataset.py, schedule.py}
│   ├── configs/<model>.py ...
│   ├── scripts/run_experiments.sh
│   └── README.md                            data source, classes, GSD, decisions, results table
├── work_dirs/<name>/<model>/                outputs, never committed
└── /data/<name>/                            tiles, never committed
```

Start a project with `cp -r tutorial/templates/project projects/<name>` and
commit the configs at every meaningful change. A config is a complete
description of an experiment; a result without its config is not
reproducible.

## 12.2 The experiment record

Keep one Markdown table per project (`projects/<name>/README.md`), one row
per run, filled from `compare_runs.py`:

| Run (work dir) | Config commit | Data version | Model | Crop / batch / iters | Val mIoU (best iter) | Test mIoU / mF | Notes |
|---|---|---|---|---|---|---|---|
| buildings/segformer_b2 | a1b2c3d | tiles_v2 (2026-09-01) | SegFormer-B2 | 512 / 4 / 40k | 81.2 (36k) | 80.1 / 88.7 | CE+2·Dice |

Data version means the `tiles_index.csv` and the tiling command; store both
with the dataset.

## 12.3 Sharing results

* Checkpoint of record: `best_mIoU_iter_*.pth`, copied with its resolved
  config and the run's `scalars.json` to the shared weights folder under
  `<project>/<model>/`. Publish with `tools/misc/publish_model.py` when a
  checkpoint leaves the team (removes optimiser state, adds a hash).
* Maps: GeoPackages with `mean_prob` so that thresholds can be revisited.
* Report figures: validation curves (`analyze_logs.py`), the comparison table,
  overlays of representative and failure cases.

## 12.4 Supervision milestones for a student project

| Week | Deliverable | Evidence |
|---|---|---|
| 1 | environment verified; dataset tiled and checked | `verify_install.py` and `check_dataset.py` outputs |
| 2 | smoke test and first full run of one baseline | log, validation curve |
| 3–4 | three models compared under identical settings | `summary.md`, parameters/FLOPs |
| 5 | error analysis and one improvement round | confusion matrix, worst-tile gallery, before/after table |
| 6 | held-out region evaluation and a mapped orthomosaic | test table, GeoPackage in QGIS |
| 7 | written report with the experiment record | `projects/<name>/README.md` |

Upgrading MMSegmentation or PyTorch means rebuilding the image, rerunning
`tests/` and re-evaluating U-MV-small on `test2` (85.4 % mIoU). Ask for help
with the command, the config, the log tail and `verify_install.py` output.
