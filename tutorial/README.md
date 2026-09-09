# Semantic segmentation of geospatial imagery with MMSegmentation — team tutorial

A self-contained course for students and colleagues who need to train,
evaluate and deploy segmentation models on their own remote-sensing data
(trees, buildings, roads, land cover, ...) with MMSegmentation, using the
team's standard environment and the *Acacia tortilis* dataset as the running
example.

| Part | Chapter | File |
|---|---|---|
| Start here | How to use this tutorial | `chapters/00_how_to_use.md` |
| I. Foundations | 1. Semantic segmentation and the OpenMMLab stack | `chapters/01_foundations.md` |
| | 2. The team environment (WSL2, Docker, GPU) | `chapters/02_environment.md` |
| | 3. Anatomy of MMSegmentation: every folder and its role | `chapters/03_anatomy.md` |
| | 4. Configuration files: inheritance, registries, overrides | `chapters/04_configs.md` |
| II. Data | 5. From GIS layers to training tiles | `chapters/05_data_preparation.md` |
| | 6. Datasets, pipelines and augmentation (RGB, RGB+NIR, multispectral) | `chapters/06_datasets_pipelines.md` |
| III. Models | 7. Models: choosing, adapting and switching architectures | `chapters/07_models.md` |
| | 8. Training: schedules, hooks, multiple runs, monitoring | `chapters/08_training.md` |
| | 9. Evaluation: metrics, comparison, error analysis | `chapters/09_evaluation.md` |
| IV. Use | 10. Prediction and export to GIS | `chapters/10_prediction.md` |
| | 11. Worked examples: acacia, buildings, roads, multispectral land cover | `chapters/11_worked_examples.md` |
| | 12. Team workflow: project template, experiment records, sharing | `chapters/12_team_workflow.md` |
| Appendices | A. Cheat-sheets · B. Troubleshooting · C. Glossary · D. Exercises | `chapters/A_cheatsheets.md` ... |

Companion material in this repository:

* `tutorial/templates/project/` — copy to start a new project (dataset base, schedule, five model configs, experiment runner).
* `tutorial/examples/` — four complete, tested example projects.
* `tools/` — `check_dataset.py`, `compare_runs.py`, `band_statistics.py`, `adapt_first_conv.py`, `download_zoo_weights.py`, `geospatial_inference.py`, `batch_geospatial_inference.py`, `inspect_checkpoint.py`, `verify_install.py`.
* `umv/datasets/transforms.py` — `LoadRasterioImage`, the multi-band loader used in chapter 6.

The complete tutorial as one document: `docs/MMSegmentation_Geospatial_Tutorial.pdf` (rebuild with `python tools/build_tutorial_pdf.py`).
