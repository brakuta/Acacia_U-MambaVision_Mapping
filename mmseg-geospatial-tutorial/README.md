# Semantic segmentation of geospatial imagery with MMSegmentation

A self-contained course and toolkit for training, evaluating and deploying
semantic segmentation models on remote-sensing data (trees, buildings, roads,
land cover, ...) with [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
1.2.2. It depends only on the OpenMMLab packages and a standard geospatial
stack. The *Acacia tortilis* crown tiles of Gibril et al. (2026) serve as the
running example dataset; three further examples cover buildings, roads and
10-band multispectral land cover.

**Read the course:** `docs/MMSegmentation_Geospatial_Tutorial.pdf`
(or the chapters in `chapters/`).

## Contents

| Part | Chapters |
|---|---|
| Foundations | 1 segmentation and the OpenMMLab stack · 2 environment (WSL2, Docker, GPU) · 3 anatomy of MMSegmentation · 4 configuration files |
| Data | 5 from GIS layers to tiles · 6 datasets, pipelines, augmentation (RGB, RGB+NIR, multispectral) |
| Models | 7 choosing, adapting, switching architectures · 8 training · 9 evaluation |
| Use | 10 prediction and GIS export · 11 worked examples · 12 team workflow |
| Appendices | A cheat-sheets · B troubleshooting · C glossary · D exercise solutions |

## What is in the repository

```
geoseg/            LoadRasterioImage (multi-band loader), tiling/vectorisation inference helpers, checkpoint utilities
tools/             train.py, test.py (vendored from MMSegmentation 1.2.2, + --test-split), make_tiles.py, check_dataset.py,
                   band_statistics.py, compare_runs.py, adapt_first_conv.py, download_zoo_weights.py,
                   geospatial_inference.py, batch_geospatial_inference.py, inspect_checkpoint.py, verify_install.py
templates/project/ copy to start a project: dataset base, schedule, four model configs, experiment runner
examples/          acacia_rgb · buildings_rgb · roads_rgb · lulc_multispectral (all configs are built in the tests)
docker/            Dockerfile (PyTorch 2.6 + CUDA 11.8 + MMCV 2.1.0 + MMSegmentation 1.2.2 + GDAL), compose file
chapters/          the course as Markdown; tools/build_tutorial_pdf.py renders docs/…pdf
tests/             CPU tests: configs build and train one step, multispectral pipeline, tiling, vectorisation, tools
```

## Quick start

```bash
cp docker/.env.example docker/.env                      # DATA_DIR: folder with /data/<project> datasets
docker compose --env-file docker/.env -f docker/docker-compose.yml build
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm mmseg
python tools/verify_install.py --config templates/project/configs/segformer_b2.py
python -m pytest tests -q
cp -r templates/project projects/my_project              # then edit projects/my_project/configs/_base_/dataset.py
python tools/train.py projects/my_project/configs/segformer_b2.py --work-dir work_dirs/my_project/segformer_b2
```

## Versions

MMSegmentation 1.2.2 · MMEngine 0.10.7 · MMCV 2.1.0 · PyTorch 2.6.0 (CUDA 11.8) · Python 3.11.
Every signature and tool path in the course was verified against these versions.

## License and citation

Code: MIT. MMSegmentation is Apache-2.0 (`tools/train.py` and `tools/test.py`
are vendored from v1.2.2). Example data: Gibril, M. B. A. et al. (2026),
*Int. J. Appl. Earth Obs. Geoinf.* 148, 105214, doi:10.1016/j.jag.2026.105214
(dataset not publicly shareable).
