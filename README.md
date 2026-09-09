# U-MV: Regional-scale *Acacia tortilis* crown mapping from UAV imagery

**U-MV (U-shaped MambaVision)** is a lightweight hybrid segmentation framework
that couples a MambaVision encoder (Mamba + Transformer + CNN) with a compact
U-Net-style decoder for delineating *Acacia tortilis* crowns in ultra-high
resolution UAV orthomosaics. It is implemented as an extension of
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and ships
training configurations, released checkpoints and geospatial inference tools
that export crown polygons directly to GIS formats.

Official implementation of:

> Gibril, M. B. A., Al-Ruzouq, R., Shanableh, A., Lamghari, F., El-Keblawy, A.,
> Hammouri, N., Makawy, S., Jena, R., Mansour, A., Ghebremeskel, S. Z.,
> Alafayfeh, N. S., Almarzooqi, M. A. (2026). *Regional-scale Acacia tortilis
> crown mapping from UAV remote sensing using semi-automated annotation and a
> lightweight hybrid segmentation framework.* International Journal of Applied
> Earth Observation and Geoinformation, 148, 105214.
> https://doi.org/10.1016/j.jag.2026.105214

<p align="center">
  <img src="Assets/mamba_vision_architecture.png" alt="U-MV architecture" width="900">
</p>

**Figure 1.** U-MV: four-stage MambaVision encoder (strides 4–32) and U-Net
decoder with skip connections, applied to 1024 × 1024 UAV tiles.

## Highlights

- **Three variants** (tiny / small / base; 35.4 M and 54.2 M parameters for
  tiny and small) with released checkpoints; test-set mIoU 85.30–85.44 % and
  mF-score 91.52–91.61 %.
- **Self-contained MMSegmentation extension**: `pip install -e .` registers the
  backbone, decoder and dataset; no files are copied into MMSegmentation.
- **Reproducible environment**: Dockerfile pinned to the original stack
  (PyTorch 2.6.0 + CUDA 11.8, MMCV 2.1.0, MMSegmentation 1.2.2, mamba-ssm 2.2.4).
- **Regional inference**: seam-free sliding-window prediction over gigapixel
  orthomosaics, streaming vectorisation to GeoPackage/Shapefile with per-crown
  `area` and `mean_prob` attributes.

## Repository layout

```
umv/                         installable package (registered with MMSegmentation)
├── models/mamba_vision.py   MambaVisionBackbone (Hugging Face nvidia/MambaVision-*-1K)
├── models/unet_head.py      GenericUNetHead (lightweight U-Net decoder)
├── datasets/uav_acacia.py   UAVAcaciaDataset (background / acacia)
├── inference/               tiling, blending, vectorisation, pipeline, CLI options
└── checkpoint.py            checkpoint inspection and variant detection
configs/
├── _base_/                  models/umv_unet.py · datasets/uav_acacia_dataset.py ·
│                            schedules/schedule_100k_adamw.py · default_runtime.py
└── mambavision/             U-MV-tiny.py · U-MV-small.py · U-MV-base.py
tools/
├── train.py, test.py        MMSegmentation v1.2.2 entry points (+ --test-split)
├── geospatial_inference.py  one orthomosaic -> crown polygons
├── batch_geospatial_inference.py   folder tree -> crown polygons
├── verify_install.py, inspect_checkpoint.py, download_backbones.py, check_dataset.py
docker/                      Dockerfile · docker-compose.yml · run.sh
requirements/                core.txt · geo.txt · mamba.txt · dev.txt
Pretrained_Weights/          released checkpoints (Git LFS) + README
tests/                       CPU unit tests (tiling, vectorisation, pipeline)
docs/                        01 installation … 08 hand-over checklist, REVISION_NOTES
```

## Quick start (Docker, recommended)

```bash
git clone https://github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping.git
cd U-MV-Acacia-tortilis-Crown-Mapping
git lfs install && git lfs pull                       # released checkpoints (~815 MB), optional
cp docker/.env.example docker/.env                    # set DATA_DIR (dataset) and WEIGHTS_DIR (checkpoints)
docker compose --env-file docker/.env -f docker/docker-compose.yml build   # 30-60 min (compiles MMCV, mamba-ssm)
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm umv
# inside the container: dataset at /data, checkpoints at /weights
python tools/verify_install.py --variant small
python tools/check_dataset.py /data --splits train val test2 Generalizability
```

Manual installation (conda, CUDA 11.8 toolkit with `nvcc`) is described in
[docs/01_installation.md](docs/01_installation.md).

## Data layout

```
/data                                    (DATA_DIR on the host)
├── img_dir/{train,val,test,Generalizability}/*.tif   RGB, 8-bit, 1024 x 1024
└── ann_dir/{train,val,test,Generalizability}/*.tif   single-band, 0 = background, 1 = acacia
```

Details, integrity checks and ArcGIS Pro export notes:
[docs/02_data_preparation.md](docs/02_data_preparation.md).

## Training

```bash
python tools/train.py configs/mambavision/U-MV-small.py            # also U-MV-tiny.py, U-MV-base.py
python tools/train.py configs/mambavision/U-MV-small.py --resume   # continue
```

AdamW (lr 1e-4, backbone × 0.1, weight decay 0.05), polynomial decay, 100k
iterations, batch 2 × 1024², CE + 3·Dice loss, best-mIoU checkpointing. See
[docs/03_training.md](docs/03_training.md).

## Evaluation

```bash
python tools/test.py configs/mambavision/U-MV-small.py Pretrained_Weights/U-MV-small_latest.pth
python tools/test.py configs/mambavision/U-MV-small.py Pretrained_Weights/U-MV-small_latest.pth --test-split Generalizability
```

Reports mIoU, mF-score, precision and recall per class
([docs/04_evaluation.md](docs/04_evaluation.md)).

## Regional inference

```bash
python tools/geospatial_inference.py \
  --config configs/mambavision/U-MV-small.py --checkpoint Pretrained_Weights/U-MV-small_latest.pth \
  --input /data/orthos/block12_cog.tif --output /data/predictions/block12_crowns.gpkg \
  --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob

python tools/batch_geospatial_inference.py \
  --config configs/mambavision/U-MV-small.py --checkpoint Pretrained_Weights/U-MV-small_latest.pth \
  --input-dir /data/orthos --output-dir /data/predictions --scratch-dir /tmp/geospatial_work --skip-existing
```

Parameters, channel-order caveat and output attributes:
[docs/05_inference.md](docs/05_inference.md).

## Pretrained weights

| Variant | Encoder | File (Git LFS) | Size |
|---|---|---|---|
| U-MV-tiny | MambaVision-T-1K (80/160/320/640) | `Pretrained_Weights/U-MV-tiny_latest.pth` | 159 MB |
| U-MV-small | MambaVision-S-1K (96/192/384/768) | `Pretrained_Weights/U-MV-small_latest.pth` | 234 MB |
| U-MV-base | MambaVision-B-1K (128/256/512/1024) | `Pretrained_Weights/U-MV-base_latest.pth` | 422 MB |

`python tools/inspect_checkpoint.py <file>.pth` identifies the variant of any
local checkpoint. Checkpoints and training configs from the original
MMSegmentation work directories (`best_mIoU_iter_*.pth`,
`mambavision-*_generic-unet_acacia.py`) are accepted directly by every tool;
legacy configs are translated on load by `umv.compat.load_config`. See
[Pretrained_Weights/README.md](Pretrained_Weights/README.md) and
[docs/08_handover_checklist.md](docs/08_handover_checklist.md).

## Reproducibility and hand-over

- Consolidated hand-over report (PDF, 26 pages): [docs/U-MV_Technical_Handover_Guide.pdf](docs/U-MV_Technical_Handover_Guide.pdf); regenerate with `python tools/build_handover_pdf.py`
- Paper-to-config mapping and known discrepancies: [docs/07_reproducibility.md](docs/07_reproducibility.md)
- Step-by-step hand-over checklist: [docs/08_handover_checklist.md](docs/08_handover_checklist.md)
- Troubleshooting: [docs/06_troubleshooting.md](docs/06_troubleshooting.md)
- What changed in this revision and why: [docs/REVISION_NOTES.md](docs/REVISION_NOTES.md)
- Frozen environment of the original experiments: `docs/reference/environment.frozen.yml`

## Citation

```bibtex
@article{Gibril2026UMV,
  title   = {Regional-scale Acacia tortilis crown mapping from UAV remote sensing using semi-automated annotation and a lightweight hybrid segmentation framework},
  author  = {Gibril, Mohamed Barakat A. and Al-Ruzouq, Rami and Shanableh, Abdallah and Lamghari, Fouad and El-Keblawy, Ali and Hammouri, Nezar and Makawy, Safa and Jena, Ratiranjan and Mansour, Ahmed and Ghebremeskel, Simon Zerisenay and Alafayfeh, Nedal Salem and Almarzooqi, Mohamed Abdulrhaim},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  volume  = {148},
  pages   = {105214},
  year    = {2026},
  doi     = {10.1016/j.jag.2026.105214}
}
```

Code archive: Zenodo, https://doi.org/10.5281/zenodo.18068379.

## License and acknowledgements

Released under the MIT License (see `LICENSE`). Built on
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (Apache-2.0;
`tools/train.py` and `tools/test.py` are vendored from v1.2.2) and
[MambaVision](https://github.com/NVlabs/MambaVision) (NVIDIA Source Code
License-NC for the pretrained encoders, distributed via the Hugging Face Hub).
UAV data were provided by the UAE Ministry of Climate Change and Environment;
the work was supported by the Fujairah Research Centre (Project No. 133049)
and the University of Sharjah.

Contact: Mohamed Barakat A. Gibril, mbgibril@sharjah.ac.ae.
