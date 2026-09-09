# Revision notes (September 2026)

This revision prepares the repository for hand-over and independent
reproduction. The network architecture, released configurations and
checkpoints are unchanged; the changes concern packaging, correctness of the
auxiliary code, portability and documentation.

## Defects found in the previous revision

| # | Defect | Effect | Resolution |
|---|---|---|---|
| 1 | `mmseg/dataset/ade.py` (singular `dataset`) was meant to overwrite MMSegmentation's `mmseg/datasets/ade.py`; the copy-in instructions never placed it there. | Configs referenced `ADE20KDataset`, which expects `.jpg` images and 150 classes; a fresh installation could not load `.tif` tiles or report two-class metrics. | New registered dataset `UAVAcaciaDataset` in `umv/datasets/`; no upstream file is modified. |
| 2 | `img_ratios` undefined in the dataset config. | `Config.fromfile` raised `NameError` on every config. | Defined. |
| 3 | `configs/_base_/default_runtime.py` absent from the repository. | Configs unloadable outside a full MMSegmentation checkout. | Added (verbatim from v1.2.2). |
| 4 | Inference scripts swapped RGB→BGR whenever `bgr_to_rgb=True`. | Network received BGR tiles although it was trained on RGB. | Removed; `--band-order bgr` retained for comparison. |
| 5 | Batch script used `mode='whole'` with `ori_shape=(h, w)` for border tiles. | Padded border tiles were resized to the valid size, misaligning predictions along raster edges. | Both scripts now use `encode_decode` at tile resolution and crop. |
| 6 | Centre-crop writer double-wrote border regions when a raster edge fell inside the first tile of a row. | Silent averaging at borders. | Edge-aligned grid with mid-point write boundaries; property-tested (`tests/test_tiling.py`). |
| 7 | Minimum-area filtering documented in the README but not implemented. | — | `--min-area` implemented; `area` attribute added. |
| 8 | Hard-coded paths, thresholds and device settings inside the scripts. | Edits required for every run. | argparse CLI with documented defaults. |
| 9 | `num_workers=42` in the dataset config. | Failure or thrashing on smaller hosts. | 8 (documented). |
| 10 | Backbone always downloaded ImageNet weights, even when loading a fine-tuned checkpoint; no offline path. | Network access mandatory at inference time. | `pretrained=False` when a checkpoint is given; `HF_HUB_OFFLINE` / `local_files_only` supported; `tools/download_backbones.py`. |
| 11 | `environment.yml` was a full conda dump of the container's base environment (`prefix: /opt/conda`, 300 packages). | Not recreatable; misleading as a specification. | Moved to `docs/reference/environment.frozen.yml`; replaced by `requirements/*.txt`, `pyproject.toml` and `docker/Dockerfile`. |
| 12 | `Assets/mamba_vision_architecture.png` was 92 MB (14 639 × 14 198 px). | Slow clones, README rendering issues. | Downscaled to 2 400 px (2.5 MB); the original remains in Git history. |
| 13 | `.idea/workspace.xml` committed. | IDE noise. | Removed and ignored. |
| 14 | `tqdm==4.65.2`, `torch==2.6.0+cu118` pins in `requirements.txt` not installable from PyPI without extra index. | `pip install -r requirements.txt` failed. | Split requirements; PyTorch/MMCV/mamba-ssm installed explicitly. |

## Structural changes

* `umv/` installable package (`pip install -e .`) with `models/`, `datasets/`,
  `inference/`, `checkpoint.py`. Registered names `GenericUNetHead`,
  `mamba_{tiny,small,base}_vision_timm` are kept, and `MambaVisionBackbone`
  is added, so configs embedded in old checkpoints still resolve.
* Module attribute names (`backbone.backbone.*`, `decode_head.decoder_stages.*`,
  `decode_head.segmentation_head.*`, `decode_head.conv_seg.*`) are preserved;
  released checkpoints load without key remapping.
* Configs split into `_base_/models`, `_base_/datasets`, `_base_/schedules`,
  `_base_/default_runtime.py`; variant files contain only variant-specific
  fields. Numerical hyper-parameters are unchanged.
* `tools/train.py` and `tools/test.py` vendored from MMSegmentation v1.2.2
  (plus `--test-split`), so the repository is self-contained.
* New tools: `verify_install.py`, `inspect_checkpoint.py`,
  `download_backbones.py`, `batch_geospatial_inference.py` (renamed from
  `Batch_processing_geospatial_inference.py`).
* `tests/` (14 tests) covering tiling, blending, vectorisation, checkpoint
  variant detection and the end-to-end inference pipeline on synthetic
  GeoTIFFs (CPU).

## Compatibility with the archived work directories

`umv/compat.py::load_config` detects configs that import
`mmseg.custom_models.*` (the archived training configs) and maps them to the
new package, including `ADE20KDataset` -> `UAVAcaciaDataset`. All tools use
it, so `tools/test.py <archived config> <archived checkpoint>` works without
edits. `docker/.env` (`DATA_DIR`, `WEIGHTS_DIR`) decouples the container from
the location of the hand-over folder.

## Validation performed in this revision

* All three configs load; models build with a stub encoder and run a training
  step (CE + Dice losses), `encode_decode`, and sliding-window prediction on
  CPU (MMEngine 0.10.7, MMSegmentation 1.2.2).
* The inference pipeline recovers synthetic crowns with correct CRS, areas
  (±5 %), attributes and seam-free tiling for both blending modes.
* The Git LFS objects of the three released checkpoints are present on GitHub.

## Not validated here (requires a GPU host)

* The Docker image build (MMCV and mamba-ssm compilation) and the Hugging Face
  backbone instantiation; the Hub was unreachable from the sandbox in which
  this revision was prepared. Run `docs/08_handover_checklist.md` steps 5–10 on
  the workstation and report any deviation.
* Numerical equivalence of the released checkpoints with the paper's metrics.

## Decisions left to the authors

1. Whether to harmonise the tiny configuration with the 100k schedule
   (`docs/07_reproducibility.md`, §7.2).
2. Whether to publish the checkpoints on Zenodo/Hugging Face in addition to
   Git LFS (bandwidth quota of GitHub LFS is 1 GB month⁻¹ on free plans; three
   full downloads exhaust it).
3. Whether regional maps produced with the earlier BGR-swapped scripts should
   be regenerated.
