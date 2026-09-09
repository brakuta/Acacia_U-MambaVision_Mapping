#!/usr/bin/env python3
"""Build the U-MV technical hand-over report (PDF) from the repository docs.

    pip install reportlab            # fonts: Liberation (serif/sans) and DejaVu Sans Mono
    python tools/build_handover_pdf.py [--out docs/U-MV_Technical_Handover_Guide.pdf]

The report consolidates README.md, docs/01-08, docs/REVISION_NOTES.md and
Pretrained_Weights/README.md into one numbered document (see tools/_pdfbook.py).
"""
import datetime

from _pdfbook import (NUM, ROOT, S, Doc, NextPageTemplate, PageBreak, Paragraph, Spacer, chapter, cm,  # noqa: F401
                      heading, image, md_to_flowables, read, toc_block)

VERSION = '1.1'
DATE = datetime.date(2026, 9, 9).strftime('%d %B %Y')
OUT = ROOT / 'docs' / 'U-MV_Technical_Handover_Guide.pdf'

story = []
# cover
story += [Spacer(1, 4.5 * cm),
          Paragraph('U-MV: Regional-scale <i>Acacia tortilis</i> crown mapping from UAV imagery', S['cover_t']),
          Spacer(1, 0.6 * cm),
          Paragraph('Technical hand-over and reproduction guide', S['cover_s']),
          Paragraph('Software, environment, data, training, evaluation and regional inference', S['cover_s']),
          Spacer(1, 1.6 * cm),
          Paragraph(f'Document version {VERSION} · {DATE}', S['cover_m']),
          Paragraph('Repository: github.com/brakuta/U-MV-Acacia-tortilis-Crown-Mapping', S['cover_m']),
          Paragraph('Reference: Gibril et al. (2026), Int. J. Appl. Earth Obs. Geoinf. 148, 105214, '
                    'doi:10.1016/j.jag.2026.105214', S['cover_m']),
          Spacer(1, 3.5 * cm),
          Paragraph('Prepared by Mohamed Barakat A. Gibril<br/>GIS and Remote Sensing Center, Research Institute of '
                    'Sciences and Engineering, University of Sharjah', S['cover_m']),
          Paragraph('Contact: mbgibril@sharjah.ac.ae', S['cover_m']),
          Spacer(1, 1.2 * cm),
          Paragraph('<i>The dataset described herein is not publicly shareable and is provided for project use only.</i>',
                    S['cover_m'])]
story.insert(0, NextPageTemplate('cover'))
story += [NextPageTemplate('main'), PageBreak()]

story += toc_block()

# 1 purpose
NUM.chapter('1')
story += [PageBreak()] + heading(1, 'Purpose and scope of this document')
story += md_to_flowables("""
This document accompanies the hand-over of the U-MV (U-shaped MambaVision) framework for *Acacia tortilis*
crown mapping. It is written for a team member who has (i) the public repository and (ii) read access to the
project archive `A.tortilis_Data & Model` on the shared drive, which holds the dataset and the original
MMSegmentation work directories with the trained checkpoints. It enables that person to install the software,
verify it, reproduce the published accuracy figures and produce regional crown maps from new UAV orthomosaics
without further assistance.

The document consolidates the repository documentation (`README.md`, `docs/01` to `docs/08`, `docs/REVISION_NOTES.md`)
into one reference. The repository remains the authoritative source; where the two differ, the repository is more
recent.

## How to use this document

- **Chapter 3 is the shortest path**: what to copy, how to configure, and a step table with an expected outcome
  for every step. A reader in a hurry can work from chapter 3 alone.
- Chapter 2 describes the framework; chapters 4 to 8 give the full installation, data, training, evaluation and
  inference reference behind the steps of chapter 3.
- Chapter 9 maps the paper to the configuration files and records the reconciliation with the original
  training logs. Chapter 10 lists failure modes with remedies.
- Appendices record what changed in the code revision of September 2026, describe the released weights, and
  provide a command reference.

## Conventions

Commands are given for a shell inside the Docker container, where the dataset is mounted at `/data` and the checkpoint
folder at `/weights`. Paths on the host are placeholders and must be adapted. Windows paths seen from WSL2 have the form
`/mnt/<drive>/...`; paths containing spaces must be quoted.
""")

# 2 overview
readme = read('README.md')
ov = readme.split('## Highlights', 1)[1].split('## Quick start', 1)[0]
NUM.chapter('2')
story += [PageBreak()] + heading(1, 'Framework overview')
story += md_to_flowables("""
U-MV couples the hierarchical MambaVision encoder of Hatamizadeh and Kautz (2025), a hybrid of Mamba state-space
mixers, self-attention and convolution, with a lightweight U-Net-style decoder. The encoder produces feature maps at
strides 4, 8, 16 and 32; the decoder upsamples the deepest map stage by stage, concatenating skip features and applying
two 3 × 3 convolution–BatchNorm–ReLU blocks per stage, and a 1 × 1 convolution yields two-class logits. Three encoder
sizes are used: tiny (80/160/320/640 channels), small (96/192/384/768) and base (128/256/512/1024). Training minimises
cross-entropy plus three times the Dice loss on 1024 × 1024 UAV tiles at 2.5 cm ground sampling distance.
""")
img = image(ROOT / 'Assets/mamba_vision_architecture.png')
story += [img, Paragraph('Figure 1. U-MV architecture: four-stage MambaVision encoder (top) and U-Net decoder with '
                         'skip connections applied to a 1024 × 1024 UAV tile.', S['caption'])]
story += md_to_flowables('## Highlights' + ov)
story += md_to_flowables("""
## Published accuracy

| Model | Validation mIoU | Validation mF-score | Test mIoU | Test mF-score |
|---|---:|---:|---:|---:|
| U-MV-t | 87.91 | 93.20 | 85.44 | 91.61 |
| U-MV-s | 88.02 | 93.27 | 85.38 | 91.58 |
| U-MV-b | 88.11 | 93.32 | 85.30 | 91.52 |

Values in percent (paper, Table 1). On the generalisability set (~11 km², 2 165 tiles) U-MV-s reached 89.48 % mIoU and
94.17 % mF-score. Training on a TITAN RTX (24 GB) with batch size 2 took 9.33 h (tiny), 11.8 h (small) and 18.12 h
(base) for 100 000 iterations.
""")

# 3 hand-over guide (complete procedure)
h08 = read('docs/08_handover_checklist.md').split('\n', 1)[1]  # drop title line
story += chapter('3', 'Hand-over guide: what you receive and how to proceed', h08)

# 4-8
story += chapter('4', 'Software environment and installation', read('docs/01_installation.md'))
story += chapter('5', 'Data', read('docs/02_data_preparation.md'))
story += chapter('6', 'Training', read('docs/03_training.md'))
story += chapter('7', 'Evaluation', read('docs/04_evaluation.md'))
story += chapter('8', 'Regional inference on orthomosaics', read('docs/05_inference.md'))
# 9, 10
story += chapter('9', 'Reproducibility and paper-to-code mapping', read('docs/07_reproducibility.md'))
story += chapter('10', 'Troubleshooting', read('docs/06_troubleshooting.md'))

# appendices
story += chapter('A', 'Revision notes (September 2026)', read('docs/REVISION_NOTES.md'))
story += chapter('B', 'Pretrained weights', read('Pretrained_Weights/README.md'))
story += chapter('C', 'Command reference', """
## Environment

```
cp docker/.env.example docker/.env                 # set DATA_DIR and WEIGHTS_DIR
docker compose --env-file docker/.env -f docker/docker-compose.yml build
docker compose --env-file docker/.env -f docker/docker-compose.yml run --rm umv
python tools/verify_install.py --variant small     # stack check + forward pass
python tools/download_backbones.py                 # cache backbones; then export HF_HUB_OFFLINE=1
python -m pytest tests -q                          # unit tests (CPU)
```

## Data and checkpoints

```
python tools/check_dataset.py /data --splits train val test2 Generalizability
python tools/inspect_checkpoint.py "/weights/mambavision-s_generic-unet_acacia-88"
```

## Training

```
python tools/train.py configs/mambavision/U-MV-small.py
python tools/train.py configs/mambavision/U-MV-small.py --resume --work-dir work_dirs/U-MV-small
python tools/train.py configs/mambavision/U-MV-small.py \\
  --cfg-options randomness.seed=0 randomness.deterministic=True
```

## Evaluation

```
CKPT="/weights/mambavision-s_generic-unet_acacia-88"      # folder resolves to best_mIoU_iter_*.pth
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split test2
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split Generalizability
python tools/test.py configs/mambavision/U-MV-small.py "$CKPT" --test-split test2 \\
  --out preds/ --show-dir vis/
```

## Regional inference

```
CKPT="/weights/mambavision-s_generic-unet_acacia-88"
python tools/geospatial_inference.py \\
  --config configs/mambavision/U-MV-small.py --checkpoint "$CKPT" \\
  --input /data/orthos/block12_cog.tif \\
  --output /data/predictions/block12_crowns.gpkg \\
  --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob

python tools/batch_geospatial_inference.py \\
  --config configs/mambavision/U-MV-small.py --checkpoint "$CKPT" \\
  --input-dir /data/orthos --output-dir /data/predictions \\
  --scratch-dir /tmp/geospatial_work --skip-existing --continue-on-error
```

## Orthomosaic preparation

```
gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512 \\
  -co NUM_THREADS=ALL_CPUS in.tif out_cog.tif
```
""")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=str(OUT))
    from pathlib import Path
    out = Path(ap.parse_args().out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Doc(str(out), header='U-MV · Acacia tortilis crown mapping · Technical hand-over guide',
        footer_right=f'v{VERSION} · {DATE}', title='U-MV Acacia tortilis crown mapping: technical hand-over guide',
        author='Mohamed Barakat A. Gibril', subject='Reproduction and hand-over documentation').multiBuild(story)
    print('written', out, out.stat().st_size // 1024, 'kB')
