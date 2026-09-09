# 2. Data preparation

## 2.1 Source data

The framework was developed on RGB orthomosaics acquired with a senseFly eBee X
(S.O.D.A. camera) at ~122 m AGL, processed in Pix4Dmapper to a ground sampling
distance (GSD) of 2.5–3 cm. Annotations were produced with the semi-automated
workflow described in Section 3.1 of the paper (SAM-LoRA crown proposals,
Mask2Former–Swin candidate detection, visual verification).

## 2.2 Tiles and masks

Orthomosaics and rasterised crown polygons were partitioned into 1024 × 1024
pixel image–mask pairs. This tile size was chosen so that a mean crown
(~17.5 m², ~5 m diameter, ~28 000 pixels at 2.5 cm) is embedded in sufficient
context (shadows, terrain transitions, neighbouring land cover).

Requirements for new data:

| Item | Specification |
|---|---|
| Images | 3-band RGB, 8-bit, `.tif` (GeoTIFF or plain TIFF). Other suffixes: set `img_suffix` in the dataset config. |
| Masks | Single-band 8-bit index raster, `0 = background`, `1 = acacia`, `.tif` (or `.png` with `seg_map_suffix='.png'`). Not RGB, not 0/255. |
| Names | Identical basenames for image and mask within a split. |
| Size | 1024 × 1024 recommended; other sizes work (the pipeline resizes to 1024 on the long side and random-crops 1024 × 1024 during training). |
| Ignore value | 255 in a mask is ignored by the loss (`seg_pad_val=255`). |

Directory layout (mounted at `/data` in the container):

```
/data
├── img_dir/
│   ├── train/             4 893 pairs in the archive (26 615 used for the published models)
│   ├── val/               2 407
│   ├── test2/             3 123   (in-distribution test)
│   └── Generalizability/  2 162   (out-of-distribution test, separate region)
└── ann_dir/
    ├── train/  val/  test2/  Generalizability/
```

The archived training folder was pruned after the published runs (see the
reproducibility chapter); the other three splits are complete.

A different root is selected without editing configs:

```bash
python tools/train.py configs/mambavision/U-MV-small.py \
  --cfg-options train_dataloader.dataset.data_root=/data/UAV_Acacia \
                val_dataloader.dataset.data_root=/data/UAV_Acacia \
                test_dataloader.dataset.data_root=/data/UAV_Acacia
```

## 2.3 Integrity check

```bash
python tools/check_dataset.py /data --splits train val test2 Generalizability
```

reports pair counts, tile sizes, dtypes, mask values and ArcGIS side-car files
(`*.tif.aux.xml`, `*.tif.ovr`) per split. Side-cars are created when tiles are
opened in ArcGIS Pro; the loader ignores them (only files ending in `.tif` are
listed), but they can be deleted from working copies to save space.

A split whose folder is not named `test` (for example `test2`) is evaluated
without renaming: `python tools/test.py <config> <ckpt> --test-split test2`.

### Manual check

```python
import numpy as np, rasterio, glob, os
root = '/data'
for split in ['train', 'val', 'test2', 'Generalizability']:
    imgs = sorted(glob.glob(f'{root}/img_dir/{split}/*.tif'))
    anns = sorted(glob.glob(f'{root}/ann_dir/{split}/*.tif'))
    assert [os.path.basename(p) for p in imgs] == [os.path.basename(p) for p in anns], split
    with rasterio.open(imgs[0]) as im, rasterio.open(anns[0]) as an:
        assert im.count == 3 and im.dtypes[0] == 'uint8', im.profile
        assert an.count == 1 and set(np.unique(an.read(1))) <= {0, 1, 255}, an.profile
        assert (im.width, im.height) == (an.width, an.height)
    print(split, len(imgs), 'pairs OK')
```

## 2.4 Export from ArcGIS Pro (summary)

1. Rasterise the verified crown polygons to the orthomosaic grid (`Polygon to
   Raster`, cell size = orthomosaic cell size, value field = 1, NoData → 0).
2. Export image and mask tiles with the same fishnet (`Split Raster`, tile size
   1024 × 1024, format TIFF, no overlap for training tiles).
3. Convert masks to 8-bit unsigned (`Copy Raster`, pixel type 8_BIT_UNSIGNED)
   and verify with the script above.

## 2.5 Location of the dataset

The dataset folder can reside anywhere; its path is supplied once in
`docker/.env` (`DATA_DIR`) or on the command line
(`--cfg-options train_dataloader.dataset.data_root=...`). Prefer a local SSD
over a network share, because training performs ~30 000 random tile reads per
epoch. When copying, ArcGIS side-cars can be excluded:

```powershell
robocopy "<source>\Data used to build the model" "D:\UAV_Acacia_Data" /E /XF *.aux.xml *.ovr *.xml /MT:16
```

If a network drive must be read directly from WSL2, mount it first
(`sudo mkdir -p /mnt/z && sudo mount -t drvfs Z: /mnt/z`) and quote the path.

## 2.6 Orthomosaics for regional inference

Inference operates on whole orthomosaics. Convert them to Cloud-Optimised
GeoTIFF (COG) once; windowed reads are then efficient from any storage:

```bash
gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512 -co NUM_THREADS=ALL_CPUS in.tif out_cog.tif
```
