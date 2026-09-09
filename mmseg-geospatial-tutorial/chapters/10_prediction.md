# 10. Prediction and export to GIS

## 10.1 Single images from Python

```python
from mmseg.apis import init_model, inference_model, show_result_pyplot
model = init_model('projects/buildings/configs/segformer_b2.py',
                   'work_dirs/buildings/segformer_b2/best_mIoU_iter_36000.pth', device='cuda:0')
result = inference_model(model, '/data/buildings/img_dir/test/b12_000000000012.tif')
mask = result.pred_sem_seg.data[0].cpu().numpy()          # H x W class indices
show_result_pyplot(model, '/data/.../b12_000000000012.tif', result, out_file='pred.png', show=False)
```

`MMSegInferencer(model=config, weights=ckpt)` offers the same for folders
(`inferencer(folder, out_dir='out', pred_out_dir='pred')`). Both use the
config's `test_pipeline`, so they load with OpenCV: for multispectral data use
the geospatial pipeline below or feed arrays through `LoadImageFromNDArray`.

## 10.2 Orthomosaics: the geospatial pipeline

`tools/geospatial_inference.py` (one raster) and
`tools/batch_geospatial_inference.py` (a folder tree) implement sliding-window
inference over gigapixel GeoTIFFs with any MMSegmentation config:

```bash
python tools/geospatial_inference.py \
  --config projects/buildings/configs/segformer_b2.py \
  --checkpoint work_dirs/buildings/segformer_b2 \
  --input /data/orthos/city_block_cog.tif --output /data/predictions/city_block_buildings.gpkg \
  --tile-size 512 --overlap 128 --thresh 0.5 --min-area 4.0 --connectivity 8 --save-prob
```

Steps: edge-aligned tiling with overlap; per-tile softmax of the target
class; centre-crop (seam-free) or Hann blending; probability GeoTIFF in the
input CRS; threshold; GDAL polygonisation; area filter; per-polygon
`mean_prob`; GeoPackage or Shapefile. Parameters and their meaning are in
`docs/05_inference.md`.

For multi-class models the script exports one class (`--class-id`); run it once
per class of interest, or use `--save-prob` per class and combine the
probability rasters in GIS (argmax). Bands other than 8-bit RGB: `--bands` and
the model's normalisation are read from the config's `data_preprocessor`, so a
10-band model works unchanged provided the orthomosaic has the same band
order and scaling as the training tiles.

## 10.3 Post-processing in GIS

| Need | Operation |
|---|---|
| remove specks | `--min-area` (m²) at export, or *Select by Attributes* on `area` |
| keep confident objects | filter on `mean_prob` (e.g. ≥ 0.6) |
| separate touching crowns | watershed on the probability raster (SAGA/GRASS in QGIS), or an instance model |
| smooth building outlines | *Simplify Building* / *Regularize Building Footprint* (ArcGIS Pro) |
| road centrelines | thin the raster (*Thin* tool) and vectorise as lines |
| areas by class | *Tabulate Area* on the argmax raster |

## 10.4 Throughput and memory

Inference memory is set by `--tile-size`, `--batch-size` and precision
(`--precision fp16` default). A 24 GB GPU handles batch 8 of 1024² tiles for
SegFormer-B2. I/O dominates on network drives: stage the input with
`--scratch-dir /tmp/geospatial_work` and convert orthomosaics to COG first.

## Exercise 9

Map one orthomosaic with `--blend center` and `--blend hann`, and compare the
crown outlines along a tile boundary in QGIS (`--overlap 0` shows the seams).
