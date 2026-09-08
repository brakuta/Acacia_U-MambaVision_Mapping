# 5. Regional inference on orthomosaics

`tools/geospatial_inference.py` (one image) and
`tools/batch_geospatial_inference.py` (folder tree) share the implementation in
`umv/inference/`. They turn a georeferenced orthomosaic into a GeoPackage (or
Shapefile) of crown polygons with per-polygon attributes, without ever holding
the full raster in memory.

## 5.1 Processing steps

1. **Staging (optional, `--scratch-dir`).** The input is copied to a fast local
   directory. This is recommended on WSL2/Docker when the data reside on a
   Windows drive (`/mnt/<drive>`), where random window reads are slow.
2. **Tiling.** An edge-aligned grid of `--tile-size` windows with `--overlap`
   pixels is generated; the last tile of each row/column ends exactly at the
   raster border, so every tile has full context.
3. **Prediction.** Tiles are normalised as in training (ImageNet mean/std, RGB)
   and passed through `EncoderDecoder.encode_decode`, yielding logits at tile
   resolution; the softmax probability of the acacia class is kept.
4. **Blending.** `--blend center` (default) writes only the region between the
   mid-points of neighbouring overlaps, so every pixel is predicted exactly once
   from the tile in which it lies farthest from the border (seam-free by
   construction). `--blend hann` averages all overlapping predictions with a
   2-D Hann weight.
5. **Probability raster.** Accumulators are finalised into a tiled, compressed
   GeoTIFF in the CRS of the input (`--save-prob` keeps it; `--prob-dtype uint8`
   stores 0–255).
6. **Vectorisation.** The raster is thresholded at `--thresh`, polygonised with
   GDAL `Polygonize` (streaming, 4- or 8-connectivity), filtered by
   `--min-area` (CRS units, m² for projected CRS) and enriched with `area` and
   `mean_prob` (mean probability inside each polygon, for threshold tuning in
   GIS).

## 5.2 Commands

```bash
python tools/geospatial_inference.py \
  --config configs/mambavision/U-MV-small.py \
  --checkpoint Pretrained_Weights/U-MV-small_latest.pth \
  --input  /data/orthos/Fujairah_block12_cog.tif \
  --output /data/predictions/Fujairah_block12_crowns.gpkg \
  --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob

python tools/batch_geospatial_inference.py \
  --config configs/mambavision/U-MV-small.py \
  --checkpoint Pretrained_Weights/U-MV-small_latest.pth \
  --input-dir /data/orthos --output-dir /data/predictions \
  --scratch-dir /tmp/geospatial_work --skip-existing --continue-on-error
```

The batch script mirrors the input folder structure, skips finished outputs
(`--skip-existing`, resumable) and writes `batch_summary.json`.

## 5.3 Parameters

| Parameter | Default | Guidance |
|---|---|---|
| `--tile-size` | 1024 | Training tile size; keep. |
| `--overlap` | 256 | ≥ 128. Larger overlap = more context at the write boundary, more compute. |
| `--blend` | center | `hann` is marginally smoother on ambiguous crowns at a ~30 % higher cost. |
| `--batch-size` | 8 | 16 fits on 24 GB GPUs in fp16 for U-MV-s; reduce on OOM. |
| `--precision` | fp16 | CUDA autocast; `fp32` for reference comparisons. |
| `--thresh` | 0.35 | Probability threshold used for the regional maps. Tune per region with `mean_prob`; 0.5 is the argmax-equivalent. |
| `--min-area` | 0 | e.g. `1.0` m² removes spurious specks; the mean crown is ~17.5 m². |
| `--connectivity` | 4 | 8 merges diagonally touching pixels. |
| `--bands` | 1 2 3 | Band indices fed as R, G, B (for 4-band RGBA orthomosaics `1 2 3` is still correct). |
| `--band-order` | rgb | **Do not** set `bgr` for standard orthomosaics; see below. |
| `--prob-dtype` | uint8 | `float32` for full-precision probabilities (4× larger). |
| `--num-workers` | 4 | Raster reading processes. |
| `--mp-context` | (torch default) | `spawn` if forked workers misbehave with GDAL. |

## 5.4 Channel order

During training, `LoadImageFromFile` reads TIFFs with OpenCV (BGR order) and
`SegDataPreProcessor(bgr_to_rgb=True)` converts them to RGB before
normalisation; the network therefore expects **RGB**. Rasterio returns bands in
file order, which is RGB for standard orthomosaics, so the inference pipeline
applies no channel swap. The previous versions of the scripts swapped the
channels whenever `bgr_to_rgb=True` was set, which fed BGR tiles to the
network; `--band-order bgr` reproduces that behaviour only for comparison.

## 5.5 Outputs

| File | Content |
|---|---|
| `<name>.gpkg` (layer `<name>`) or `<name>.shp` | Crown polygons; attributes `area` (CRS units), `mean_prob` (0–1). |
| `<name>_prob.tif` (with `--save-prob`) | Single-band probability raster; 0–255 for `uint8`. |
| `batch_summary.json` (batch) | Per-image tiles, polygons, timings, errors. |

Post-processing in GIS: select polygons by `mean_prob`, dissolve touching
crowns if instance separation is not required, and intersect with survey-zone
boundaries.

## 5.6 Throughput

On a TITAN RTX with U-MV-s, fp16 and batch 8, throughput is roughly 4–6 tiles
of 1024 × 1024 per second (≈ 25–40 ha min⁻¹ at 2.5 cm GSD), dominated by
raster I/O when reading from a Windows-mounted drive; staging to `/tmp`
removes that bottleneck.
