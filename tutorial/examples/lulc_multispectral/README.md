# Example 4 — Land use / land cover from 10-band Sentinel-2 tiles (multi-class, 16-bit)

What changes when the input is not 8-bit RGB:

1. Loading: `LoadRasterioImage` (rasterio) with band selection and reflectance
   scaling replaces `LoadImageFromFile` (OpenCV, 8-bit, 3 bands).
2. Augmentation: `PhotoMetricDistortion` is removed (it assumes 8-bit BGR);
   geometric augmentations remain.
3. Normalisation: `mean`/`std` have one value per band and `bgr_to_rgb=False`.
4. Model: `backbone.in_channels=10`; the pretrained first convolution is
   adapted with `tools/adapt_first_conv.py`, or trained from scratch.
5. Visualisation hooks expect 3 bands; keep `draw=False` or provide RGB copies.
