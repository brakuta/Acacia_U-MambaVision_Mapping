# Appendix D. Exercise solutions

**1.** Batch 2 × 100 000 iterations = 200 000 samples: 7.5 epochs over 26 615
tiles, 40.9 epochs over 4 893. On the smaller set the same iteration budget
over-fits earlier; plan 20–40k iterations and rely on the best-validation
checkpoint, or keep 100k with early stopping.

**2.** DeepLabV3+-R50-d8: backbone `ResNetV1c` with `out_indices=(0,1,2,3)`
and dilations (1,1,2,4), so stage outputs are at strides 4, 8, 8, 8;
`DepthwiseSeparableASPPHead` consumes `in_index=3` (2048 channels) plus the
low-level `c1` features from stage 0 (256 channels); `FCNHead` auxiliary on
`in_index=2` (1024 channels); losses CE 1.0 and CE 0.4. U-MV: backbone
`MambaVisionBackbone`, head `GenericUNetHead` on all four levels, losses CE 1.0
+ Dice 3.0, no auxiliary head.

**3.** (a) `--cfg-options optim_wrapper.optimizer.lr=5e-5`; (b) crop size,
`model.data_preprocessor.size`, `RandomResize.scale`, `RandomCrop.crop_size`
and `test_cfg` must change together — in the template only `crop_size` in
`_base_/dataset.py` needs editing because the others derive from it;
(c) `--cfg-options model.auxiliary_head=None`.

**4.** With 2 × 2 blocks one quarter of the raster is a contiguous test region
(one landscape type, pessimistic and noisy); with 8 × 8 blocks the test tiles
are interleaved with training tiles (optimistic). The 8 × 8 estimate is
closer to interpolation within the surveyed area; performance on a new region
is estimated only by holding out an entire orthomosaic.

**5.** `LoadRasterioImage(bands=[3, 2, 1, 4])` returns R, G, B, NIR;
`bgr_to_rgb=False`; `mean=[123.675, 116.28, 103.53, m_nir]`,
`std=[58.395, 57.12, 57.375, s_nir]` with the NIR statistics from
`band_statistics.py --bands 4`; `backbone.in_channels=4` with an adapted first
convolution (`adapt_first_conv.py --bands 4 --mode map --rgb-bands 0 1 2`).

**6.** MiT-B1: `embed_dims=64`, `num_layers=[2, 2, 2, 2]`, head
`in_channels=[64, 128, 320, 512]`, checkpoint `mit_b1_20220624-02e5a6a1.pth`.
At 512²: B1 13.7 M parameters / 15 GFLOPs, B2 24.7 M / 25 GFLOPs,
DeepLabV3+-R50-d8 43.6 M / 176 GFLOPs (2 classes; `get_flops.py --shape 512 512`).

**7.** Time per iteration from the log (`time`) × 40 000; epochs = 40 000 ×
batch / tiles.

**8.** Report the target-class IoU (73 %) together with mIoU; `aAcc` is
dominated by the background (97 % of pixels) and would be 98 % even for a
model that misses a quarter of the targets.

**9.** With `--overlap 0` the centre-crop writer degenerates to independent
tiles and outlines break at tile edges; with `--overlap 256` both blends give
continuous outlines; Hann blending is marginally smoother on ambiguous crowns.
