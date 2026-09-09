# Appendix C. Glossary

| Term | Definition |
|---|---|
| AMP | automatic mixed precision: 16-bit arithmetic where safe, 32-bit elsewhere; halves memory |
| auxiliary head | secondary classifier on an intermediate feature level used during training only |
| backbone / encoder | the network part that turns pixels into multi-scale features |
| BN / SyncBN / GN / LN | batch, synchronised batch, group and layer normalisation |
| checkpoint | file with model weights (and optimiser state) at one iteration |
| COG | cloud-optimised GeoTIFF: tiled, with overviews; fast windowed reads |
| config | Python file defining data, model, schedule and runtime of an experiment |
| data preprocessor | module that normalises, pads and stacks a batch on the GPU |
| decode head / decoder | the network part that turns features into per-pixel class scores |
| Dice loss | 1 − 2·overlap/(sum of areas); region-based, imbalance-robust |
| FLOPs | floating-point operations of one forward pass; cost measure |
| GSD | ground sampling distance: ground size of one pixel |
| hook | callback executed at fixed points of the training loop (logging, checkpoints, ...) |
| ignore index | label (255) excluded from loss and metrics |
| IoU | intersection over union of predicted and reference pixels of a class |
| iteration | one optimiser step; MMSegmentation schedules count iterations |
| Lovasz loss | convex surrogate of the IoU, optimised directly |
| metainfo | class names and palette attached to a dataset |
| mIoU / mFscore | class-averaged IoU / F-score |
| pipeline | ordered list of transforms applied to each sample |
| registry | table mapping `type` strings to classes |
| sliding-window inference | predicting a large image in overlapping windows and merging |
| tile | fixed-size crop of a raster used as one sample |
| TTA | test-time augmentation: averaging predictions over flips and scales |
| work directory | output folder of a run |
