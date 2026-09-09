# Example 1 — Acacia tortilis crowns from UAV RGB (binary)

Example dataset: 1024 x 1024 RGB tiles at 2.5 cm GSD with binary crown masks
(0 = background, 1 = acacia), split into `train`, `val`, `test2` and a
spatially separate `Generalizability` region (Gibril et al., 2026). The
dataset is project-confidential; ask the data owner for access and mount it at
`/data/acacia`.

`configs/_base_/dataset.py` is the generic template pointed at these tiles;
four zoo models share it so that architectures can be compared under identical
data, augmentation and schedule.
