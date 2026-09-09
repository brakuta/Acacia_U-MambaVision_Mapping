# Pretrained U-MV weights

The three released checkpoints are stored with **Git LFS**.  A plain `git clone`
without LFS yields 134-byte pointer files, not weights.  Fetch the binaries with:

```bash
git lfs install
git lfs pull                       # all three (~815 MB)
git lfs pull --include="Pretrained_Weights/U-MV-small_latest.pth"   # one variant
```

| Variant     | File                     | Size   | SHA-256 (LFS oid)                                                   |
|-------------|--------------------------|--------|---------------------------------------------------------------------|
| U-MV-tiny   | `U-MV-tiny_latest.pth`   | 159 MB | `85796b37582647a6d83973e240ebf30e0a562676bd44ae84eb81c953dc8900ac` |
| U-MV-small  | `U-MV-small_latest.pth`  | 234 MB | `0c8d2b3dcdbf6272ea12d9fcdac2cba79ca185277cb2f5d1eff55e5d065ed9d8` |
| U-MV-base   | `U-MV-base_latest.pth`   | 422 MB | `fcd86b17fb1c49e4f9555538526cc0edff805d0303463bbcc9336b7e537b66d3` |

Verify a download with `sha256sum <file>` and identify the variant of any
local checkpoint with:

```bash
python tools/inspect_checkpoint.py /path/to/checkpoint.pth
```

A checkpoint obtained elsewhere (e.g. a `best_mIoU_iter_*.pth` from a
`work_dirs/` folder) can be used directly by passing its path to
`tools/test.py` or the inference scripts; no renaming is required.

## Provenance (verified September 2026)

The released files are byte-identical to the best-validation checkpoints of
the original MMSegmentation work directories (SHA-256 verified):

| Released file | Work directory | Checkpoint | Iteration |
|---|---|---|---|
| `U-MV-tiny_latest.pth` | `mambavision-t_generic-unet_acacia` | `best_mIoU_iter_100000.pth` | 100 000 |
| `U-MV-small_latest.pth` | `mambavision-s_generic-unet_acacia-88` | `best_mIoU_iter_95000.pth` | 95 000 |
| `U-MV-base_latest.pth` | `mambavision-b_generic-unet_acacia` | `best_mIoU_iter_60000.pth` | 60 000 |

The `_latest` suffix therefore denotes the latest release, not the last
training iteration. `iter_100000.pth` files in the work directories are the
final-iteration checkpoints and were not released.

## Original work directories

Passing a work-directory folder to any tool selects its `best_mIoU_iter_*.pth`:

```bash
python tools/inspect_checkpoint.py "/weights/mambavision-s_generic-unet_acacia-88"
python tools/test.py configs/mambavision/U-MV-small.py "/weights/mambavision-s_generic-unet_acacia-88" --test-split test2
```
