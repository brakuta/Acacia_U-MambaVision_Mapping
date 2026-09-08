# 3. Training

## 3.1 Model and optimisation settings

All settings are declared in `configs/` and inherited by the three variant
files; nothing is hard-coded in Python.

| Setting | Value | Config location |
|---|---|---|
| Encoder | MambaVision-T/S/B, ImageNet-1K initialisation (Hugging Face Hub) | `_base_/models/umv_unet.py` → `model.backbone` |
| Encoder widths | T (80, 160, 320, 640) · S (96, 192, 384, 768) · B (128, 256, 512, 1024) | `mambavision/U-MV-*.py` |
| Decoder | U-Net style, widths (256, 128, 64, 32), BN + ReLU, bilinear ×2 upsampling | `model.decode_head` |
| Loss | CE + 3 × Dice (Eq. 1 of the paper) | `decode_head.loss_decode` |
| Optimiser | AdamW, lr 1e-4, β = (0.9, 0.999), weight decay 0.05; backbone lr multiplier 0.1 (effective 1e-5); no decay on normalisation layers | `_base_/schedules/schedule_100k_adamw.py` |
| Schedule | 100 000 iterations, polynomial decay (power 0.9), validation every 5 000 iterations | same |
| Gradient clipping | L2 norm 0.01 | `optim_wrapper.clip_grad` |
| Batch | 2 × 1024 × 1024 tiles | `_base_/datasets/uav_acacia_dataset.py` |
| Augmentation | RandomResize (0.5–2.0), RandomCrop 1024 (`cat_max_ratio=0.75`), horizontal flip, photometric distortion | same |
| Checkpointing | every 5 000 iterations, plus `best_mIoU_iter_*.pth` on validation mIoU | `default_hooks.checkpoint` |
| Test-time inference | sliding window 1024 × 1024, stride 512 | `model.test_cfg` |

The tiny configuration keeps the hyper-parameters with which the released
U-MV-tiny checkpoint was trained (lr 1e-5, 150 000 iterations); see
`docs/07_reproducibility.md` for the rationale.

## 3.2 Commands

```bash
# inside the container, dataset mounted at /data
python tools/train.py configs/mambavision/U-MV-tiny.py
python tools/train.py configs/mambavision/U-MV-small.py
python tools/train.py configs/mambavision/U-MV-base.py
```

Useful options (inherited from MMSegmentation):

| Option | Effect |
|---|---|
| `--work-dir work_dirs/umv_small_run2` | Output directory (default `work_dirs/<config-name>`). |
| `--resume` | Continue from the latest checkpoint in the work directory. |
| `--amp` | Automatic mixed precision (halves activation memory; not used for the published runs). |
| `--cfg-options train_dataloader.num_workers=16` | Override any config key. |
| `--cfg-options randomness.seed=0 randomness.deterministic=True` | Fixed seed and deterministic cuDNN. |

Outputs in the work directory: `<timestamp>/<timestamp>.log`,
`vis_data/scalars.json` (loss and metric curves), `iter_*.pth`,
`best_mIoU_iter_*.pth`, and a copy of the resolved configuration.

## 3.3 Expected resources (paper, TITAN RTX 24 GB, batch 2)

| Variant | Parameters | FLOPs (1024²) | Training time (100k it.) | Validation mIoU |
|---|---|---|---|---|
| U-MV-t | 35.41 M | 0.144 T | 9.33 h | 87.91 % |
| U-MV-s | 54.24 M | 0.215 T | 11.8 h | 88.02 % |
| U-MV-b | not reported | 0.396 T | 18.12 h | 88.11 % |

## 3.4 Practical notes

* **First run downloads the backbone.** The MambaVision code and weights are
  fetched from the Hugging Face Hub (`nvidia/MambaVision-*-1K`) when the model
  is built. Pre-download with `tools/download_backbones.py` for offline hosts.
* **DataLoader workers.** The base config uses 8 workers. The original runs
  used 42 on a 64 GB workstation; increase `train_dataloader.num_workers` if
  the GPU is starved (check the `data_time` field in the log).
* **WSL2 memory.** Large Mamba backbones can trigger CUDA system-memory
  fallback and freeze WSL2 when host RAM is exhausted. Mitigations: disable the
  sysmem fallback in the NVIDIA Control Panel (*CUDA – Sysmem Fallback Policy →
  Prefer No Sysmem Fallback*), keep `.wslconfig` memory below physical RAM, and
  drop the page cache between runs (`sync; echo 3 > /proc/sys/vm/drop_caches`
  from the WSL2 shell as root).
* **Monitoring.** `tail -f work_dirs/<name>/<timestamp>/<timestamp>.log`, or
  add `vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]`
  to `configs/_base_/default_runtime.py` (requires `pip install tensorboard`).
