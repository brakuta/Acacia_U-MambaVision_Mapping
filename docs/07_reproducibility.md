# 7. Reproducibility notes and paper ↔ code mapping

## 7.1 Mapping of Section 3 of the paper to the configuration

| Paper statement | Implementation |
|---|---|
| MambaVision T/S/B encoder, ImageNet-1K initialisation via timm/Hugging Face (§3.3, §3.5) | `MambaVisionBackbone(variant=..., pretrained=True)` → `AutoModel.from_pretrained('nvidia/MambaVision-{T,S,B}-1K', trust_remote_code=True)` |
| Encoder widths (80,160,320,640) / (96,192,384,768) / (128,256,512,1024) (§3.3) | `decode_head.encoder_channels` in `configs/mambavision/U-MV-*.py` |
| Lightweight CNN decoder with skip connections (§3.3) | `GenericUNetHead`, widths (256,128,64,32) |
| L = L_CE + 3 L_Dice (Eq. 1) | `loss_decode=[CrossEntropyLoss(1.0), DiceLoss(3.0)]` |
| AdamW, β=(0.9,0.999), weight decay 0.05 (§3.5) | `optimizer` in `_base_/schedules/schedule_100k_adamw.py` |
| Polynomial decay, power 0.9 (§3.5) | `PolyLR(power=0.9, eta_min=0)` |
| 100 000 iterations, validation every 5 000 (§3.5, §4.2) | `train_cfg`, `val_interval=5000` |
| Batch size 2, tiles 1024 × 1024 (§3.2, §3.5) | `train_dataloader.batch_size=2`, `crop_size=(1024,1024)` |
| Gradient clipping (§3.5) | `clip_grad=dict(max_norm=0.01, norm_type=2)` |
| Random crop, horizontal flip, photometric perturbation (§3.5) | `train_pipeline` |
| Best validation checkpoint retained (§3.5) | `CheckpointHook(save_best='mIoU')` |
| Metrics: precision, recall, mIoU, mF-score (§3.6) | `IoUMetric(iou_metrics=['mIoU','mFscore'])` |
| Docker on a TITAN RTX workstation, 64 GB RAM (§3.5) | `docker/Dockerfile` (PyTorch 2.6.0 + CUDA 11.8 base) |

## 7.2 Discrepancies requiring the authors' confirmation

1. **Learning rate.** The paper reports an initial learning rate of 1 × 10⁻⁵.
   The released small/base configs use 1 × 10⁻⁴ with a backbone multiplier of
   0.1, i.e. 1 × 10⁻⁵ for the encoder and 1 × 10⁻⁴ for the decoder. The
   released tiny config uses 1 × 10⁻⁵ (encoder 1 × 10⁻⁶). The configs are
   kept verbatim; the paper's statement most plausibly refers to the encoder
   rate of the small/base runs.
2. **Iterations of the tiny variant.** The tiny config specifies 150 000
   iterations, the paper and the other variants 100 000. The released
   U-MV-tiny checkpoint is named `_latest`, so the iteration at which it was
   saved is recorded in its `meta['iter']` field
   (`tools/inspect_checkpoint.py`).
3. **`data_preprocessor.size`.** (512, 512) in the tiny config versus
   (1024, 1024) elsewhere; inert for 1024 × 1024 tiles (padding only).
4. **Seeds.** No seed was fixed in the original configs; runs are therefore
   not bit-reproducible. For controlled comparisons add
   `--cfg-options randomness.seed=0 randomness.deterministic=True`.
5. **MMSegmentation revision.** The original installation used the `main`
   branch; v1.2.2 is the last tagged release and is API-identical for the
   components used here.
6. **Channel order at inference.** The original inference scripts swapped
   channels to BGR before the network; the revised pipeline feeds RGB, which
   is what the training preprocessor produced (`docs/05_inference.md`, §5.4).
   Regional maps produced with the old scripts and with the revised pipeline
   may therefore differ slightly; a re-evaluation on the test split with
   `tools/test.py` (unaffected by this change) provides the reference.

## 7.3 What is and is not fixed by the pinned environment

Fixed: library versions, CUDA kernels, preprocessing, architecture, optimiser
and schedule. Not fixed: cuDNN algorithm selection (`cudnn_benchmark=True`),
data-loading order without a seed, and the exact Hugging Face revision of the
MambaVision weights (pin a `revision=` in `MambaVisionBackbone` via
`model_name` if the Hub repository is updated upstream).

## 7.4 Archival

The code is archived on Zenodo (DOI 10.5281/zenodo.18068379). When
re-archiving this revision, include `docs/reference/environment.frozen.yml`,
the Docker image digest (`docker images --digests umv`), and the SHA-256 of
the checkpoints listed in `Pretrained_Weights/README.md`.
