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
| Docker on a TITAN RTX workstation, 64 GB RAM (§3.5) | `docker/Dockerfile` (PyTorch 2.6.0 + CUDA 11.8 base); the original container definition is archived as `docs/reference/Dockerfile.original` |

## 7.2 Reconciliation with the original training logs

The MMSegmentation work directories of the three published runs were
recovered in September 2026 and their logs compared with the paper and the
released configurations.

1. **Learning rate.** All three runs used a base learning rate of 1 × 10⁻⁴
   with a backbone multiplier of 0.1; the logs list every encoder parameter
   at 1 × 10⁻⁵. The paper's "initial learning rate of 1 × 10⁻⁵" therefore
   refers to the encoder rate. The configs in this repository reproduce the
   logged setting.
2. **Schedule of the tiny variant.** The tiny log shows 100 000 iterations
   with validation every 5 000, identical to small and base, and reports the
   best validation mIoU (87.91 %, as in Table 1 of the paper) at iteration
   100 000. The tiny config released earlier (lr 1 × 10⁻⁵, 150 000
   iterations) did not correspond to the run and has been aligned with the
   log.
3. **Checkpoints.** The released files are byte-identical (SHA-256) to
   `best_mIoU_iter_100000.pth` (tiny), `best_mIoU_iter_95000.pth` (small)
   and `best_mIoU_iter_60000.pth` (base) of the work directories, i.e. the
   best-validation checkpoints used for the paper's test results.
4. **Seeds.** No seed was fixed; runs are not bit-reproducible. For
   controlled comparisons add
   `--cfg-options randomness.seed=0 randomness.deterministic=True`.
5. **MMSegmentation revision.** The original container installed the `main`
   branch (`docs/reference/Dockerfile.original`); v1.2.2 is the last tagged
   release and is API-identical for the components used here.
6. **Channel order at inference.** The original inference scripts swapped
   channels to BGR before the network; the revised pipeline feeds RGB, which
   is what the training preprocessor produced (`docs/05_inference.md`, §5.4).
   Regional maps produced with the old scripts may therefore differ
   slightly; `tools/test.py` is unaffected and provides the reference.
7. **Training split of the archived dataset.** The archived `train` folder
   holds 4 893 tiles (1024 × 1024, identical in every copy found), whereas
   the paper reports 26 615. The tile indices run from 0 to 19 765 with
   gaps, and some files carry modification dates after the training runs
   (August 2025), which indicates that the training folder was pruned after
   the models were trained. Validation (2 407; the logs evaluate 1 204
   batches of 2), test (3 123) and generalisability (2 162) tiles are
   complete. Consequently, evaluation and inference with the released
   checkpoints are fully reproducible from the archive, whereas retraining
   from the archive reproduces the recipe but not the published model.

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
