# 6. Troubleshooting

| Symptom | Cause | Remedy |
|---|---|---|
| `ModuleNotFoundError: No module named 'mmcv._ext'` | `mmcv-lite` or a wheel built for another PyTorch/CUDA is installed. | Reinstall MMCV from source against the active PyTorch (`docs/01_installation.md`, step 2) or use the Docker image. |
| `No module named 'mamba_ssm'` / `selective_scan_cuda` import error | Kernels not built or built for a different PyTorch. | `pip install --no-build-isolation mamba-ssm==2.2.4` with `nvcc` available; confirm with `tools/verify_install.py`. |
| `KeyError: 'MambaVisionBackbone is not in the mmseg::model registry'` | `umv` not imported. | Configs must contain `custom_imports = dict(imports=['umv'])` (all shipped configs do); run scripts from the repository root or `pip install -e .`. |
| `OSError: We couldn't connect to 'https://huggingface.co'` | Backbone download blocked (proxy, offline). | Pre-download on a connected machine (`tools/download_backbones.py`), copy `hf_cache/` and set `HF_HUB_OFFLINE=1`. |
| `UnpicklingError: invalid load key, 'v'` when loading a `.pth` | The file is a Git LFS pointer. | `git lfs install && git lfs pull` (see `Pretrained_Weights/README.md`). |
| `_pickle.UnpicklingError: Weights only load failed` | PyTorch ≥ 2.6 with an old MMEngine. | Use MMEngine 0.10.7 (pinned). |
| `size mismatch for decode_head.decoder_stages.0.0.weight` | Checkpoint and config variant differ. | `python tools/inspect_checkpoint.py <ckpt>` prints the matching config. |
| Validation mIoU ≈ 50 % or all-background predictions | Masks are 0/255 or RGB, or wrong suffix. | Re-encode masks as 0/1 uint8 single band (`docs/02_data_preparation.md`, §2.3). |
| Test mIoU several points below the paper | Band order or resolution mismatch. | Ensure RGB orthomosaics, 2.5–3 cm GSD, no `--band-order bgr`. |
| CUDA out of memory during training | 1024² tiles with batch 2 need ~20 GB on U-MV-b. | Use `--amp`, or `train_dataloader.batch_size=1` with doubled iterations. |
| WSL2 freezes; GPU memory shows 24 GB + system RAM use | CUDA sysmem fallback exhausting host RAM. | Set *Prefer No Sysmem Fallback* in the NVIDIA Control Panel; reduce batch size; drop page cache between runs. |
| `RuntimeError: DataLoader worker ... Bus error` / shared-memory errors in Docker | `/dev/shm` too small. | Run with `--ipc=host` (compose file does) or `--shm-size=16g`. |
| `osgeo` missing in inference | GDAL Python bindings absent. | Install via conda-forge; the rasterio fallback only handles rasters ≤ 1.5 gigapixels. |
| Straight cut lines in the crown map | `--overlap` below ~128 or `--blend` misconfigured. | Keep the defaults (overlap 256, centre-crop). |
| Very slow inference from `/mnt/<drive>` | Windows filesystem access from WSL2. | Use `--scratch-dir /tmp/geospatial_work` and COG inputs. |
| `img_ratios` `NameError` when loading a config | Legacy dataset config. | Fixed in this revision; `img_ratios` is defined in `configs/_base_/datasets/uav_acacia_dataset.py`. |
