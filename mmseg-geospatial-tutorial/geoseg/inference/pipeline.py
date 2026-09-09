"""End-to-end sliding-window inference over georeferenced imagery.

Pipeline (per image)
--------------------
1. (optional) copy the input GeoTIFF to a fast local scratch directory
   (recommended on WSL2/Docker when the data live on a Windows drive);
2. tile the raster with overlap, run the segmentor on batches of tiles and
   accumulate the target-class probability into memory-mapped arrays with
   either centre-crop (seam-free) or Hann-weighted blending;
3. write the probability raster (GeoTIFF, same CRS/transform as the input);
4. threshold and polygonise it, filter by minimum area, attach the mean
   probability of every polygon and write a GeoPackage/Shapefile.

Preprocessing reproduces ``SegDataPreProcessor``: the network was trained on
RGB tensors (``bgr_to_rgb=True`` converts OpenCV's BGR loading order to RGB)
normalised with the ImageNet mean/std.  Rasterio returns bands in file
order, which is RGB for standard orthomosaics, so no channel swap is applied
unless ``band_order='bgr'`` is requested explicitly.
"""
from __future__ import annotations

import gc
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from .tiling import TileGrid, TiledRasterDataset, hann2d
from .vectorize import polygonize_probability


@dataclass
class InferenceSettings:
    tile_size: int = 1024
    overlap: int = 256
    batch_size: int = 8
    num_workers: int = 4
    prefetch_factor: int = 2
    blend: str = 'center'            # 'center' | 'hann'
    class_id: int = 1                # index of the acacia class
    thresh: float = 0.35             # polygonisation threshold
    min_area: float = 0.0            # CRS units (m^2 for projected CRS)
    connectivity: int = 4
    precision: str = 'fp16'          # 'fp16' | 'fp32' (CUDA autocast)
    bands: Sequence[int] = (1, 2, 3)
    band_order: str = 'rgb'
    prob_dtype: str = 'uint8'        # 'uint8' | 'float32'
    save_prob: bool = False
    output_format: str = 'gpkg'      # 'gpkg' | 'shp'
    accumulator_dtype: str = 'float16'
    finalize_rows: int = 4096
    mp_context: Optional[str] = None
    compute_mean_prob: bool = True
    extra: Dict = field(default_factory=dict)


def load_segmentor(config: str, checkpoint: str, device: str = 'cuda:0',
                   channels_last: bool = True):
    """Build the segmentor from a config and load a checkpoint (file or work directory)."""
    from mmengine.config import Config
    from mmseg.apis import init_model

    from ..checkpoint import resolve_checkpoint

    checkpoint = resolve_checkpoint(checkpoint)  # a work directory resolves to best_mIoU_iter_*.pth
    print(f'-> checkpoint: {checkpoint}')
    cfg = Config.fromfile(config)                # executes custom_imports (e.g. geoseg)
    # the checkpoint supplies all weights: do not download encoder pretraining at inference time
    if isinstance(cfg.model.get('backbone'), dict):
        cfg.model.backbone.pop('init_cfg', None)
    cfg.model.pretrained = None
    model = init_model(cfg, checkpoint, device=device)
    model.eval()
    if channels_last and device.startswith('cuda'):
        model.to(memory_format=torch.channels_last)
    return model, cfg


def _normalizer(cfg, device):
    dp = cfg.model.get('data_preprocessor', cfg.get('data_preprocessor'))
    if dp is None:
        raise RuntimeError('data_preprocessor missing from config')
    mean = torch.tensor(dp['mean'], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(dp['std'], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return mean, std


def _copy_with_progress(src: Path, dst: Path, chunk_mb: int = 64) -> None:
    from tqdm import tqdm
    total = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'rb') as fs, open(dst, 'wb') as fd, tqdm(
            total=total, unit='B', unit_scale=True, unit_divisor=1024,
            desc=f'Copying {src.name}', leave=False) as bar:
        while True:
            buf = fs.read(chunk_mb << 20)
            if not buf:
                break
            fd.write(buf)
            bar.update(len(buf))


def _predict_batch(model, imgs: torch.Tensor, mean, std, tile: int, class_id: int,
                   precision: str, device: str) -> np.ndarray:
    """Return target-class probabilities ``(B, tile, tile)`` as float32."""
    imgs = imgs.to(device, non_blocking=True).float()
    imgs = (imgs - mean) / std
    if device.startswith('cuda'):
        imgs = imgs.contiguous(memory_format=torch.channels_last)
    metas = [dict(img_shape=(tile, tile), ori_shape=(tile, tile), pad_shape=(tile, tile))] * imgs.shape[0]
    use_amp = precision == 'fp16' and device.startswith('cuda')
    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
        logits = model.encode_decode(imgs, metas)  # (B, C, tile, tile) at tile resolution
    probs = torch.softmax(logits.float(), dim=1)[:, class_id]
    return probs.cpu().numpy()


def segment_raster(model, cfg, input_path: str, output_path: str,
                   settings: InferenceSettings, device: str = 'cuda:0',
                   scratch_dir: Optional[str] = None, keep_scratch: bool = False,
                   prob_output_path: Optional[str] = None) -> Dict:
    """Segment one GeoTIFF and write crown polygons.

    Returns a dictionary with basic statistics (tiles, polygons, timings).
    """
    import rasterio
    from rasterio.windows import Window
    from tqdm import tqdm

    s = settings
    t_start = time.time()
    input_path, output_path = Path(input_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work = Path(scratch_dir) if scratch_dir else output_path.parent / f'.{output_path.stem}_work'
    work.mkdir(parents=True, exist_ok=True)

    # 1) optional staging copy
    if scratch_dir:
        local_in = work / input_path.name
        if not local_in.exists() or local_in.stat().st_size != input_path.stat().st_size:
            _copy_with_progress(input_path, local_in)
    else:
        local_in = input_path

    # 2) grid, accumulators
    with rasterio.open(local_in) as src:
        width, height = src.width, src.height
        if src.crs is None:
            print(f'   (warning) {input_path.name} has no CRS; outputs will be in pixel space')
        prob_meta = src.meta.copy()
    grid = TileGrid(width, height, s.tile_size, s.overlap)
    acc_dtype = np.dtype(s.accumulator_dtype)
    sum_path, wgt_path = work / 'sum.mmap', work / 'wgt.mmap'
    sum_mm = np.memmap(sum_path, dtype=acc_dtype, mode='w+', shape=(height, width))
    wgt_mm = np.memmap(wgt_path, dtype=acc_dtype, mode='w+', shape=(height, width))

    dataset = TiledRasterDataset(str(local_in), grid, bands=s.bands, band_order=s.band_order)
    loader_kwargs = dict(batch_size=s.batch_size, shuffle=False, num_workers=s.num_workers,
                         pin_memory=device.startswith('cuda'), drop_last=False)
    if s.num_workers > 0:
        loader_kwargs.update(persistent_workers=False, prefetch_factor=s.prefetch_factor)
        if s.mp_context:
            loader_kwargs['multiprocessing_context'] = s.mp_context
    loader = DataLoader(dataset, **loader_kwargs)

    mean, std = _normalizer(cfg, device)
    window = hann2d(s.tile_size) if s.blend == 'hann' else None

    # 3) inference
    print(f'-> {input_path.name}: {width} x {height} px, {len(grid)} tiles '
          f'(tile={s.tile_size}, overlap={s.overlap}, blend={s.blend}, {s.precision})')
    t0 = time.time()
    for tiles, idxs in tqdm(loader, desc='   inferring', unit='batch', leave=False):
        probs = _predict_batch(model, tiles, mean, std, s.tile_size, s.class_id, s.precision, device)
        for i in range(probs.shape[0]):
            idx = int(idxs[i])
            t, p = grid.tiles[idx], probs[i]
            if s.blend == 'hann':
                wg = window[:t.h, :t.w]
                sum_mm[t.y:t.y + t.h, t.x:t.x + t.w] += (p[:t.h, :t.w] * wg).astype(acc_dtype)
                wgt_mm[t.y:t.y + t.h, t.x:t.x + t.w] += wg.astype(acc_dtype)
            else:
                b = grid.write_bounds(idx)
                if b is None:
                    continue
                y0, y1, x0, x1 = b
                sum_mm[y0:y1, x0:x1] += p[y0 - t.y:y1 - t.y, x0 - t.x:x1 - t.x].astype(acc_dtype)
                wgt_mm[y0:y1, x0:x1] += acc_dtype.type(1.0)
    sum_mm.flush(); wgt_mm.flush()
    t_infer = time.time() - t0

    # 4) probability raster
    prob_path = Path(prob_output_path) if prob_output_path else (
        output_path.with_name(output_path.stem + '_prob.tif') if s.save_prob else work / 'prob.tif')
    prob_meta.update(count=1, dtype=s.prob_dtype, tiled=True, blockxsize=512, blockysize=512,
                     compress='deflate', predictor=2 if s.prob_dtype == 'uint8' else 3,
                     BIGTIFF='IF_SAFER', nodata=None)
    scale = 255.0 if s.prob_dtype == 'uint8' else 1.0
    with rasterio.open(prob_path, 'w', **prob_meta) as dst:
        for y0 in tqdm(range(0, height, s.finalize_rows), desc='   writing prob', leave=False):
            h = min(s.finalize_rows, height - y0)
            sb = np.asarray(sum_mm[y0:y0 + h], dtype=np.float32)
            wb = np.asarray(wgt_mm[y0:y0 + h], dtype=np.float32)
            out = np.zeros_like(sb)
            pos = wb > 1e-6
            out[pos] = sb[pos] / wb[pos]
            if s.prob_dtype == 'uint8':
                out = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
            dst.write(out, 1, window=Window(0, y0, width, h))
    del sum_mm, wgt_mm
    gc.collect()
    sum_path.unlink(missing_ok=True); wgt_path.unlink(missing_ok=True)
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

    # 5) vectorise
    ext = '.shp' if s.output_format == 'shp' else '.gpkg'
    vec_path = output_path.with_suffix(ext)
    n_poly = polygonize_probability(str(prob_path), str(vec_path), thresh=s.thresh,
                                    min_area=s.min_area, connectivity=s.connectivity,
                                    compute_mean_prob=s.compute_mean_prob, prob_scale=scale)

    if not (s.save_prob or prob_output_path):
        Path(prob_path).unlink(missing_ok=True)
    if not keep_scratch:
        shutil.rmtree(work, ignore_errors=True)
    stats = dict(image=str(input_path), width=width, height=height, tiles=len(grid),
                 polygons=n_poly, vector=str(vec_path) if n_poly else None,
                 prob_raster=str(prob_path) if (s.save_prob or prob_output_path) else None,
                 seconds_inference=round(t_infer, 1), seconds_total=round(time.time() - t_start, 1))
    print(f'   done in {stats["seconds_total"]} s (inference {stats["seconds_inference"]} s)')
    return stats
