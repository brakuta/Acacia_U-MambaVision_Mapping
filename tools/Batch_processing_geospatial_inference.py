#!/usr/bin/env python3
"""
High-Performance, One-to-One Batch Crown Segmentation for UAV Orthos

- Recursively processes GeoTIFFs under INPUT_ROOT_ON_HOST
- For each image, creates a single output (GPKG by default) containing polygons
  with a 'mean_prob' attribute
- Uses center-crop blending (seam-safe). Optionally switch to Hann weighting
  by setting BLEND_MODE='hann'.
"""

# ========================= USER SETTINGS =========================
CONFIG_FILE             = 'configs/custom/U-MV-small.py'
CHECKPOINT_FILE         = 'work_dirs/best_mIoU_iter_95000.pth'

INPUT_ROOT_ON_HOST      = "/data/drone_imgs/COGeotiff/"
OUTPUT_ROOT_ON_HOST     = "/data/drone_imgs/Predictions_of_U_MV-small/"

CONTAINER_WORK_DIR      = "/tmp/geospatial_work"   # per-image scratch
DELETE_WORK_DIR         = True                     # delete /tmp folder after each image

DEVICE                  = 'cuda:0'
TILE_SIZE               = 1024
OVERLAP                 = 256
BATCH_SIZE              = 16
NUM_WORKERS             = 4
PIN_MEMORY              = True
PREFETCH_FACTOR         = 2
TARGET_PINNED_BUDGET_GB = 2.0

TARGET_CLASS_ID         = 1
THRESH                  = 0.3
FINALIZE_BLOCK          = 4096

GDAL_CACHE_MB           = 2048
GDAL_THREADS            = "ALL_CPUS"
EXPORT_GPKG             = True                     # if False -> ESRI Shapefile

BLEND_MODE              = 'hann'                # 'center' | 'hann'
# ================================================================

import os, gc, shutil, ctypes
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as win_transform
from rasterio.features import geometry_mask, geometry_window
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import geopandas as gpd
from rasterstats import zonal_stats

from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=False)

# -------------------- Utilities --------------------
def copy_with_progress(src, dst, chunk_mb: int = 64):
    src, dst = Path(src), Path(dst)
    total = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst, tqdm(
        total=total, unit='B', unit_scale=True, desc=f"Copying {src.name}", leave=False
    ) as pbar:
        while True:
            buf = fsrc.read(chunk_mb * 1024 * 1024)
            if not buf: break
            fdst.write(buf); pbar.update(len(buf))

def drop_file_cache(path: str):
    if os.name != 'posix': return
    try:
        POSIX_FADV_DONTNEED = 4
        _libc = ctypes.CDLL("libc.so.6")
        fd = os.open(path, os.O_RDONLY)
        _libc.posix_fadvise(ctypes.c_int(fd), ctypes.c_longlong(0), ctypes.c_longlong(0),
                            ctypes.c_int(POSIX_FADV_DONTNEED))
        os.close(fd)
    except Exception:
        pass

def dtype_bytes(dtype): return np.dtype(dtype).itemsize

def hann2d(size):
    if size <= 1: return np.ones((size, size), dtype=np.float32)
    w = np.hanning(size).astype(np.float32)
    return np.clip(np.outer(w, w), 1e-4, None)

# -------------------- Dataset (returns CHW tensor) --------------------
class TiledGeoTiffDataset(Dataset):
    def __init__(self, cog_path, tile_size, overlap):
        self.cog_path  = str(cog_path)
        self.tile_size = tile_size
        self._ds       = None
        with rasterio.open(self.cog_path) as src:
            W, H = src.width, src.height
            self.dtype = src.dtypes[0]
        stride = tile_size - overlap
        self.coords = [(x, y, min(tile_size, W-x), min(tile_size, H-y))
                       for y in range(0, H, stride)
                       for x in range(0, W, stride)]
    def __len__(self): return len(self.coords)
    def _ensure_open(self):
        if self._ds is None:
            self._ds = rasterio.open(self.cog_path)
    def __getitem__(self, idx):
        self._ensure_open()
        x, y, w, h = self.coords[idx]
        tile = self._ds.read([1,2,3], window=Window(x, y, w, h)).transpose(1,2,0)  # HWC
        ph, pw = self.tile_size - h, self.tile_size - w
        if ph or pw:
            tile = np.pad(tile, ((0,ph),(0,pw),(0,0)), mode='constant')
        t = torch.from_numpy(np.ascontiguousarray(tile)).permute(2,0,1)            # CHW
        return t, x, y, w, h

# -------------------- Mean prob helper --------------------
def mean_probability_for_geometry(src_prob_ds, geom):
    try:
        win = geometry_window(src_prob_ds, [geom], pad_x=2, pad_y=2, north_up=True, pixel_precision=3)
        block = src_prob_ds.read(1, window=win, boundless=True, fill_value=0)
        mask = geometry_mask([geom], out_shape=(win.height, win.width),
                             transform=win_transform(win, src_prob_ds.transform), invert=True)
        vals = block[mask]
        return float(vals.mean()) if vals.size else 0.0
    except Exception:
        block = src_prob_ds.read(1)
        mask = geometry_mask([geom], out_shape=(src_prob_ds.height, src_prob_ds.width),
                             transform=src_prob_ds.transform, invert=True)
        vals = block[mask]
        return float(vals.mean()) if vals.size else 0.0

# -------------------- Vectorization (GDAL Polygonize + mean_prob) --------------------
def finalize_vectors(prob_mask_path, source_geotiff_path, out_path, thresh):
    from osgeo import gdal, ogr, osr

    print(f"  -> Vectorizing (gdal.Polygonize @ thresh={thresh})")
    tmp_bin = Path(out_path).with_suffix('.bin.tif.temp')
    tmp_raw = Path(out_path).with_suffix('.gpkg.raw.temp')

    # 1) Threshold -> temp binary raster
    with rasterio.open(prob_mask_path) as src:
        meta = src.meta.copy(); meta.update(dtype='uint8')
        with rasterio.open(tmp_bin, 'w', **meta) as dst:
            bwins = list(src.block_windows(1))
            for _, win in tqdm(bwins, desc="    1/3 Binarizing", leave=False):
                prob = src.read(1, window=win)
                dst.write((prob >= thresh).astype(np.uint8), 1, window=win)

    # 2) Polygonize to temp gpkg
    src_ds = gdal.Open(str(tmp_bin)); srcband = src_ds.GetRasterBand(1)
    drv = ogr.GetDriverByName('GPKG')
    if tmp_raw.exists(): drv.DeleteDataSource(str(tmp_raw))
    dst_ds = drv.CreateDataSource(str(tmp_raw))
    srs = osr.SpatialReference(); srs.ImportFromWkt(src_ds.GetProjection())
    layer = dst_ds.CreateLayer('polygons', srs=srs, geom_type=ogr.wkbMultiPolygon)
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
    print("    2/3 Polygonizing…")
    gdal.Polygonize(srcband, srcband, layer, 0, [])
    dst_ds = None; layer = None; src_ds = None

    if not tmp_raw.exists() or os.path.getsize(tmp_raw) == 0:
        print("    -> No polygons at this threshold.")
        tmp_bin.unlink(missing_ok=True)
        return

    # 3) Attach mean_prob and save final
    gdf = gpd.read_file(tmp_raw)
    if 'value' in gdf.columns:
        gdf = gdf[gdf['value'] == 1].drop(columns=['value'])

    with rasterio.open(source_geotiff_path) as sref:
        gdf.set_crs(sref.crs, inplace=True)

    print(f"    3/3 Computing mean_prob for {len(gdf)} polygons…")
    means = []
    with rasterio.open(prob_mask_path) as srcp:
        for geom in tqdm(gdf.geometry, leave=False):
            means.append(mean_probability_for_geometry(srcp, geom))
    gdf['mean_prob'] = np.asarray(means, dtype=np.float32).round(4)

    driver = 'GPKG' if EXPORT_GPKG else 'ESRI Shapefile'
    gdf.to_file(out_path, driver=driver)
    print(f"    -> Saved: {out_path}  ({len(gdf)} polygons)")

    tmp_bin.unlink(missing_ok=True); tmp_raw.unlink(missing_ok=True)

# --------------------------- Main Batch Execution ---------------------------
if __name__ == '__main__':
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.environ.setdefault("GDAL_CACHEMAX", str(GDAL_CACHE_MB))
    os.environ.setdefault("GDAL_NUM_THREADS", GDAL_THREADS)
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    input_root  = Path(INPUT_ROOT_ON_HOST)
    output_root = Path(OUTPUT_ROOT_ON_HOST); output_root.mkdir(parents=True, exist_ok=True)

    print(f"-> Searching for *.tif / *.tiff under: {input_root}")
    all_input_files = list(input_root.rglob('*.tif')) + list(input_root.rglob('*.tiff'))
    all_input_files = [p for p in all_input_files if '_mask' not in p.name and '_prob' not in p.name]
    if not all_input_files:
        raise FileNotFoundError(f"No GeoTIFF images found in {input_root}")
    print(f"-> Found {len(all_input_files)} images.")

    print("\n-> Loading model once for the batch…")
    cfg = Config.fromfile(CONFIG_FILE)
    cfg.model.test_cfg = dict(mode='whole')
    model = init_model(cfg, CHECKPOINT_FILE, device=DEVICE).eval().to(memory_format=torch.channels_last)

    # Keep mean/std on CPU; move to imgs.device per batch (fixes CUDA/CPU mismatch)
    dp = cfg.model.get('data_preprocessor', cfg.get('data_preprocessor', None))
    if dp is None:
        raise RuntimeError("No data_preprocessor in config.")
    MEAN_T = torch.tensor(dp['mean'], dtype=torch.float32).view(1, 3, 1, 1)
    STD_T  = torch.tensor(dp['std'],  dtype=torch.float32).view(1, 3, 1, 1)
    to_rgb = bool(dp.get('bgr_to_rgb', False))

    for host_in_path in tqdm(all_input_files, desc="Overall Progress"):
        relative_path    = host_in_path.relative_to(input_root)
        host_output_path = output_root / relative_path
        ext              = ".gpkg" if EXPORT_GPKG else ".shp"
        host_output_file = host_output_path.with_suffix(ext)

        if host_output_file.exists():
            print(f"\n-> Skipping {host_in_path.name} (exists).")
            continue

        print(f"\n{'='*22} Processing: {host_in_path.name} {'='*22}")
        work_dir = Path(CONTAINER_WORK_DIR)
        if work_dir.exists(): shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Paths in /tmp
        local_in            = work_dir / host_in_path.name
        temp_prob_mask_path = work_dir / (host_in_path.stem + "_prob.tif")  # temp only
        sum_path            = work_dir / 'sum.mmap'
        wgt_path            = work_dir / 'wgt.mmap'
        local_output_file   = work_dir / host_output_file.name

        print("-> [1/5] Copying to /tmp …")
        copy_with_progress(host_in_path, local_in)
        drop_file_cache(str(local_in))

        print("-> [2/5] Preparing DataLoader & accumulators…")
        dataset = TiledGeoTiffDataset(local_in, TILE_SIZE, OVERLAP)
        with rasterio.open(local_in) as src:
            width, height = src.width, src.height
            meta = src.meta.copy()
            meta.update(count=1, dtype='float32', tiled=True,
                        blockxsize=512, blockysize=512, compress='lzw',
                        BIGTIFF='IF_SAFER')

        sum_mm = np.memmap(sum_path, dtype=np.float16, mode='w+', shape=(height, width))
        wgt_mm = np.memmap(wgt_path, dtype=np.float16, mode='w+', shape=(height, width))

        # tune loader by pinned-memory budget
        bytes_per_tile = TILE_SIZE * TILE_SIZE * 3 * dtype_bytes(dataset.dtype)
        max_b = max(1, int((TARGET_PINNED_BUDGET_GB * (1024**3)) // (BATCH_SIZE * bytes_per_tile)))
        num_w = min(NUM_WORKERS, max_b)
        prefetch = max(1, min(PREFETCH_FACTOR, max_b // max(1, num_w)))

        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=num_w, pin_memory=PIN_MEMORY,
            persistent_workers=(num_w > 0), prefetch_factor=prefetch
        )

        print(f"-> [3/5] Inference w/{num_w} workers (prefetch={prefetch}, pin={PIN_MEMORY})…")
        pbar   = tqdm(loader, desc="  Inferring Tiles", leave=False)
        margin = OVERLAP // 2
        W_full = hann2d(TILE_SIZE) if BLEND_MODE == 'hann' else None

        for batch_tiles, bx, by, bw, bh in pbar:
            # batch_tiles is already a stacked Tensor (B,C,H,W)
            imgs = batch_tiles.to(DEVICE, non_blocking=True, dtype=torch.float32)

            if to_rgb:
                imgs = imgs[:, [2, 1, 0], ...]

            # Move mean/std to the same device/dtype as imgs (fixes CUDA/CPU mismatch)
            mean_ = MEAN_T.to(imgs.device, dtype=imgs.dtype)
            std_  = STD_T.to(imgs.device,  dtype=imgs.dtype)

            imgs = (imgs - mean_) / std_
            imgs = imgs.contiguous(memory_format=torch.channels_last)

            metas = [{'img_shape': (TILE_SIZE, TILE_SIZE),
                      'ori_shape': (bh[i].item(), bw[i].item())} for i in range(imgs.shape[0])]
            data_samples = [SegDataSample(metainfo=m) for m in metas]

            with torch.inference_mode():
                results = model(imgs, data_samples, mode='predict')

            probs_list = [torch.softmax(r.seg_logits.data, dim=0).cpu().numpy()
                          for r in results]

            bxn, byn, bwn, bhn = bx.numpy(), by.numpy(), bw.numpy(), bh.numpy()
            for i in range(len(probs_list)):
                x, y, w, h = int(bxn[i]), int(byn[i]), int(bwn[i]), int(bhn[i])
                p = probs_list[i][TARGET_CLASS_ID]
                if p.shape != (h, w):
                    p = torch.nn.functional.interpolate(
                            torch.from_numpy(p)[None, None, ...],
                            size=(h, w), mode='bilinear', align_corners=False
                        ).squeeze().numpy()

                if BLEND_MODE == 'hann':
                    wgt = W_full[:h, :w]
                    sum_mm[y:y+h, x:x+w] += (p * wgt).astype(np.float16)
                    wgt_mm[y:y+h, x:x+w] += wgt.astype(np.float16)
                else:
                    # center-crop write region (seam-safe)
                    cy0 = 0 if y == 0 else margin
                    cx0 = 0 if x == 0 else margin
                    cy1 = h if (y + h) >= height else (h - margin)
                    cx1 = w if (x + w) >= width  else (w - margin)
                    if cy0 >= cy1 or cx0 >= cx1:
                        continue
                    gy0, gy1 = y + cy0, y + cy1
                    gx0, gx1 = x + cx0, x + cx1
                    pc = p[cy0:cy1, cx0:cx1]
                    sum_mm[gy0:gy1, gx0:gx1] += pc.astype(np.float16)
                    wgt_mm[gy0:gy1, gx0:gx1] += np.float16(1.0)

        pbar.close()
        sum_mm.flush(); wgt_mm.flush()

        print("-> [4/5] Finalizing probability raster (temp only)…")
        with rasterio.open(temp_prob_mask_path, 'w', **meta) as dst:
            for y0 in tqdm(range(0, height, FINALIZE_BLOCK), desc="  Finalizing", leave=False):
                h_chunk = min(FINALIZE_BLOCK, height - y0)
                sblk = np.asarray(sum_mm[y0:y0+h_chunk, :], dtype=np.float32)
                wblk = np.asarray(wgt_mm[y0:y0+h_chunk, :], dtype=np.float32)
                out  = np.zeros_like(sblk, dtype=np.float32)
                pos  = (wblk > 1e-6)
                if np.any(pos):
                    out[pos] = sblk[pos] / wblk[pos]
                dst.write(out, 1, window=Window(0, y0, width, h_chunk))

        del sum_mm, wgt_mm; gc.collect(); torch.cuda.empty_cache()

        print("-> [5/5] Vectorizing (GDAL) and writing mean_prob…")
        finalize_vectors(str(temp_prob_mask_path), str(local_in), str(local_output_file), thresh=THRESH)

        # Copy just the final vector(s) back to host
        host_output_file.parent.mkdir(parents=True, exist_ok=True)
        if EXPORT_GPKG:
            shutil.copy(local_output_file, host_output_file)
        else:
            # shapefile sidecars
            for e in ('.shp', '.dbf', '.shx', '.prj', '.cpg'):
                sf = local_output_file.with_suffix(e)
                if sf.exists():
                    shutil.copy(sf, host_output_file.parent / sf.name)

        # Clean temp folder for this image
        if DELETE_WORK_DIR and work_dir.exists():
            try: shutil.rmtree(work_dir)
            except Exception: pass

    print("\n--- Batch processing complete! ---")
