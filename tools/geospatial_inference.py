#!/usr/bin/env python3
"""
WSL2/Docker Crown Segmentation — Seam-free + MeanProb (Final)

Key points
- Center-crop writer (no Hann needed): overlap is used only for context; only the
  inner "safe" region of each tile is written → eliminates straight cut lines.
- Robust, streaming: memmap accumulators (+ blockwise finalization).
- GDAL Polygonize for clean vectors.
- Per-polygon mean probability ("mean_prob") for threshold tuning in GIS.
- Optional deletion of /tmp/geospatial_work on success.

Tested with mmseg 1.x style configs using `encode_decode`.
"""

# ========================= USER SETTINGS =========================
CONFIG_FILE             = 'configs/custom/U-MV-small.py'
CHECKPOINT_FILE         = 'work_dirs/best_mIoU_iter_95000.pth'

INPUT_PATH_ON_HOST      = "/data/drone_imgs/Image.tif"
OUTPUT_BASENAME_ON_HOST = "/data/drone_imgs/output/Result"

CONTAINER_WORK_DIR      = "/tmp/geospatial_work"
DELETE_WORK_DIR         = True        # set True to auto-delete the work dir at the end

DEVICE                  = 'cuda:0'
TILE_SIZE               = 1024
OVERLAP                 = 256          # >= 128 recommended
BATCH_SIZE              = 16
NUM_WORKERS             = 4
PIN_MEMORY              = True
PREFETCH_FACTOR         = 2
TARGET_PINNED_BUDGET_GB = 2.0          # auto-limits #in-flight batches

TARGET_CLASS_ID         = 1            # index of your "tree" class
THRESH                  = 0.35         # polygonization threshold on probability raster
FINALIZE_BLOCK          = 4096         # rows per write when finalizing prob raster

GDAL_CACHE_MB           = 2048
GDAL_THREADS            = "ALL_CPUS"

EXPORT_GPKG             = True         # False → ESRI Shapefile
# ================================================================

import os, gc, shutil, ctypes, math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes as rio_shapes
from rasterio.features import geometry_mask
from rasterio.windows import transform as win_transform
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.features import geometry_window

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import shape

from mmengine.config import Config
from mmseg.apis import init_model
from mmseg.utils import register_all_modules
register_all_modules(init_default_scope=False)

# -------------------- Worker Init Function --------------------
def worker_init_fn(worker_id):
    os.environ["GDAL_NUM_THREADS"] = "1"

# -------------------- Utilities --------------------
def copy_with_progress(src, dst, chunk_mb: int = 64):
    src, dst = Path(src), Path(dst)
    total = src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst, tqdm(
        total=total, unit='B', unit_scale=True, unit_divisor=1024,
        desc=f"Copying {src.name} → {dst}"
    ) as pbar:
        while True:
            buf = fsrc.read(chunk_mb * 1024 * 1024)
            if not buf:
                break
            fdst.write(buf); pbar.update(len(buf))

def drop_file_cache(path: str):
    if os.name != 'posix':
        return
    try:
        POSIX_FADV_DONTNEED = 4
        _libc = ctypes.CDLL("libc.so.6")
        fd = os.open(path, os.O_RDONLY)
        _libc.posix_fadvise(ctypes.c_int(fd), ctypes.c_longlong(0),
                            ctypes.c_longlong(0), ctypes.c_int(POSIX_FADV_DONTNEED))
        os.close(fd)
    except Exception:
        pass

def dtype_bytes(dtype) -> int:
    return np.dtype(dtype).itemsize

# -------------------- Dataset (returns CHW torch tensors) --------------------
class TiledGeoTiffDataset(Dataset):
    """Each worker keeps its own open COG handle."""
    def __init__(self, cog_path, tile_size, overlap):
        self.cog_path  = str(cog_path)
        self.tile_size = tile_size
        self._ds = None
        with rasterio.open(self.cog_path) as src:
            self.W, self.H = src.width, src.height
            self.dtype = src.dtypes[0]
        stride = tile_size - overlap
        self.coords = [(x, y, min(tile_size, self.W-x), min(tile_size, self.H-y))
                       for y in range(0, self.H, stride)
                       for x in range(0, self.W, stride)]

    def __len__(self): return len(self.coords)

    def _ensure_open(self):
        if self._ds is None:
            self._ds = rasterio.open(self.cog_path)

    def __getitem__(self, idx):
        self._ensure_open()
        x, y, w, h = self.coords[idx]
        # HWC read, then pad to TILE_SIZE, then CHW tensor
        tile = self._ds.read([1,2,3], window=Window(x, y, w, h)).transpose(1,2,0)
        ph, pw = self.tile_size - h, self.tile_size - w
        if ph or pw:
            tile = np.pad(tile, ((0,ph),(0,pw),(0,0)), mode='reflect')
        t = torch.from_numpy(np.ascontiguousarray(tile)).permute(2,0,1)  # CHW
        return t, x, y, w, h

# -------------------- Mean probability per polygon --------------------
def mean_probability_for_geometry(src_prob_ds, geom):
    """Mean prob inside polygon using a small window + mask (streaming-safe)."""
    try:
        win = geometry_window(src_prob_ds, [geom], pad_x=2, pad_y=2,
                              north_up=True, pixel_precision=3)
        block = src_prob_ds.read(1, window=win, boundless=True, fill_value=0)
        mask = geometry_mask([geom],
                             out_shape=(win.height, win.width),
                             transform=win_transform(win, src_prob_ds.transform),
                             invert=True)
        vals = block[mask]
        return float(vals.mean()) if vals.size else 0.0
    except Exception:
        # rare fallback
        block = src_prob_ds.read(1)
        mask = geometry_mask([geom], out_shape=(src_prob_ds.height, src_prob_ds.width),
                             transform=src_prob_ds.transform, invert=True)
        vals = block[mask]
        return float(vals.mean()) if vals.size else 0.0

# -------------------- Polygonize + attach mean_prob --------------------
def polygonize_with_meanprob(prob_raster_path, out_path, thresh):
    """
    1) Threshold prob raster to binary (stream).
    2) GDAL Polygonize → temporary vector.
    3) Read with GeoPandas, compute mean_prob per polygon, write final file.
    """
    from osgeo import gdal, ogr, osr

    out_path = Path(out_path)
    prob_raster_path = Path(prob_raster_path)

    tmp_bin = out_path.with_suffix('.bin.tif.tmp')
    if out_path.suffix.lower() == '.gpkg':
        tmp_vec = out_path.with_name(out_path.stem + '_tmp.gpkg')
    else:
        tmp_vec = out_path.with_name(out_path.stem + '_tmp.shp')

    print(f"-> Vectorizing @ threshold={thresh} (GDAL Polygonize) …")

    # 1) write temporary binary raster blockwise
    with rasterio.open(prob_raster_path) as src:
        meta = src.meta.copy()
        meta.update(dtype='uint8')
        with rasterio.open(tmp_bin, 'w', **meta) as dst:
            for _, win in tqdm(src.block_windows(1), desc="   binarize", total=None):
                p = src.read(1, window=win)
                dst.write((p >= thresh).astype(np.uint8), 1, window=win)

    # 2) polygonize
    driver_name = 'GPKG' if tmp_vec.suffix.lower() == '.gpkg' else 'ESRI Shapefile'
    src_ds = gdal.Open(str(tmp_bin), gdal.GA_ReadOnly)
    srcband = src_ds.GetRasterBand(1)

    drv = ogr.GetDriverByName(driver_name)
    if tmp_vec.exists():
        try: drv.DeleteDataSource(str(tmp_vec))
        except Exception: pass
    dst_ds = drv.CreateDataSource(str(tmp_vec))
    srs = osr.SpatialReference(); srs.ImportFromWkt(src_ds.GetProjection())
    layer_name = tmp_vec.stem
    layer = dst_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbMultiPolygon)
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
    gdal.SetConfigOption('GDAL_CACHEMAX', str(GDAL_CACHE_MB))
    gdal.Polygonize(srcband, srcband, layer, 0, [], callback=None)
    layer = None; dst_ds = None; src_ds = None

    # 3) read, filter, attach mean prob, save final
    gdf = gpd.read_file(tmp_vec)
    if 'value' in gdf.columns:
        gdf = gdf[gdf['value'] == 1].drop(columns=['value'])

    if len(gdf) == 0:
        print("   -> No polygons at this threshold.")
        tmp_bin.unlink(missing_ok=True)
        if driver_name == 'ESRI Shapefile':
            for f in tmp_vec.parent.glob(tmp_vec.stem + '.*'):
                f.unlink(missing_ok=True)
        else:
            tmp_vec.unlink(missing_ok=True)
        return

    with rasterio.open(prob_raster_path) as srcp:
        means = []
        for geom in tqdm(gdf.geometry, desc="   mean_prob"):
            means.append(mean_probability_for_geometry(srcp, geom))
    gdf['mean_prob'] = np.asarray(means, dtype=np.float32).round(4)

    # Shapefile field-name safety
    driver_out = 'GPKG' if EXPORT_GPKG else 'ESRI Shapefile'
    if driver_out == 'ESRI Shapefile':
        gdf = gdf.rename(columns={'mean_prob': 'meanprob'})

    gdf.to_file(out_path, driver=driver_out)
    print(f"   -> Saved: {out_path}  ({len(gdf)} polygons)")

    # cleanup temps
    tmp_bin.unlink(missing_ok=True)
    if driver_name == 'ESRI Shapefile':
        for f in tmp_vec.parent.glob(tmp_vec.stem + '.*'):
            f.unlink(missing_ok=True)
    else:
        tmp_vec.unlink(missing_ok=True)

# --------------------------- Main ---------------------------
def main():
    # env / perf hints
    os.environ.setdefault("GDAL_CACHEMAX", str(GDAL_CACHE_MB))
    os.environ.setdefault("GDAL_NUM_THREADS", GDAL_THREADS)
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    # Paths
    host_in   = Path(INPUT_PATH_ON_HOST)
    host_base = Path(OUTPUT_BASENAME_ON_HOST)
    out_dir   = host_base.parent; out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(CONTAINER_WORK_DIR)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    ext            = ".gpkg" if EXPORT_GPKG else ".shp"
    local_in       = work_dir / host_in.name
    local_base     = work_dir / host_base.name
    prob_raster    = local_base.with_name(local_base.name + "_prob.tif")
    sum_path       = local_base.with_name(local_base.name + ".sum.fp16.mmap")
    wgt_path       = local_base.with_name(local_base.name + ".wgt.fp16.mmap")
    local_vectors  = local_base.with_suffix(ext)
    host_vectors   = out_dir / local_vectors.name

    # 1) copy input to /tmp
    print("\n-> [STEP 1/6] Copying input COG to fast storage …")
    copy_with_progress(host_in, local_in); drop_file_cache(str(local_in))

    # 2) load model + preproc
    print("\n-> [STEP 2/6] Loading model …")
    cfg = Config.fromfile(CONFIG_FILE)
    model = init_model(cfg, CHECKPOINT_FILE, device=DEVICE).eval()
    model.to(memory_format=torch.channels_last)

    dp = cfg.model.get('data_preprocessor', cfg.get('data_preprocessor', None))
    if dp is None:
        raise RuntimeError("No data_preprocessor in config.")
    mean_np = np.array(dp['mean'], dtype=np.float32)
    std_np  = np.array(dp['std'],  dtype=np.float32)
    to_rgb  = bool(dp.get('bgr_to_rgb', False))
    mean_gpu = torch.tensor(mean_np, dtype=torch.float16, device=DEVICE).view(1,3,1,1)
    std_gpu  = torch.tensor(std_np,  dtype=torch.float16, device=DEVICE).view(1,3,1,1)

    # 3) dataset / memmaps
    print("\n-> [STEP 3/6] Preparing DataLoader & accumulators …")
    dataset = TiledGeoTiffDataset(local_in, TILE_SIZE, OVERLAP)
    with rasterio.open(local_in) as src:
        width, height = src.width, src.height
        meta_prob = src.meta.copy()
        meta_prob.update(count=1, dtype='float32', tiled=True,
                         blockxsize=512, blockysize=512,
                         compress='lzw', BIGTIFF='IF_SAFER')

    sum_mm = np.memmap(sum_path, dtype=np.float16, mode='w+', shape=(height, width))
    wgt_mm = np.memmap(wgt_path, dtype=np.float16, mode='w+', shape=(height, width))

    # auto-tune inflight batches for pinned memory
    bytes_per_tile = TILE_SIZE*TILE_SIZE*3*dtype_bytes(dataset.dtype)
    max_inflight   = max(1, int((TARGET_PINNED_BUDGET_GB*(1024**3)) // (bytes_per_tile * BATCH_SIZE)))
    tuned_workers  = min(NUM_WORKERS, max_inflight)
    tuned_prefetch = max(1, min(PREFETCH_FACTOR, max(1, max_inflight // max(1, tuned_workers))))
    tuned_pin      = PIN_MEMORY

    def build_loader(workers, prefetch, batch, pin):
        return DataLoader(dataset, batch_size=batch, shuffle=False,
                          num_workers=workers, pin_memory=pin, drop_last=False,
                          persistent_workers=(workers > 0), prefetch_factor=prefetch,
                          worker_init_fn=worker_init_fn)

    loader = build_loader(tuned_workers, tuned_prefetch, BATCH_SIZE, tuned_pin)

    # 4) inference (center-crop writer → NO seams)
    print(f"\n-> [STEP 4/6] Inference with {tuned_workers} workers × batch {BATCH_SIZE} "
          f"(prefetch={tuned_prefetch}, pin={tuned_pin}) …")

    margin = OVERLAP // 2  # inner region to write
    def run_loop(dl):
        pbar = tqdm(total=len(dl.dataset), desc="Tiles inferred", unit="tile")
        for tiles, bx, by, bw, bh in dl:
            # dtype scaling
            if tiles.dtype == torch.uint8:   scale = 1/255.0
            elif tiles.dtype == torch.uint16: scale = 1/65535.0
            else:                             scale = 1.0

            imgs = tiles.to(DEVICE, non_blocking=True).to(
                torch.float16, memory_format=torch.channels_last) * scale
            if to_rgb:
                imgs = imgs[:, [2,1,0], ...]
            imgs = (imgs - mean_gpu * scale) / (std_gpu * scale)
            metas = [dict(img_shape=(TILE_SIZE, TILE_SIZE),
                          ori_shape=(TILE_SIZE, TILE_SIZE))] * imgs.shape[0]

            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model.encode_decode(imgs, metas)          # (B,C,H,W)
                probs  = torch.softmax(logits, dim=1).float().cpu().numpy()

            # write ONLY the center region of each tile
            bx = bx.numpy(); by = by.numpy(); bw = bw.numpy(); bh = bh.numpy()
            for i in range(imgs.shape[0]):
                x, y, w, h = int(bx[i]), int(by[i]), int(bw[i]), int(bh[i])

                # dynamic crop (avoid cropping against image boundary)
                l = 0 if x == 0 else margin
                t = 0 if y == 0 else margin
                r = 0 if (x + w) >= width  else margin
                b = 0 if (y + h) >= height else margin
                ly0, ly1 = t, h - b
                lx0, lx1 = l, w - r
                if lx0 >= lx1 or ly0 >= ly1:
                    continue

                gy0, gy1 = y + ly0, y + ly1
                gx0, gx1 = x + lx0, x + lx1
                p = probs[i, TARGET_CLASS_ID, ly0:ly1, lx0:lx1]

                sum_mm[gy0:gy1, gx0:gx1] += p.astype(np.float16)
                wgt_mm[gy0:gy1, gx0:gx1] += np.float16(1.0)

            pbar.update(imgs.shape[0])
        pbar.close()

    try:
        run_loop(loader)
    except RuntimeError as e:
        if "pin memory" in str(e).lower() or "out of memory" in str(e).lower():
            print("\n!! DataLoader OOM → fallback (workers=2, half batch, no pin).")
            gc.collect(); torch.cuda.empty_cache()
            loader = build_loader(2, 1, max(1, BATCH_SIZE//2), False)
            run_loop(loader)
        else:
            raise

    sum_mm.flush(); wgt_mm.flush()
    drop_file_cache(str(local_in))

    # 5) finalize probability raster
    print("\n-> [STEP 5/6] Finalizing probability raster …")
    with rasterio.open(prob_raster, 'w', **meta_prob) as dst:
        for y0 in tqdm(range(0, height, FINALIZE_BLOCK), desc="Finalize"):
            h = min(FINALIZE_BLOCK, height - y0)
            sblk = np.asarray(sum_mm[y0:y0+h, :], dtype=np.float32)
            wblk = np.asarray(wgt_mm[y0:y0+h, :], dtype=np.float32)
            out  = np.zeros_like(sblk, dtype=np.float32)
            pos  = (wblk > 1e-6)
            if np.any(pos):
                out[pos] = sblk[pos] / wblk[pos]
            dst.write(out, 1, window=Window(0, y0, width, h))

    del sum_mm, wgt_mm; gc.collect(); torch.cuda.empty_cache()
    sum_path.unlink(missing_ok=True)
    wgt_path.unlink(missing_ok=True)
    drop_file_cache(str(prob_raster))

    # 6) vectorize + copy out
    print("\n-> [STEP 6/6] Vectorizing and attaching mean_prob …")
    polygonize_with_meanprob(str(prob_raster), str(local_vectors), THRESH)

    # copy results back to host
    stem = local_vectors.stem
    for piece in work_dir.glob(f"{stem}.*"):
        shutil.copy(piece, out_dir / piece.name)
        print("   ", out_dir / piece.name)

    # optional cleanup
    if DELETE_WORK_DIR:
        try:
            shutil.rmtree(work_dir)
            print(f"\nScratch deleted: {work_dir}")
        except Exception as e:
            print(f"\n(Warning) Could not delete {work_dir}: {e}")
    else:
        print(f"\nScratch kept: {work_dir}")

    print("\n--- Pipeline finished successfully! ---")

if __name__ == '__main__':
    # safer for DataLoader on Windows/WSL
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()