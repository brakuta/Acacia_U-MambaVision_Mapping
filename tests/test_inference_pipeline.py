"""End-to-end sliding-window inference on a synthetic GeoTIFF with a stub
segmentor.  Requires rasterio/geopandas; runs on CPU in a few seconds."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip('rasterio')
gpd = pytest.importorskip('geopandas')

from umv.inference.pipeline import InferenceSettings, segment_raster  # noqa: E402


class BrightnessSegmentor(torch.nn.Module):
    """Predicts 'acacia' where the (normalised) mean brightness is high.

    Deterministic and translation-invariant, so tile boundaries must not be
    visible in the output if the tiling/blending is correct.
    """

    def encode_decode(self, imgs, metas):
        b = imgs.mean(dim=1, keepdim=True)          # (B,1,H,W) normalised brightness
        logit_fg = (b - 0.5) * 8.0
        return torch.cat([-logit_fg, logit_fg], dim=1)


class FakeCfg(dict):
    def __init__(self):
        super().__init__()
        self.model = dict(data_preprocessor=dict(mean=[123.675, 116.28, 103.53],
                                                 std=[58.395, 57.12, 57.375], bgr_to_rgb=True))


def make_raster(path: Path, w=2300, h=1700, n_disks=12, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((3, h, w), 60, dtype=np.uint8)
    disks = []
    yy, xx = np.mgrid[0:h, 0:w]
    while len(disks) < n_disks:
        cx, cy, r = rng.integers(80, w - 80), rng.integers(80, h - 80), rng.integers(25, 70)
        if any((cx - a) ** 2 + (cy - b) ** 2 < (r + c + 12) ** 2 for a, b, c in disks):
            continue  # keep disks disjoint so each yields exactly one polygon
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[:, m] = 230
        disks.append((int(cx), int(cy), int(r)))
    transform = rasterio.transform.from_origin(400000.0, 2800000.0, 0.025, 0.025)  # 2.5 cm GSD
    with rasterio.open(path, 'w', driver='GTiff', width=w, height=h, count=3, dtype='uint8',
                       crs='EPSG:32640', transform=transform, tiled=True, blockxsize=256,
                       blockysize=256) as dst:
        dst.write(img)
    return disks


@pytest.mark.parametrize('blend', ['center', 'hann'])
def test_pipeline_recovers_disks(tmp_path, blend):
    raster = tmp_path / 'ortho.tif'
    disks = make_raster(raster)
    out = tmp_path / 'out' / 'ortho_crowns.gpkg'
    s = InferenceSettings(tile_size=512, overlap=128, batch_size=4, num_workers=0, blend=blend,
                          thresh=0.5, min_area=0.5, precision='fp32', prob_dtype='uint8',
                          save_prob=True)
    stats = segment_raster(BrightnessSegmentor().eval(), FakeCfg(), str(raster), str(out), s,
                           device='cpu', scratch_dir=str(tmp_path / 'scratch'))
    assert stats['polygons'] == len(disks)
    gdf = gpd.read_file(out)
    assert len(gdf) == len(disks) and gdf.crs.to_epsg() == 32640
    assert set(['area', 'mean_prob']).issubset(gdf.columns)
    assert (gdf['mean_prob'] > 0.9).all()
    # area check: pi r^2 * GSD^2 within a few percent for every disk
    areas = sorted(gdf['area'].tolist())
    expect = sorted(np.pi * r * r * 0.025 ** 2 for _, _, r in disks)
    assert np.allclose(areas, expect, rtol=0.05)
    # probability raster exists, matches input geometry, no seams (all 0 or 255 here)
    with rasterio.open(stats['prob_raster']) as p:
        assert (p.width, p.height) == (2300, 1700) and p.crs.to_epsg() == 32640
        arr = p.read(1)
        assert set(np.unique(arr)).issubset({0, 255})
    assert not (tmp_path / 'scratch').exists()  # scratch removed


def test_min_area_filter_and_shapefile(tmp_path):
    raster = tmp_path / 'ortho.tif'
    disks = make_raster(raster, w=1100, h=900, n_disks=6, seed=1)
    out = tmp_path / 'ortho_crowns.shp'
    big = sum(1 for _, _, r in disks if np.pi * r * r * 0.025 ** 2 >= 5.0)
    s = InferenceSettings(tile_size=512, overlap=128, batch_size=2, num_workers=0, thresh=0.5,
                          min_area=5.0, precision='fp32', output_format='shp', prob_dtype='float32')
    stats = segment_raster(BrightnessSegmentor().eval(), FakeCfg(), str(raster), str(out), s, device='cpu')
    assert stats['polygons'] == big
    if big:
        assert out.exists() and out.with_suffix('.prj').exists()
    json.dumps(stats)  # serialisable summary


def test_band_order_swap_changes_input(tmp_path):
    from umv.inference.tiling import TileGrid, TiledRasterDataset
    raster = tmp_path / 'rgb.tif'
    img = np.zeros((3, 64, 64), dtype=np.uint8); img[0] = 200  # red only
    with rasterio.open(raster, 'w', driver='GTiff', width=64, height=64, count=3, dtype='uint8') as dst:
        dst.write(img)
    grid = TileGrid(64, 64, 64, 0)
    t_rgb, idx = TiledRasterDataset(str(raster), grid, band_order='rgb')[0]
    t_bgr, _ = TiledRasterDataset(str(raster), grid, band_order='bgr')[0]
    assert idx == 0 and t_rgb.shape == (3, 64, 64)
    assert t_rgb[0].max() == 200 and t_rgb[2].max() == 0
    assert t_bgr[2].max() == 200 and t_bgr[0].max() == 0
