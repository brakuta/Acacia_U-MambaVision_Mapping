"""Properties of the tiling grid and centre-crop blending (no GPU, no rasterio)."""
import numpy as np
import pytest

from geoseg.inference.tiling import TileGrid, hann2d


@pytest.mark.parametrize('w,h,tile,overlap', [
    (1024, 1024, 1024, 256), (2300, 1700, 1024, 256), (1030, 1030, 1024, 256),
    (900, 700, 1024, 256), (3000, 3000, 512, 128), (2049, 1025, 1024, 0),
])
def test_center_crop_regions_partition_the_raster(w, h, tile, overlap):
    grid = TileGrid(w, h, tile, overlap)
    weight = np.zeros((h, w), dtype=np.int32)
    for idx, t in enumerate(grid.tiles):
        assert t.x + t.w <= w and t.y + t.h <= h
        assert t.w == min(tile, w) and t.h == min(tile, h)  # edge-aligned: full tiles
        b = grid.write_bounds(idx)
        if b is None:
            continue
        y0, y1, x0, x1 = b
        assert t.y <= y0 <= y1 <= t.y + t.h and t.x <= x0 <= x1 <= t.x + t.w
        weight[y0:y1, x0:x1] += 1
    # every pixel written exactly once -> no gaps, no double counting, no seams
    assert weight.min() == 1 and weight.max() == 1


def test_grid_rejects_bad_overlap():
    with pytest.raises(ValueError):
        TileGrid(100, 100, 64, 64)


def test_hann_window_positive_and_symmetric():
    win = hann2d(64)
    assert win.shape == (64, 64) and win.min() > 0
    assert np.allclose(win, win.T) and np.allclose(win, win[::-1, ::-1])
