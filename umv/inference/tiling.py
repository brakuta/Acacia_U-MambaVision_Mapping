"""Tiling utilities for sliding-window inference over large GeoTIFFs.

Grid construction
-----------------
Tile origins are placed every ``stride = tile_size - overlap`` pixels; the last
tile of every row/column is shifted so that it ends exactly at the raster
border (edge-aligned).  Consequently every tile has full spatial context and
padding is only needed when the raster itself is smaller than a tile.

Seam-free (centre-crop) blending
--------------------------------
Each tile writes only the region between the mid-points of its overlaps with
the neighbouring tiles.  The written regions partition the raster exactly
(every pixel is predicted once, from the tile in which it is farthest from
the border), which removes tile seams without any weighting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Tile:
    x: int  # column offset (pixels)
    y: int  # row offset (pixels)
    w: int  # valid width  (== tile_size unless the raster is narrower)
    h: int  # valid height (== tile_size unless the raster is shorter)


def _origins(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    xs = list(range(0, length - tile, stride))
    xs.append(length - tile)  # edge-aligned last tile
    return xs


def _write_bounds(origins: List[int], sizes: List[int], length: int) -> List[Tuple[int, int]]:
    """Per-origin [start, end) write ranges from overlap mid-points."""
    bounds = []
    for j, (o, s) in enumerate(zip(origins, sizes)):
        start = 0 if j == 0 else (origins[j] + origins[j - 1] + sizes[j - 1]) // 2
        end = length if j == len(origins) - 1 else (origins[j + 1] + o + s) // 2
        bounds.append((start, end))
    return bounds


class TileGrid:
    """Regular, edge-aligned grid of overlapping tiles covering a raster."""

    def __init__(self, width: int, height: int, tile_size: int, overlap: int) -> None:
        if not 0 <= overlap < tile_size:
            raise ValueError('overlap must satisfy 0 <= overlap < tile_size')
        self.width, self.height = int(width), int(height)
        self.tile_size, self.overlap = int(tile_size), int(overlap)
        self.stride = self.tile_size - self.overlap

        xs = _origins(self.width, self.tile_size, self.stride)
        ys = _origins(self.height, self.tile_size, self.stride)
        ws = [min(self.tile_size, self.width - x) for x in xs]
        hs = [min(self.tile_size, self.height - y) for y in ys]
        xb = _write_bounds(xs, ws, self.width)
        yb = _write_bounds(ys, hs, self.height)

        self.tiles: List[Tile] = []
        self._bounds: List[Tuple[int, int, int, int]] = []  # global (y0, y1, x0, x1)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                self.tiles.append(Tile(x, y, ws[i], hs[j]))
                self._bounds.append((yb[j][0], yb[j][1], xb[i][0], xb[i][1]))

    def __len__(self) -> int:
        return len(self.tiles)

    def write_bounds(self, idx: int) -> Optional[Tuple[int, int, int, int]]:
        """Global ``(y0, y1, x0, x1)`` region owned by tile ``idx`` (or None)."""
        y0, y1, x0, x1 = self._bounds[idx]
        if y0 >= y1 or x0 >= x1:
            return None
        return y0, y1, x0, x1

    def center_crop_bounds(self, idx: int) -> Optional[Tuple[int, int, int, int]]:
        """Same region as :meth:`write_bounds` but in tile-local coordinates."""
        b = self.write_bounds(idx)
        if b is None:
            return None
        t = self.tiles[idx]
        return b[0] - t.y, b[1] - t.y, b[2] - t.x, b[3] - t.x


def hann2d(size: int, floor: float = 1e-4) -> np.ndarray:
    """Separable 2-D Hann window (clipped away from zero for weighting)."""
    if size <= 1:
        return np.ones((size, size), dtype=np.float32)
    w = np.hanning(size).astype(np.float32)
    return np.clip(np.outer(w, w), floor, None)


class TiledRasterDataset(Dataset):
    """Yields ``(tile_tensor[C,T,T], idx)`` for each grid tile.

    Tiles narrower/shorter than ``tile_size`` (raster smaller than a tile) are
    reflect-padded.  Each DataLoader worker opens its own raster handle
    lazily, which keeps the object picklable for ``spawn`` contexts.
    """

    def __init__(self, raster_path: str, grid: TileGrid, bands: Sequence[int] = (1, 2, 3),
                 band_order: str = 'rgb') -> None:
        self.raster_path = str(raster_path)
        self.grid = grid
        self.bands = list(bands)
        self.flip = band_order.lower() == 'bgr'
        self._ds = None

    def __len__(self) -> int:
        return len(self.grid)

    def _ensure_open(self):
        if self._ds is None:
            import rasterio
            self._ds = rasterio.open(self.raster_path)
        return self._ds

    def __getitem__(self, idx: int):
        from rasterio.windows import Window
        ds = self._ensure_open()
        t = self.grid.tiles[idx]
        arr = ds.read(self.bands, window=Window(t.x, t.y, t.w, t.h))  # C,H,W
        if self.flip:
            arr = arr[::-1]
        ph, pw = self.grid.tile_size - t.h, self.grid.tile_size - t.w
        if ph or pw:
            mode = 'reflect' if (ph < t.h and pw < t.w) else 'edge'
            arr = np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode=mode)
        return torch.from_numpy(np.ascontiguousarray(arr)), idx
