"""Raster loading transforms for MMSegmentation based on rasterio.

MMSegmentation's default ``LoadImageFromFile`` decodes images with OpenCV,
which handles 8-bit, 1- or 3-band files only and returns BGR order.  Its
``LoadSingleRSImageFromFile`` reads any GeoTIFF through GDAL but offers no
band selection or radiometric scaling.  :class:`LoadRasterioImage` covers
the remaining needs of geospatial projects:

* any number of bands (RGB, RGB+NIR, multispectral, SAR stacks);
* band selection and re-ordering (``bands=[3, 2, 1]`` for BGR-stored files);
* 16-bit or float inputs scaled to a common range (``scale=1/10000``);
* nodata handling (``nodata_fill``);
* no channel-order surprises: bands are returned in the order requested.

Configuration example (Sentinel-2 style 10-band, 16-bit tiles)::

    train_pipeline = [
        dict(type='LoadRasterioImage', bands=[1,2,3,4,5,6,7,8,9,10], scale=1/10000.0),
        dict(type='LoadAnnotations'),
        ...
    ]
    data_preprocessor = dict(type='SegDataPreProcessor', size=crop_size,
                             mean=[0.1]*10, std=[0.1]*10, bgr_to_rgb=False)
    model = dict(backbone=dict(in_channels=10, ...))

Set ``bgr_to_rgb=False`` in the data preprocessor: this loader never
produces BGR.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadRasterioImage(BaseTransform):
    """Load a (multi-band) raster with rasterio.

    Required keys: ``img_path``. Added/modified keys: ``img`` (H, W, C
    float32), ``img_shape``, ``ori_shape``, ``num_bands``.

    Args:
        bands: 1-based band indices to read, in output order. ``None`` reads
            all bands in file order.
        scale: Multiplicative factor applied after reading (e.g. ``1/255``
            or ``1/10000``). ``None`` leaves values unchanged.
        offset: Additive offset applied after scaling.
        nodata_fill: Value written where the raster's nodata mask is set.
        clip: Optional ``(low, high)`` clipping range applied last.
        to_float32: Cast to float32 (recommended; the preprocessor expects
            float or uint8).
    """

    def __init__(self,
                 bands: Optional[Sequence[int]] = None,
                 scale: Optional[float] = None,
                 offset: float = 0.0,
                 nodata_fill: Optional[float] = 0.0,
                 clip: Optional[Sequence[float]] = None,
                 to_float32: bool = True) -> None:
        self.bands = list(bands) if bands is not None else None
        self.scale = scale
        self.offset = offset
        self.nodata_fill = nodata_fill
        self.clip = tuple(clip) if clip is not None else None
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict:
        import rasterio
        with rasterio.open(results['img_path']) as src:
            bands = self.bands or list(range(1, src.count + 1))
            arr = src.read(bands)  # (C, H, W)
            if self.nodata_fill is not None and src.nodata is not None:
                mask = src.read_masks(bands) == 0
                arr = arr.astype(np.float32)
                arr[mask] = self.nodata_fill
        img = np.transpose(arr, (1, 2, 0))
        if self.to_float32 or self.scale is not None or self.offset:
            img = img.astype(np.float32)
        if self.scale is not None:
            img = img * np.float32(self.scale)
        if self.offset:
            img = img + np.float32(self.offset)
        if self.clip is not None:
            img = np.clip(img, self.clip[0], self.clip[1])
        results['img'] = np.ascontiguousarray(img)
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results['num_bands'] = img.shape[2]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(bands={self.bands}, scale={self.scale}, '
                f'offset={self.offset}, nodata_fill={self.nodata_fill}, clip={self.clip})')
