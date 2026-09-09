import numpy as np
import pytest

rasterio = pytest.importorskip('rasterio')
mmseg = pytest.importorskip('mmseg')

from geoseg.transforms import LoadRasterioImage  # noqa: E402


def _write(path, arr, nodata=None):
    c, h, w = arr.shape
    with rasterio.open(path, 'w', driver='GTiff', width=w, height=h, count=c, dtype=arr.dtype, nodata=nodata) as dst:
        dst.write(arr)


def test_multiband_uint16_scaling_and_band_order(tmp_path):
    arr = np.stack([np.full((8, 8), v, dtype=np.uint16) for v in (100, 2000, 30000, 5)])
    p = tmp_path / 'ms.tif'
    _write(p, arr)
    out = LoadRasterioImage(bands=[3, 2, 1, 4], scale=1 / 10000.0)(dict(img_path=str(p)))
    img = out['img']
    assert img.shape == (8, 8, 4) and img.dtype == np.float32 and out['num_bands'] == 4
    assert np.allclose(img[0, 0], [3.0, 0.2, 0.01, 0.0005])
    assert out['img_shape'] == (8, 8)


def test_nodata_fill_and_clip(tmp_path):
    arr = np.full((1, 4, 4), 500, dtype=np.uint16); arr[0, 0, 0] = 0
    p = tmp_path / 'nd.tif'
    _write(p, arr, nodata=0)
    out = LoadRasterioImage(nodata_fill=-1, clip=(0, 400))(dict(img_path=str(p)))
    assert out['img'][0, 0, 0] == 0  # nodata -> -1 -> clipped to 0
    assert out['img'][1, 1, 0] == 400


def test_registered_and_usable_in_pipeline(tmp_path):
    from mmseg.registry import TRANSFORMS
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)  # make 'mmseg' the default scope for Compose
    assert 'LoadRasterioImage' in TRANSFORMS.module_dict
    arr = np.random.randint(0, 255, (3, 16, 16), dtype=np.uint8)
    p = tmp_path / 'rgb.tif'; _write(p, arr)
    from mmcv.transforms import Compose
    pipe = Compose([dict(type='LoadRasterioImage'), dict(type='PackSegInputs')])
    res = pipe(dict(img_path=str(p)))
    assert tuple(res['inputs'].shape) == (3, 16, 16)
