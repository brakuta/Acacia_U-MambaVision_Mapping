"""Every template/example config must load, build on CPU and run a training step on tiny synthetic data."""
import glob
import os

import numpy as np
import pytest
import torch

mmseg = pytest.importorskip('mmseg')
rasterio = pytest.importorskip('rasterio')

from mmengine.structures import PixelData  # noqa: E402
from mmseg.registry import MODELS  # noqa: E402
from mmseg.structures import SegDataSample  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402

from mmengine.config import Config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'templates', '**', 'configs', '*.py'), recursive=True)
                 + glob.glob(os.path.join(ROOT, 'examples', '**', 'configs', '*.py'), recursive=True))


@pytest.fixture(scope='module', autouse=True)
def _scope():
    register_all_modules(init_default_scope=True)


@pytest.mark.parametrize('path', CONFIGS, ids=[os.path.relpath(c, ROOT) for c in CONFIGS])
def test_config_loads_and_model_trains_one_step(path):
    cfg = Config.fromfile(path)
    n_cls = len(cfg.train_dataloader.dataset.metainfo['classes'])
    assert cfg.model.decode_head.num_classes == n_cls
    # do not download pretrained weights in the test
    cfg.model.pretrained = None
    cfg.model.backbone.init_cfg = None
    model = MODELS.build(cfg.model)
    model.init_weights()
    in_ch = cfg.model.backbone.get('in_channels', 3)
    size = 64
    x = torch.rand(2, in_ch, size, size)
    samples = []
    for _ in range(2):
        s = SegDataSample(metainfo=dict(img_shape=(size, size), ori_shape=(size, size), pad_shape=(size, size)))
        s.gt_sem_seg = PixelData(data=torch.randint(0, n_cls, (1, size, size)))
        samples.append(s)
    model.train()
    losses = model(x, samples, mode='loss')
    total = sum(v for k, v in losses.items() if 'loss' in k)
    assert torch.isfinite(total)
    total.backward()
    model.eval()
    with torch.no_grad():
        preds = model(x[:1], samples[:1], mode='predict')
    assert tuple(preds[0].pred_sem_seg.data.shape) == (1, size, size)


def test_multispectral_pipeline_end_to_end(tmp_path):
    """LoadRasterioImage -> augmentation -> PackSegInputs -> data_preprocessor -> UPerNet(10 bands)."""
    cfg = Config.fromfile(os.path.join(ROOT, 'examples/lulc_multispectral/configs/upernet_r50_10band.py'))
    root = tmp_path / 'lulc'
    for d in ('img_dir/train', 'ann_dir/train'):
        (root / d).mkdir(parents=True)
    for i in range(3):
        img = np.random.randint(0, 12000, (10, 96, 96), dtype=np.uint16)
        with rasterio.open(root / f'img_dir/train/t{i}.tif', 'w', driver='GTiff', width=96, height=96, count=10, dtype='uint16') as dst:
            dst.write(img)
        m = np.random.randint(0, 7, (1, 96, 96), dtype=np.uint8); m[0, :8, :8] = 255
        with rasterio.open(root / f'ann_dir/train/t{i}.tif', 'w', driver='GTiff', width=96, height=96, count=1, dtype='uint8') as dst:
            dst.write(m)
    from mmseg.registry import DATASETS
    ds_cfg = cfg.train_dataloader.dataset
    ds_cfg.data_root = str(root)
    ds_cfg.pipeline[3]['crop_size'] = (64, 64)      # RandomCrop
    ds_cfg.pipeline[2]['scale'] = (64, 64)          # RandomResize
    ds_cfg.pipeline[2]['ratio_range'] = (1.0, 1.0)  # deterministic size for the batch below
    ds = DATASETS.build(ds_cfg)
    assert len(ds) == 3
    item = ds[0]
    # RandomResize may yield 63 px before the crop; the data preprocessor pads to the crop size
    assert item['inputs'].shape[0] == 10 and max(item['inputs'].shape[1:]) <= 64 and item['inputs'].dtype == torch.float32
    assert item['data_samples'].gt_sem_seg.data.shape[-2:] == tuple(item['inputs'].shape[1:])
    # batch through the data preprocessor and the 10-band model
    cfg.model.pretrained = None; cfg.model.backbone.init_cfg = None
    cfg.model.data_preprocessor.size = (64, 64)
    model = MODELS.build(cfg.model)
    batch = model.data_preprocessor(dict(inputs=[ds[0]['inputs'], ds[1]['inputs']],
                                         data_samples=[ds[0]['data_samples'], ds[1]['data_samples']]), training=True)
    assert tuple(batch['inputs'].shape) == (2, 10, 64, 64)
    losses = model(batch['inputs'], batch['data_samples'], mode='loss')
    assert all(torch.isfinite(v) for k, v in losses.items() if 'loss' in k)


def test_compare_runs_and_adapt_first_conv(tmp_path):
    import json, subprocess, sys  # noqa: E401
    run = tmp_path / 'work_dirs' / 'proj' / 'modelA' / '20260101_000000' / 'vis_data'
    run.mkdir(parents=True)
    rows = [dict(step=100, loss=0.9), dict(step=2000, mIoU=70.0, mFscore=80.0), dict(step=4000, mIoU=75.5, mFscore=84.1),
            dict(step=6000, mIoU=74.0, mFscore=83.0)]
    (run / 'scalars.json').write_text('\n'.join(json.dumps(r) for r in rows))
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools/compare_runs.py'), str(tmp_path / 'work_dirs'),
                          '--out', str(tmp_path / 's.md')], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert '75.50' in out.stdout and '4000' in out.stdout and 'modelA' in out.stdout
    # first-conv adaptation
    sd = {'stem.0.weight': torch.randn(32, 3, 3, 3), 'other': torch.zeros(2)}
    src = tmp_path / 'r50.pth'; torch.save({'state_dict': sd}, src)
    out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools/adapt_first_conv.py'), str(src), str(tmp_path / 'r50_10.pth'),
                          '--key', 'stem.0.weight', '--bands', '10', '--mode', 'map', '--rgb-bands', '2', '1', '0'],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    new = torch.load(tmp_path / 'r50_10.pth', weights_only=False)['state_dict']['stem.0.weight']
    assert tuple(new.shape) == (32, 10, 3, 3)
    assert torch.allclose(new[:, 2], sd['stem.0.weight'][:, 0]) and torch.allclose(new[:, 0], sd['stem.0.weight'][:, 2])


def test_make_tiles_spatial_split(tmp_path):
    import subprocess, sys  # noqa: E401
    W = H = 1024
    img = np.random.randint(0, 255, (3, H, W), dtype=np.uint8)
    mask = np.zeros((1, H, W), dtype=np.uint8); mask[0, 200:600, 300:700] = 1; mask[0, :64, :] = 255
    tr = rasterio.transform.from_origin(500000, 2500000, 0.05, 0.05)
    for name, arr, dt in (('ortho.tif', img, 'uint8'), ('labels.tif', mask, 'uint8')):
        with rasterio.open(tmp_path / name, 'w', driver='GTiff', width=W, height=H, count=arr.shape[0], dtype=dt,
                           crs='EPSG:32640', transform=tr) as d:
            d.write(arr)
    out = tmp_path / 'ds'
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'tools/make_tiles.py'), '--image', str(tmp_path / 'ortho.tif'),
                        '--mask', str(tmp_path / 'labels.tif'), '--out', str(out), '--tile', '256', '--split-blocks', '2',
                        '--val-frac', '0.25', '--test-frac', '0.25'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    n = {s: len(list((out / 'img_dir' / s).glob('*.tif'))) for s in ('train', 'val', 'test')}
    assert sum(n.values()) == 16 and n['val'] == 4 and n['test'] == 4
    t = next((out / 'img_dir' / 'train').glob('*.tif'))
    with rasterio.open(t) as d, rasterio.open(out / 'ann_dir' / 'train' / t.name) as m:
        assert d.crs.to_epsg() == 32640 and d.count == 3 and m.count == 1 and set(np.unique(m.read(1))) <= {0, 1, 255}
    assert (out / 'tiles_index.csv').exists()
