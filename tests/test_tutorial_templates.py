"""Every tutorial config must load, build on CPU and run a training step on tiny synthetic data.

The MambaVision-based config is loaded only (its encoder needs CUDA kernels)."""
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

from umv.compat import load_config  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = sorted(glob.glob(os.path.join(ROOT, 'tutorial', '**', 'configs', '*.py'), recursive=True))


@pytest.fixture(scope='module', autouse=True)
def _scope():
    register_all_modules(init_default_scope=True)


@pytest.mark.parametrize('path', CONFIGS, ids=[os.path.relpath(c, ROOT) for c in CONFIGS])
def test_config_loads_and_model_trains_one_step(path):
    cfg = load_config(path)
    n_cls = len(cfg.train_dataloader.dataset.metainfo['classes'])
    assert cfg.model.decode_head.num_classes == n_cls
    if cfg.model.backbone.type.startswith('MambaVision'):
        return  # needs CUDA kernels
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
    cfg = load_config(os.path.join(ROOT, 'tutorial/examples/lulc_multispectral/configs/upernet_r50_10band.py'))
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
    ds = DATASETS.build(ds_cfg)
    assert len(ds) == 3
    item = ds[0]
    assert tuple(item['inputs'].shape) == (10, 64, 64) and item['inputs'].dtype == torch.float32
    assert item['data_samples'].gt_sem_seg.data.shape[-2:] == (64, 64)
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
