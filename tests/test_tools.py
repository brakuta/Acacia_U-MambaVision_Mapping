"""Tests of the general-purpose data and experiment tools."""
import os

import numpy as np
import pytest
import torch

rasterio = pytest.importorskip('rasterio')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
