"""Legacy (work-dir) configs are translated to the new package on load."""
import textwrap

import pytest

mmseg = pytest.importorskip('mmseg')

from umv.compat import load_config  # noqa: E402

LEGACY = textwrap.dedent('''
    custom_imports = dict(imports=['mmseg.custom_models.mamba_vision',
                                   'mmseg.custom_models.generic_unet_head'], allow_failed_imports=False)
    dataset_type = 'ADE20KDataset'
    train_dataloader = dict(dataset=dict(type='ADE20KDataset', data_root='/data'))
    val_dataloader = dict(dataset=dict(type='ADE20KDataset', data_root='/data'))
    test_dataloader = dict(dataset=dict(type='ADE20KDataset', data_root='/data'))
    model = dict(type='EncoderDecoder', backbone=dict(type='mamba_small_vision_timm'),
                 decode_head=dict(type='GenericUNetHead', encoder_channels=[96, 192, 384, 768], num_classes=2))
''')


def test_legacy_config_is_translated(tmp_path):
    p = tmp_path / 'mambavision-s_generic-unet_acacia.py'
    p.write_text(LEGACY)
    cfg = load_config(str(p))
    assert cfg.custom_imports['imports'] == ['umv']
    for k in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
        assert cfg[k].dataset.type == 'UAVAcaciaDataset'
    from mmseg.registry import DATASETS, MODELS
    assert cfg.model.backbone.type in MODELS.module_dict
    assert cfg.model.decode_head.type in MODELS.module_dict
    assert 'UAVAcaciaDataset' in DATASETS.module_dict


def test_new_config_untouched():
    cfg = load_config('configs/mambavision/U-MV-small.py')
    assert cfg.model.backbone.type == 'MambaVisionBackbone'
    assert cfg.train_dataloader.dataset.type == 'UAVAcaciaDataset'
