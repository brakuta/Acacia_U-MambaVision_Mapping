import torch

from umv.checkpoint import detect_variant
from umv.models.mamba_vision import VARIANTS, resolve_variant


def test_detect_variant_from_decoder_stage0():
    for name, v in VARIANTS.items():
        c = v['channels']
        sd = {'decode_head.decoder_stages.0.0.weight': torch.zeros(256, c[-1] + c[-2], 3, 3)}
        assert detect_variant(sd) == name
    assert detect_variant({}) is None


def test_resolve_variant_aliases():
    assert resolve_variant('T') == 'tiny' and resolve_variant('s') == 'small' and resolve_variant('Base') == 'base'


def test_resolve_checkpoint_prefers_best(tmp_path):
    from umv.checkpoint import resolve_checkpoint
    for n in ('best_mIoU_iter_85000.pth', 'best_mIoU_iter_95000.pth', 'iter_100000.pth'):
        (tmp_path / n).write_bytes(b'x')
    (tmp_path / 'last_checkpoint').write_text(str(tmp_path / 'iter_100000.pth'))
    assert resolve_checkpoint(str(tmp_path)).endswith('best_mIoU_iter_95000.pth')
    assert resolve_checkpoint(str(tmp_path), prefer='last').endswith('iter_100000.pth')
    assert resolve_checkpoint(str(tmp_path / 'iter_100000.pth')).endswith('iter_100000.pth')
    (tmp_path / 'best_mIoU_iter_85000.pth').unlink(); (tmp_path / 'best_mIoU_iter_95000.pth').unlink()
    assert resolve_checkpoint(str(tmp_path)).endswith('iter_100000.pth')  # falls back to last
