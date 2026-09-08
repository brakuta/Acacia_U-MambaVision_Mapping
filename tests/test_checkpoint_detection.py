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
