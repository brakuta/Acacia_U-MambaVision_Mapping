"""Utilities for inspecting U-MV checkpoints (MMEngine format)."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

import torch

from .models.mamba_vision import VARIANTS

#: in_channels of decoder stage 0 (= C4 + C3) -> variant name.
_STAGE0_IN_CHANNELS = {
    v['channels'][-1] + v['channels'][-2]: name for name, v in VARIANTS.items()
}


def is_lfs_pointer(path: str) -> bool:
    """True if ``path`` is a Git LFS pointer file rather than the binary."""
    try:
        with open(path, 'rb') as f:
            head = f.read(64)
        return head.startswith(b'version https://git-lfs.github.com/spec/')
    except OSError:
        return False


def resolve_checkpoint(path: str, prefer: str = 'best') -> str:
    """Resolve a checkpoint path, accepting an MMSegmentation work directory.

    If ``path`` is a directory, ``best_mIoU_iter_*.pth`` is selected (the
    highest iteration if several exist).  With ``prefer='last'`` the file
    named in ``last_checkpoint`` (or the highest ``iter_*.pth``) is returned
    instead.  Regular files are returned unchanged.
    """
    import glob
    import os
    import re

    if not os.path.isdir(path):
        return path
    if prefer == 'best':
        cands = glob.glob(os.path.join(path, 'best_mIoU_iter_*.pth'))
        if cands:
            return max(cands, key=lambda p: int(re.findall(r'(\d+)\.pth$', p)[0]))
    last = os.path.join(path, 'last_checkpoint')
    if os.path.isfile(last):
        name = open(last).read().strip()
        cand = name if os.path.isabs(name) and os.path.isfile(name) else os.path.join(path, os.path.basename(name))
        if os.path.isfile(cand):
            return cand
    cands = glob.glob(os.path.join(path, 'iter_*.pth'))
    if cands:
        return max(cands, key=lambda p: int(re.findall(r'(\d+)\.pth$', p)[0]))
    raise FileNotFoundError(f'No best_mIoU_iter_*.pth, last_checkpoint or iter_*.pth found in {path}')


def load_checkpoint_file(path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Return ``(state_dict, meta)`` from an MMEngine/PyTorch checkpoint."""
    path = resolve_checkpoint(path)
    if is_lfs_pointer(path):
        raise RuntimeError(
            f'{path} is a Git LFS pointer, not a checkpoint. Run `git lfs install && '
            'git lfs pull` (see Pretrained_Weights/README.md) or point to a local .pth file.')
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only
        ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return OrderedDict(ckpt['state_dict']), dict(ckpt.get('meta', {}) or {})
    if isinstance(ckpt, dict):
        return OrderedDict(ckpt), {}
    raise ValueError(f'Unrecognised checkpoint structure in {path}')


def detect_variant(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    """Infer the MambaVision variant from the decoder's first stage.

    The first decoder convolution receives ``C4 + C3`` channels, which is
    960 (tiny), 1152 (small) or 1536 (base).
    """
    key = 'decode_head.decoder_stages.0.0.weight'
    if key not in state_dict:
        return None
    return _STAGE0_IN_CHANNELS.get(int(state_dict[key].shape[1]))


def summarize(path: str) -> Dict[str, Any]:
    sd, meta = load_checkpoint_file(path)
    n_params = sum(int(t.numel()) for t in sd.values() if torch.is_tensor(t))
    n_backbone = sum(int(t.numel()) for k, t in sd.items() if k.startswith('backbone.') and torch.is_tensor(t))
    n_head = sum(int(t.numel()) for k, t in sd.items() if k.startswith('decode_head.') and torch.is_tensor(t))
    prefixes = sorted({k.split('.')[0] for k in sd})
    return dict(
        path=path,
        variant=detect_variant(sd),
        num_tensors=len(sd),
        num_params=n_params,
        num_params_backbone=n_backbone,
        num_params_decode_head=n_head,
        top_level_prefixes=prefixes,
        iteration=meta.get('iter'),
        epoch=meta.get('epoch'),
        mmengine_version=meta.get('mmengine_version'),
        seed=meta.get('seed'),
        dataset_meta=meta.get('dataset_meta'),
        has_config=bool(meta.get('cfg')),
    )
