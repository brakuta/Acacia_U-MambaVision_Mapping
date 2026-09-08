"""MambaVision backbone wrapper for MMSegmentation.

The backbone is the hierarchical hybrid Mamba-Transformer encoder of
Hatamizadeh and Kautz (2025), obtained from the Hugging Face Hub
(``nvidia/MambaVision-{T,S,B}-1K``) with ImageNet-1K pre-trained parameters.
The Hugging Face model returns ``(pooled_features, stage_features)``; the
wrapper exposes the four stage feature maps (strides 4, 8, 16 and 32) to the
decode head.

Checkpoint compatibility
------------------------
The wrapped model is stored in the attribute ``self.backbone`` so that the
parameter names of the released U-MV checkpoints
(``backbone.backbone.<hf-parameter>``) are preserved.  Do not rename this
attribute.

Offline use
-----------
The Hugging Face model code and weights are cached under ``$HF_HOME``
(default ``~/.cache/huggingface``).  Once cached, the backbone can be built
without network access by setting ``local_files_only=True`` in the config or
by exporting ``HF_HUB_OFFLINE=1``.  When a fine-tuned U-MV checkpoint is
loaded afterwards, ``pretrained=False`` skips the ImageNet weight download
and instantiates the architecture from its configuration only (the cached
remote code is still required).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from mmengine.logging import print_log
from mmengine.model import BaseModule
from mmseg.registry import MODELS

#: Variant metadata: Hugging Face model identifier and stage channel widths.
VARIANTS: Dict[str, Dict] = {
    'tiny': dict(hf_id='nvidia/MambaVision-T-1K', channels=(80, 160, 320, 640)),
    'small': dict(hf_id='nvidia/MambaVision-S-1K', channels=(96, 192, 384, 768)),
    'base': dict(hf_id='nvidia/MambaVision-B-1K', channels=(128, 256, 512, 1024)),
}

_ALIASES = {'t': 'tiny', 's': 'small', 'b': 'base', 'T': 'tiny', 'S': 'small', 'B': 'base'}


def resolve_variant(name: str) -> str:
    key = _ALIASES.get(name, name).lower()
    if key not in VARIANTS:
        raise ValueError(f"Unknown MambaVision variant '{name}'. "
                         f"Expected one of {sorted(VARIANTS)}.")
    return key


def _env_flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes', 'on')


@MODELS.register_module()
class MambaVisionBackbone(BaseModule):
    """MambaVision encoder exposing multi-scale features.

    Args:
        variant: ``'tiny'``, ``'small'`` or ``'base'``. Ignored if
            ``model_name`` is given.
        model_name: Explicit Hugging Face identifier or local directory of a
            MambaVision model (overrides ``variant``).
        pretrained: Load ImageNet-1K weights from the Hub/cache. Set to
            ``False`` when a U-MV checkpoint will be loaded afterwards.
        local_files_only: Never contact the network; use the cache only.
            Defaults to the value of ``HF_HUB_OFFLINE``.
        cache_dir: Optional Hugging Face cache directory.
        out_indices: Stage indices to return (0-based, strides 4/8/16/32).
        frozen: Freeze all backbone parameters (feature-extractor mode).
    """

    def __init__(self,
                 variant: str = 'small',
                 model_name: Optional[str] = None,
                 pretrained: bool = True,
                 local_files_only: Optional[bool] = None,
                 cache_dir: Optional[str] = None,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 frozen: bool = False,
                 init_cfg: Optional[dict] = None,
                 **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        if kwargs:
            print_log(f'MambaVisionBackbone: ignoring unused arguments {sorted(kwargs)}',
                      logger='current', level=30)
        self.variant = resolve_variant(variant) if model_name is None else None
        self.model_name = model_name or VARIANTS[self.variant]['hf_id']
        self.out_indices = tuple(out_indices)
        self.channels: Tuple[int, ...] = (
            tuple(VARIANTS[self.variant]['channels']) if self.variant else ())

        if local_files_only is None:
            local_files_only = _env_flag('HF_HUB_OFFLINE')
        hub_kwargs = dict(trust_remote_code=True, local_files_only=local_files_only)
        if cache_dir:
            hub_kwargs['cache_dir'] = cache_dir

        from transformers import AutoConfig, AutoModel  # deferred: heavy import

        if pretrained:
            self.backbone = AutoModel.from_pretrained(self.model_name, **hub_kwargs)
        else:
            config = AutoConfig.from_pretrained(self.model_name, **hub_kwargs)
            self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        # Hugging Face returns models in eval mode; the runner toggles modes.
        self.backbone.train()

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self._is_init = True  # weights come from the Hub or a checkpoint

    def init_weights(self) -> None:  # noqa: D401
        """No-op: parameters originate from the Hub or a U-MV checkpoint."""
        print_log('MambaVisionBackbone.init_weights: skipped (pre-trained '
                  'weights are loaded at construction time).', logger='current')

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.backbone(x)
        # Hugging Face MambaVision: (avg_pooled_features, [stage1..stage4])
        feats = out[1] if isinstance(out, (tuple, list)) else out
        if not isinstance(feats, (tuple, list)) or len(feats) < max(self.out_indices) + 1:
            raise RuntimeError(
                'Unexpected MambaVision output structure; expected a tuple '
                '(pooled, list_of_stage_features).')
        return [feats[i] for i in self.out_indices]


# --------------------------------------------------------------------------
# Backward-compatible registry names used by the original configurations.
# --------------------------------------------------------------------------
@MODELS.register_module()
class mamba_tiny_vision_timm(MambaVisionBackbone):  # noqa: N801
    def __init__(self, **kwargs):
        kwargs.pop('variant', None)
        super().__init__(variant='tiny', **kwargs)


@MODELS.register_module()
class mamba_small_vision_timm(MambaVisionBackbone):  # noqa: N801
    def __init__(self, **kwargs):
        kwargs.pop('variant', None)
        super().__init__(variant='small', **kwargs)


@MODELS.register_module()
class mamba_base_vision_timm(MambaVisionBackbone):  # noqa: N801
    def __init__(self, **kwargs):
        kwargs.pop('variant', None)
        super().__init__(variant='base', **kwargs)
