"""Lightweight U-Net-style decode head for MMSegmentation.

The decoder consumes the four encoder stages (strides 4/8/16/32), starts from
the deepest map and, at every stage, bilinearly upsamples the running
representation by a factor of two, concatenates the next shallower skip
feature and applies two 3x3 convolution-BatchNorm-ReLU blocks.  A 1x1
convolution produces the class logits at stride 4; MMSegmentation resizes
them to the input resolution.

Checkpoint compatibility
------------------------
Attribute names (``decoder_stages``, ``upsample``, ``segmentation_head``) and
the layer ordering inside each stage are those of the released U-MV
checkpoints and must not be changed.  ``conv_seg`` and ``dropout`` are
created by :class:`BaseDecodeHead` and are intentionally unused; they are
kept so that checkpoints load without unexpected-key warnings.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS


@MODELS.register_module()
class GenericUNetHead(BaseDecodeHead):
    """U-Net decoder whose stage widths follow ``decoder_channels``.

    Args:
        encoder_channels: Channel widths of the encoder stages, shallow to
            deep, e.g. ``[96, 192, 384, 768]`` for MambaVision-S.
        decoder_channels: Output widths of the successive decoder stages,
            deep to shallow.  The number of stages equals
            ``len(decoder_channels)``; the last stage has no skip input when
            ``len(decoder_channels) == len(encoder_channels)``.
        channels: Passed to :class:`BaseDecodeHead` (unused by the forward
            pass); defaults to ``decoder_channels[0]``.
        in_index: Indices of the backbone outputs to use.
    """

    def __init__(self,
                 encoder_channels: Sequence[int],
                 decoder_channels: Sequence[int] = (256, 128, 64, 32),
                 channels: int | None = None,
                 input_transform: str = 'multiple_select',
                 in_index: Sequence[int] | None = None,
                 **kwargs) -> None:
        encoder_channels = list(encoder_channels)
        if channels is None:
            channels = decoder_channels[0]
        if in_index is None:
            in_index = list(range(len(encoder_channels)))
        super().__init__(
            in_channels=encoder_channels,
            channels=channels,
            num_classes=kwargs.pop('num_classes'),
            input_transform=input_transform,
            in_index=list(in_index),
            **kwargs)
        self.encoder_channels = encoder_channels
        self.decoder_channels = tuple(decoder_channels)

        self.decoder_stages = nn.ModuleList()
        prev_ch = encoder_channels[-1]
        for i, out_ch in enumerate(self.decoder_channels):
            skip_ch = encoder_channels[-(i + 2)] if (i + 1) < len(encoder_channels) else 0
            self.decoder_stages.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch + skip_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ))
            prev_ch = out_ch

        # align_corners=True reproduces the released checkpoints exactly.
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.segmentation_head = nn.Conv2d(self.decoder_channels[-1], self.out_channels, kernel_size=1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        feats = self._transform_inputs(inputs)  # honours in_index
        x = list(feats)[::-1]                    # deep -> shallow
        out = x[0]
        for idx, stage in enumerate(self.decoder_stages):
            skip = x[idx + 1] if idx + 1 < len(x) else None
            out = self.upsample(out)
            if skip is not None:
                if out.shape[-2:] != skip.shape[-2:]:  # odd spatial sizes
                    out = F.interpolate(out, size=skip.shape[-2:], mode='bilinear', align_corners=True)
                out = torch.cat([out, skip], dim=1)
            out = stage(out)
        return self.segmentation_head(out)
