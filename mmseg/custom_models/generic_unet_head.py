import torch
import torch.nn as nn
from mmseg.registry import MODELS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

@MODELS.register_module()
class GenericUNetHead(BaseDecodeHead):
    """
    U-Net decoder head that dynamically connects previous decoder output
    to the next skip feature, so channel dims always match.
    """
    def __init__(self,
                 encoder_channels,               # e.g. [256,512,1024,2048]
                 decoder_channels=(256, 128, 64, 32),
                 channels=None,
                 input_transform='multiple_select',
                 in_index=None,
                 **kwargs):
        # Default channels for the final seg head
        if channels is None:
            channels = decoder_channels[0]
        # Default skip indices if not provided
        if in_index is None:
            in_index = list(range(len(encoder_channels)))
        super().__init__(
            in_channels=encoder_channels,
            channels=channels,
            num_classes=kwargs.pop('num_classes'),
            input_transform=input_transform,
            in_index=in_index,
            **kwargs)

        # Build decoder stages *dynamically*
        self.decoder_stages = nn.ModuleList()
        prev_ch = encoder_channels[-1]  # start from last backbone output
        
        # For each decoder stage, pull next skip from encoder_channels
        for i, out_ch in enumerate(decoder_channels):
            # skip index from the end: encoder_channels[-(i+2)]
            skip_ch = encoder_channels[-(i+2)] if (i+1) < len(encoder_channels) else 0
            in_ch = prev_ch + skip_ch
            self.decoder_stages.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            prev_ch = out_ch

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=True)
        self.segmentation_head = nn.Conv2d(decoder_channels[-1],
                                           self.out_channels,
                                           kernel_size=1)

    def forward(self, inputs):
        # inputs: list of backbone feature maps [c1,c2,c3,c4]
        x = inputs[::-1]   # [c4, c3, c2, c1]
        out = x[0]
        for idx, stage in enumerate(self.decoder_stages):
            skip = x[idx+1] if idx+1 < len(x) else None
            out = self.upsample(out)
            if skip is not None:
                out = torch.cat([out, skip], dim=1)
            out = stage(out)
        return self.segmentation_head(out)
