
"""Custom MambaVision Backbone Wrappers for MMSegmentation."""

from typing import List
import torch.nn as nn
import torch
from transformers import AutoModel
from mmseg.registry import MODELS


class mamba_vision_tim(nn.Module):
    """
    Base class that loads a specified MambaVision model from Hugging Face.
    This class is not registered and should be inherited by a model-specific class.
    """
    def __init__(self, model_name: str):
        super().__init__()
        
        # Load the MambaVision model specified by the subclass
        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Explicitly set the backbone to training mode
        self.backbone.train()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass returns the list of feature maps from the backbone."""
        features = self.backbone(x)
        # The Hugging Face model output contains feature maps at index 1
        return features[1]


@MODELS.register_module()
class mamba_tiny_vision_timm(mamba_vision_tim):
    """Registered MMSegmentation backbone that loads MambaVision-Tiny."""
    def __init__(self, **kwargs):
        # Pass the correct model name to the base class
        super().__init__(model_name="nvidia/MambaVision-T-1K")

@MODELS.register_module()
class mamba_small_vision_timm(mamba_vision_tim):
    """Registered MMSegmentation backbone that loads MambaVision-Small."""
    def __init__(self, **kwargs):
        # Pass the correct model name to the base class
        super().__init__(model_name="nvidia/MambaVision-S-1K")

@MODELS.register_module()
class mamba_base_vision_timm(mamba_vision_tim):
    """Registered MMSegmentation backbone that loads MambaVision-Base."""
    def __init__(self, **kwargs):
        # Pass the correct model name to the base class
        super().__init__(model_name="nvidia/MambaVision-B-1K")