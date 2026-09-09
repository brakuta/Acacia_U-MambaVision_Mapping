"""U-MV: U-shaped MambaVision segmentation framework for *Acacia tortilis*
crown mapping from ultra-high-resolution UAV imagery.

Importing this package registers the custom backbone, decode head and
dataset with the MMSegmentation registries.  Configuration files reference
it through::

    custom_imports = dict(imports=['umv'], allow_failed_imports=False)
"""
from .version import __version__  # noqa: F401
from . import datasets, models  # noqa: F401  (registration side effects)

__all__ = ['__version__', 'models', 'datasets']
