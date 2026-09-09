"""geoseg: small helpers for geospatial semantic segmentation with MMSegmentation.

Importing the package registers ``LoadRasterioImage`` (multi-band raster
loader) with MMSegmentation's transform registry; configs activate it with
``custom_imports = dict(imports=['geoseg'], allow_failed_imports=False)``.
"""
from .transforms import LoadRasterioImage  # noqa: F401  (registration)

__version__ = '1.0.0'
__all__ = ['LoadRasterioImage', '__version__']
