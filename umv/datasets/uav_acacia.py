"""UAV *Acacia tortilis* crown dataset.

Directory layout (``data_root``)::

    img_dir/{train,val,test,Generalizability}/*.tif
    ann_dir/{train,val,test,Generalizability}/*.tif   (or .png)

Annotations are single-band index rasters with ``0 = background`` and
``1 = acacia``.  Image and mask basenames must match within a split.
"""
from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class UAVAcaciaDataset(BaseSegDataset):
    """Binary UAV crown dataset (background / acacia)."""

    METAINFO = dict(
        classes=('background', 'acacia'),
        palette=[[0, 0, 0], [255, 0, 37]])

    def __init__(self,
                 img_suffix: str = '.tif',
                 seg_map_suffix: str = '.tif',
                 reduce_zero_label: bool = False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
