"""Loading of legacy U-MV configurations.

The configurations stored in the original MMSegmentation work directories
(``mambavision-*_generic-unet_acacia.py``) reference modules of the former
copy-in layout (``mmseg.custom_models.*``) and the ``ADE20KDataset`` class
that had been patched in place to read two-class ``.tif`` masks.  Neither
exists in a clean installation.  :func:`load_config` loads such files by
rewriting these references to their equivalents in :mod:`umv`, so that the
archived configs remain usable for evaluation, inference and resumed
training without manual editing.
"""
from __future__ import annotations

from typing import Any, Dict

from mmengine.config import Config
from mmengine.logging import print_log

LEGACY_MODULE_PREFIX = 'mmseg.custom_models'
LEGACY_DATASET = 'ADE20KDataset'
NEW_DATASET = 'UAVAcaciaDataset'


def _is_legacy(cfg: Config) -> bool:
    ci = cfg.get('custom_imports')
    if not ci:
        return False
    imports = ci.get('imports', []) if isinstance(ci, dict) else []
    if isinstance(imports, str):
        imports = [imports]
    return any(str(m).startswith(LEGACY_MODULE_PREFIX) for m in imports)


def _rewrite_datasets(node: Any) -> int:
    """Replace legacy dataset type names recursively; returns count."""
    n = 0
    if isinstance(node, dict):
        if node.get('type') == LEGACY_DATASET:
            node['type'] = NEW_DATASET
            n += 1
        for v in node.values():
            n += _rewrite_datasets(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            n += _rewrite_datasets(v)
    return n


def load_config(path: str, **kwargs) -> Config:
    """``Config.fromfile`` with transparent support for legacy U-MV configs."""
    cfg = Config.fromfile(path, import_custom_modules=False, **kwargs)
    if _is_legacy(cfg):
        n = 0
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            if key in cfg:
                n += _rewrite_datasets(cfg[key])
        if cfg.get('dataset_type') == LEGACY_DATASET:
            cfg.dataset_type = NEW_DATASET
        cfg.custom_imports = dict(imports=['umv'], allow_failed_imports=False)
        print_log(f'Legacy U-MV config detected ({path}): custom_imports rewritten to '
                  f"'umv' and {n} dataset entries mapped {LEGACY_DATASET} -> {NEW_DATASET}.",
                  logger='current')
    ci: Dict = cfg.get('custom_imports') or {}
    if ci.get('imports'):
        from mmengine.utils import import_modules_from_strings
        import_modules_from_strings(**ci)
    return cfg
