#!/usr/bin/env python3
"""Segment every GeoTIFF below a folder (recursively), mirroring the folder tree.

Example::

    python tools/batch_geospatial_inference.py \
        --config projects/<project>/configs/segformer_b2.py \
        --checkpoint work_dirs/<project>/segformer_b2 \
        --input-dir /data/orthos --output-dir /data/predictions \
        --scratch-dir /tmp/geospatial_work --skip-existing

Existing outputs are skipped with ``--skip-existing`` so that an interrupted
batch can be resumed.  A ``batch_summary.json`` is written to the output dir.
"""
import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from geoseg.inference.cli import add_common_arguments, settings_from_args
from geoseg.inference.pipeline import load_segmentor, segment_raster


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pattern', nargs='+', default=['*.tif', '*.tiff'], help='glob patterns (recursive)')
    p.add_argument('--exclude', nargs='*', default=['_mask', '_prob'],
                   help='substrings of file names to skip')
    p.add_argument('--skip-existing', action='store_true')
    p.add_argument('--continue-on-error', action='store_true')
    add_common_arguments(p)
    a = p.parse_args()

    in_root, out_root = Path(a.input_dir), Path(a.output_dir)
    files = sorted({f for pat in a.pattern for f in in_root.rglob(pat)
                    if not any(x in f.name for x in a.exclude)})
    if not files:
        raise SystemExit(f'No rasters matching {a.pattern} under {in_root}')
    print(f'-> {len(files)} rasters found under {in_root}')

    import torch
    torch.backends.cudnn.benchmark = True
    model, cfg = load_segmentor(a.config, a.checkpoint, device=a.device)
    settings = settings_from_args(a)
    ext = '.shp' if a.format == 'shp' else '.gpkg'

    summary, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        out = (out_root / f.relative_to(in_root)).with_suffix(ext)
        if a.skip_existing and out.exists():
            print(f'[{i}/{len(files)}] skip (exists): {out}')
            summary.append(dict(image=str(f), vector=str(out), skipped=True))
            continue
        print(f'[{i}/{len(files)}] {f}')
        try:
            stats = segment_raster(model, cfg, str(f), str(out), settings, device=a.device,
                                   scratch_dir=a.scratch_dir, keep_scratch=a.keep_scratch)
            summary.append(stats)
        except Exception as e:  # noqa: BLE001
            if not a.continue_on_error:
                raise
            print(f'   !! failed: {e}')
            summary.append(dict(image=str(f), error=str(e)))
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / 'batch_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'-> batch finished in {(time.time() - t0) / 60:.1f} min; summary: {out_root / "batch_summary.json"}')


if __name__ == '__main__':
    main()
