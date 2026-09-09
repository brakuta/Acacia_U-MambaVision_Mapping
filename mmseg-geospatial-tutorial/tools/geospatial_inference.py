#!/usr/bin/env python3
"""Segment one georeferenced orthomosaic and export class polygons.

Example (inside the Docker container, data mounted at /data)::

    python tools/geospatial_inference.py \
        --config projects/buildings/configs/segformer_b2.py \
        --checkpoint work_dirs/buildings/segformer_b2 \
        --input /data/orthos/block12.tif \
        --output /data/predictions/block12_buildings.gpkg \
        --scratch-dir /tmp/geospatial_work --min-area 1.0 --save-prob
"""
import argparse
import json

import _bootstrap  # noqa: F401
from geoseg.inference.cli import add_common_arguments, settings_from_args
from geoseg.inference.pipeline import load_segmentor, segment_raster


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', required=True, help='GeoTIFF (COG recommended)')
    p.add_argument('--output', required=True, help='output vector path (.gpkg or .shp)')
    add_common_arguments(p)
    a = p.parse_args()

    import torch
    torch.backends.cudnn.benchmark = True
    model, cfg = load_segmentor(a.config, a.checkpoint, device=a.device)
    stats = segment_raster(model, cfg, a.input, a.output, settings_from_args(a),
                           device=a.device, scratch_dir=a.scratch_dir, keep_scratch=a.keep_scratch)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
