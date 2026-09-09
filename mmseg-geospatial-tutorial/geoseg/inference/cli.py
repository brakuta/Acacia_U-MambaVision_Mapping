"""Shared command-line options for the inference scripts."""
from __future__ import annotations

import argparse

from .pipeline import InferenceSettings


def add_common_arguments(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument('--config', required=True, help='MMSegmentation config of the trained model')
    p.add_argument('--checkpoint', required=True,
                   help='checkpoint (.pth) or a work directory containing best_mIoU_iter_*.pth')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--scratch-dir', default=None,
                   help='Fast local directory used to stage the input and accumulators '
                        '(recommended on WSL2/Docker, e.g. /tmp/geospatial_work)')
    p.add_argument('--keep-scratch', action='store_true')
    g = p.add_argument_group('tiling')
    g.add_argument('--tile-size', type=int, default=1024)
    g.add_argument('--overlap', type=int, default=256, help='>=128 recommended')
    g.add_argument('--blend', choices=['center', 'hann'], default='center')
    g.add_argument('--batch-size', type=int, default=8)
    g.add_argument('--num-workers', type=int, default=4)
    g.add_argument('--prefetch-factor', type=int, default=2)
    g.add_argument('--mp-context', choices=['fork', 'spawn', 'forkserver'], default=None)
    g.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    g.add_argument('--bands', type=int, nargs=3, default=(1, 2, 3), metavar='B',
                   help='1-based band indices to feed as R G B (default 1 2 3)')
    g.add_argument('--band-order', choices=['rgb', 'bgr'], default='rgb',
                   help='Order of the selected bands in the file (default rgb; no swap)')
    v = p.add_argument_group('vectorisation')
    v.add_argument('--class-id', type=int, default=1, help='index of the class to export')
    v.add_argument('--thresh', type=float, default=0.35, help='probability threshold')
    v.add_argument('--min-area', type=float, default=0.0, help='minimum polygon area in CRS units')
    v.add_argument('--connectivity', type=int, choices=[4, 8], default=4)
    v.add_argument('--no-mean-prob', action='store_true', help='skip per-polygon mean probability')
    v.add_argument('--format', choices=['gpkg', 'shp'], default='gpkg')
    v.add_argument('--save-prob', action='store_true', help='keep the probability GeoTIFF next to the vectors')
    v.add_argument('--prob-dtype', choices=['uint8', 'float32'], default='uint8')
    return p


def settings_from_args(a: argparse.Namespace) -> InferenceSettings:
    return InferenceSettings(
        tile_size=a.tile_size, overlap=a.overlap, batch_size=a.batch_size,
        num_workers=a.num_workers, prefetch_factor=a.prefetch_factor, blend=a.blend,
        class_id=a.class_id, thresh=a.thresh, min_area=a.min_area,
        connectivity=a.connectivity, precision=a.precision, bands=tuple(a.bands),
        band_order=a.band_order, prob_dtype=a.prob_dtype, save_prob=a.save_prob,
        output_format=a.format, mp_context=a.mp_context, compute_mean_prob=not a.no_mean_prob)
