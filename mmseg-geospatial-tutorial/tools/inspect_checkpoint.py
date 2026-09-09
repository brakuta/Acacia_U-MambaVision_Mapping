#!/usr/bin/env python3
"""Print what an MMSegmentation checkpoint contains (or resolve a work directory to its best checkpoint).

    python tools/inspect_checkpoint.py work_dirs/buildings/segformer_b2 [--keys]
"""
import argparse
import json

import _bootstrap  # noqa: F401
from geoseg.checkpoint import load_checkpoint_file, resolve_checkpoint, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', help='.pth file or a work directory')
    p.add_argument('--keys', action='store_true', help='list parameter names and shapes')
    p.add_argument('--config-text', action='store_true', help='print the training config stored in meta')
    a = p.parse_args()
    a.checkpoint = resolve_checkpoint(a.checkpoint)
    print(json.dumps(summarize(a.checkpoint), indent=2, default=str))
    if a.keys or a.config_text:
        sd, meta = load_checkpoint_file(a.checkpoint)
        if a.keys:
            for k, v in sd.items():
                print(f'{k:80s} {tuple(v.shape) if hasattr(v, "shape") else type(v).__name__}')
        if a.config_text:
            print(meta.get('cfg', '(no config stored in meta)'))


if __name__ == '__main__':
    main()
