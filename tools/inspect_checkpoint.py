#!/usr/bin/env python3
"""Print what a U-MV checkpoint contains and which variant it belongs to.

    python tools/inspect_checkpoint.py Pretrained_Weights/U-MV-small_latest.pth [--keys]
"""
import argparse
import json

import _bootstrap  # noqa: F401
from umv.checkpoint import load_checkpoint_file, resolve_checkpoint, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', help='.pth file or a work directory (resolves best_mIoU_iter_*.pth)')
    p.add_argument('--keys', action='store_true', help='list all parameter names and shapes')
    p.add_argument('--config-text', action='store_true', help='print the training config stored in meta')
    a = p.parse_args()

    a.checkpoint = resolve_checkpoint(a.checkpoint)
    info = summarize(a.checkpoint)
    print(json.dumps(info, indent=2, default=str))
    if info['variant'] is None:
        print('!! Could not infer the variant: decode_head.decoder_stages.0.0.weight missing or unexpected.')
    else:
        print(f'=> use configs/mambavision/U-MV-{info["variant"]}.py with this checkpoint')
    if a.keys or a.config_text:
        sd, meta = load_checkpoint_file(a.checkpoint)
        if a.keys:
            for k, v in sd.items():
                print(f'{k:80s} {tuple(v.shape) if hasattr(v, "shape") else type(v).__name__}')
        if a.config_text:
            print(meta.get('cfg', '(no config stored in meta)'))


if __name__ == '__main__':
    main()
