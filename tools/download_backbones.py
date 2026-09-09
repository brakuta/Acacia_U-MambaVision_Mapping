#!/usr/bin/env python3
"""Pre-download the MambaVision backbones (code + ImageNet-1K weights) into the
Hugging Face cache so that training/inference can run offline afterwards.

    python tools/download_backbones.py --variants tiny small base
    # later: export HF_HUB_OFFLINE=1
"""
import argparse

import _bootstrap  # noqa: F401
from umv.models.mamba_vision import VARIANTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument('--cache-dir', default=None, help='override HF cache directory')
    p.add_argument('--build', action='store_true', help='also instantiate the model (checks mamba_ssm)')
    a = p.parse_args()

    from huggingface_hub import snapshot_download
    for v in a.variants:
        repo = VARIANTS[v]['hf_id']
        path = snapshot_download(repo_id=repo, cache_dir=a.cache_dir)
        print(f'[{v}] {repo} -> {path}')
        if a.build:
            from transformers import AutoModel
            m = AutoModel.from_pretrained(repo, trust_remote_code=True, cache_dir=a.cache_dir)
            print(f'      built: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M parameters')


if __name__ == '__main__':
    main()
