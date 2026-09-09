#!/usr/bin/env python3
"""Summarise MMSegmentation runs into one table (best and last validation metrics).

    python tools/compare_runs.py work_dirs/buildings            # all runs below a folder
    python tools/compare_runs.py work_dirs/a work_dirs/b --out summary.md --csv summary.csv

A run is any directory containing MMSegmentation's ``<timestamp>/vis_data/scalars.json``.
For each run the table reports the iteration and value of the best mIoU, the
last validation metrics, the number of parameters if a checkpoint exists, and the
test-split metrics if ``tools/test.py`` wrote a JSON under ``<run>/test``.
"""
import argparse
import glob
import json
import os
import sys

import _bootstrap  # noqa: F401

METRICS = ('mIoU', 'mFscore', 'mAcc', 'aAcc')


def _read_scalars(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def summarise_run(run_dir):
    scalar_files = sorted(glob.glob(os.path.join(run_dir, '*', 'vis_data', 'scalars.json')))
    if not scalar_files:
        return None
    rows = [r for f in scalar_files for r in _read_scalars(f)]
    val = [r for r in rows if 'mIoU' in r]
    out = dict(run=os.path.relpath(run_dir), n_val=len(val))
    if val:
        best = max(val, key=lambda r: r['mIoU'])
        last = val[-1]
        out.update(best_iter=best.get('step'), best_mIoU=best['mIoU'], best_mFscore=best.get('mFscore'),
                   last_iter=last.get('step'), last_mIoU=last['mIoU'], last_mFscore=last.get('mFscore'))
    train = [r for r in rows if 'loss' in r]
    if train:
        out['last_loss'] = train[-1]['loss']
    test_files = sorted(glob.glob(os.path.join(run_dir, 'test', '*', '*.json')))
    for tf in test_files[::-1]:
        try:
            d = json.load(open(tf))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(d, dict) and 'mIoU' in d:
            out.update(test_mIoU=d['mIoU'], test_mFscore=d.get('mFscore'))
            break
    return out


def find_runs(roots):
    runs = []
    for root in roots:
        for f in glob.glob(os.path.join(root, '**', 'vis_data', 'scalars.json'), recursive=True):
            runs.append(os.path.dirname(os.path.dirname(os.path.dirname(f))))
    return sorted(set(runs))


def to_markdown(rows):
    cols = ['run', 'best_iter', 'best_mIoU', 'best_mFscore', 'last_iter', 'last_mIoU', 'test_mIoU', 'test_mFscore', 'last_loss']
    fmt = lambda v: '' if v is None else (f'{v:.2f}' if isinstance(v, float) else str(v))  # noqa: E731
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(fmt(r.get(c)) for c in cols) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('roots', nargs='+', help='work directories or their parents')
    p.add_argument('--out', help='write a Markdown table to this file')
    p.add_argument('--csv', help='write a CSV to this file')
    p.add_argument('--sort', default='best_mIoU')
    a = p.parse_args()
    rows = [s for s in (summarise_run(r) for r in find_runs(a.roots)) if s]
    if not rows:
        sys.exit('no runs with vis_data/scalars.json found')
    rows.sort(key=lambda r: r.get(a.sort) or -1, reverse=True)
    md = to_markdown(rows)
    print(md)
    if a.out:
        open(a.out, 'w').write(md + '\n')
    if a.csv:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
