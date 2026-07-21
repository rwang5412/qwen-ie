#!/usr/bin/env python3
"""Build a stratified pilot manifest: N seeded-random rows per
(subset, kind) group from pairs_bbox.jsonl, so one small run exercises every
edit type on every subset before committing the full arrays.

Stdlib only. Run:
  python3 make_pilot_sample.py \
      --manifest ~/Desktop/Monet-SFT-125k/counterfactual/pairs_bbox.jsonl \
      --out ~/Desktop/Monet-SFT-125k/counterfactual/pilot_sample.jsonl --per-group 6
"""
import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--per-group', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    groups = defaultdict(list)
    for line in open(args.manifest):
        r = json.loads(line)
        groups[(r['subset'], r['kind'])].append(r)

    rng = random.Random(args.seed)
    picked = []
    for key in sorted(groups):
        rows = groups[key]
        sel = rng.sample(rows, min(args.per_group, len(rows)))
        picked.extend(sel)
        print(f'{key[0]:12s} {key[1]:9s} pool={len(rows):6d} sampled={len(sel)}')
    with open(args.out, 'w') as f:
        for r in picked:
            f.write(json.dumps(r) + '\n')
    print(f'wrote {len(picked)} rows -> {args.out}')


if __name__ == '__main__':
    main()
