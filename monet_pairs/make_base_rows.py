#!/usr/bin/env python3
"""Emit base_rows.json: Monet-SFT-125k converted to the trainer-native
interleaved conversation format.

One record per trajectory:
  {"id": "<subset>_<sample_id>",
   "images": ["<subset>/images/..jpg", ...],          # positional
   "conversations": [{"from": "human"|"gpt", "value": "...<image>\n..."}]}

Conversion rules (VERIFY against the three-row dump from Monet's loader
before training — the sidecar does not depend on these, only this file does):
  - system turns dropped,
  - each image part becomes a positional "<image>\n" placeholder,
  - "<abs_vis_token></abs_vis_token>" markers dropped (the trainer inserts
    its <latent> block at the aux <image> position per stage logic),
  - <observation> tags kept verbatim (span extractor needs them).

Stdlib only. Run:
  python3 make_base_rows.py --root ~/Desktop/Monet-SFT-125k \
      --out ~/Desktop/Monet-SFT-125k/data/base_rows.json
"""
import argparse
import json
import os

SUBSETS = ['Visual_CoT', 'ReFocus', 'CogCoM', 'Zebra_CoT_count',
           'Zebra_CoT_geometry', 'Zebra_CoT_visual_search']
ROLE = {'user': 'human', 'assistant': 'gpt'}


def convert(row, subset):
    rid = f"{subset}_{row['metadata']['sample_id']}"
    images, convs = [], []
    for turn in row['data']:
        if turn['role'] == 'system':
            continue
        value = ''
        for part in turn['content']:
            if part['type'] == 'image':
                images.append(part['image'])
                value += '<image>\n'
            else:
                value += part['text'].replace('<abs_vis_token></abs_vis_token>', '')
        convs.append({'from': ROLE[turn['role']], 'value': value.strip()})
    assert images and convs, f'empty row {rid}'
    n_ph = sum(c['value'].count('<image>') for c in convs)
    assert n_ph == len(images), f'{rid}: {n_ph} placeholders vs {len(images)} images'
    return {'id': rid, 'images': images, 'conversations': convs}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--subsets', nargs='*', default=SUBSETS)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out, ids = [], set()
    for subset in args.subsets:
        rows = json.load(open(os.path.join(args.root, subset, 'train.json')))
        for row in rows:
            rec = convert(row, subset)
            assert rec['id'] not in ids, f"duplicate id {rec['id']}"
            ids.add(rec['id'])
            out.append(rec)
        print(f'{subset}: {len(rows)} rows')
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'wrote {len(out)} base rows -> {args.out}')


if __name__ == '__main__':
    main()
