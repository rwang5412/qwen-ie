#!/usr/bin/env python3
"""Emit the final counterfactual sidecar: data/cf/pairs.jsonl + cf/images/ +
splits/eval_flip_ids.txt, in the frozen training-time schema.

Runs LAST, after step5 (verified.jsonl) and step7 (Z' cache). Ships only
verifier_pass==true rows, but records the flag so rejection is auditable.
Asserts, per the contract:
  - every row_id resolves 1:1 into base_rows.json,
  - z_prime_path exists and holds a [K, H] fp16 tensor,
  - bbox_norm is in [0,1] (normalized here, once — pixel bboxes die here),
  - split values map {train -> train, holdout -> eval_flip} and are final.

y / y_prime are answer strings: the \\boxed{...} or FINAL ANSWER value where
one exists (ReFocus), else the observation content (Visual_CoT).

Run:
  python3 emit_sidecar.py --verified verified.jsonl --zdir zprime_cache \
      --base data/base_rows.json --data-dir data
"""
import argparse
import json
import os
import re
import shutil

from PIL import Image


def extract_answer(text, obs_content):
    m = re.search(r'\\boxed\{(.*?)\}', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'FINAL ANSWER:\s*(.+?)\s*$', text.strip(), re.I)
    if m:
        return m.group(1).strip().rstrip('.')
    return obs_content


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verified', required=True, help='step5 out-manifest')
    ap.add_argument('--zdir', required=True, help='step7 Z-prime cache dir')
    ap.add_argument('--base', required=True, help='base_rows.json (for 1:1 assert)')
    ap.add_argument('--data-dir', required=True, help='output data/ root')
    args = ap.parse_args()

    base_ids = {r['id'] for r in json.load(open(args.base))}
    cf_img = os.path.join(args.data_dir, 'cf', 'images')
    cf_z = os.path.join(args.data_dir, 'cf', 'z')
    splits_dir = os.path.join(args.data_dir, 'splits')
    for d in (cf_img, cf_z, splits_dir):
        os.makedirs(d, exist_ok=True)

    import torch
    n = n_skip = 0
    eval_ids = []
    pairs_path = os.path.join(args.data_dir, 'cf', 'pairs.jsonl')
    with open(pairs_path, 'w') as out_f:
        for line in open(args.verified):
            r = json.loads(line)
            if not r.get('verified'):
                n_skip += 1
                continue
            rid = r['id']
            assert rid in base_ids, f'{rid} not in base_rows.json'

            zsrc = os.path.join(args.zdir, f'{rid}.pt')
            assert os.path.exists(zsrc), f'missing Z-prime cache {zsrc}'
            z = torch.load(zsrc, map_location='cpu')['z_prime']
            assert z.dim() == 2, f'{rid}: Z-prime must be [K, H], got {tuple(z.shape)}'
            zdst = os.path.join(cf_z, f'{rid}.pt')
            torch.save(z.half(), zdst)

            aux_dst = os.path.join(cf_img, f'{rid}_aux.png')
            full_dst = os.path.join(cf_img, f'{rid}_full.png')
            shutil.copyfile(r['aux_new'], aux_dst)
            shutil.copyfile(r['orig_new'], full_dst)

            W, H = Image.open(r['orig_new']).size
            x1, y1, x2, y2 = r['bbox']
            bbox_norm = [round(x1 / W, 4), round(y1 / H, 4),
                         round(x2 / W, 4), round(y2 / H, 4)]
            assert all(0.0 <= v <= 1.0 for v in bbox_norm), f'{rid}: bad bbox_norm'

            split = {'train': 'train', 'holdout': 'eval_flip'}[r['split']]
            if split == 'eval_flip':
                eval_ids.append(rid)

            rel = lambda p: os.path.relpath(p, args.data_dir)
            out_f.write(json.dumps({
                'row_id': rid,
                'aux_prime_path': rel(aux_dst),
                'orig_prime_path': rel(full_dst),
                'obs': r['obs'],
                'obs_prime': r['obs_new'],
                'y': extract_answer(r['y'], r['obs']),
                'y_prime': extract_answer(r['y_new'], r['obs_new']),
                'z_prime_path': rel(zdst),
                'bbox_norm': bbox_norm,
                'verifier_pass': True,
                'split': split,
            }) + '\n')
            n += 1

    with open(os.path.join(splits_dir, 'eval_flip_ids.txt'), 'w') as f:
        f.write('\n'.join(eval_ids) + '\n')
    print(f'shipped {n} pairs ({len(eval_ids)} eval_flip), '
          f'{n_skip} verifier-rejected rows excluded')
    print('wrote', pairs_path)


if __name__ == '__main__':
    main()
