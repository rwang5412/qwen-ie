#!/usr/bin/env python3
"""Render before/after sheets for step3 pilot output.

Each sheet: top row = aux | aux' (the edited crop), bottom row = original |
original' (bbox outlined so you can see the edit region), caption = id, kind,
edit prompt, obs -> obs', roi_delta.

Run on the Mac (needs PIL — use the viz venv):
  ~/Desktop/Monet-SFT-125k/_viz_work/venv/bin/python make_compare_sheets.py \
      --pilot ~/Desktop/monet_pilot/pilot8.jsonl \
      --edited-dir ~/Desktop/monet_pilot/lightning8 \
      --orig-root ~/Desktop/Monet-SFT-125k/counterfactual/images \
      --out ~/Desktop/monet_pilot/compare8
"""
import argparse
import json
import os

from PIL import Image, ImageDraw

GAP, CAP_H = 10, 78


def fit_h(im, h):
    return im.resize((max(1, round(im.width * h / im.height)), h))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', required=True, help='pilot .jsonl from step3')
    ap.add_argument('--edited-dir', required=True, help='pulled edited_pilot dir')
    ap.add_argument('--orig-root', required=True, help='local counterfactual/images')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    made = 0
    for line in open(args.pilot):
        r = json.loads(line)
        rid = r['id']
        p_aux2 = os.path.join(args.edited_dir, f'{rid}_aux.png')
        p_orig2 = os.path.join(args.edited_dir, f'{rid}_orig.png')
        if not (os.path.exists(p_aux2) and os.path.exists(p_orig2)):
            continue
        aux = Image.open(os.path.join(args.orig_root, r['aux_image'])).convert('RGB')
        orig = Image.open(os.path.join(args.orig_root, r['orig_image'])).convert('RGB')
        aux2 = Image.open(p_aux2).convert('RGB')
        orig2 = Image.open(p_orig2).convert('RGB')
        # left column stays pristine (the unedited pair exactly as shipped);
        # the red edit-region outline goes only on the edited original.
        ImageDraw.Draw(orig2).rectangle(r['bbox'], outline=(255, 0, 0),
                                        width=max(2, orig.width // 250))

        ht = min(320, aux.height)
        hb = min(480, orig.height)
        top = [fit_h(aux, ht), fit_h(aux2, ht)]
        bot = [fit_h(orig, hb), fit_h(orig2, hb)]
        W = max(sum(i.width for i in row) + GAP for row in (top, bot))
        sheet = Image.new('RGB', (W, ht + hb + 2 * GAP + CAP_H), 'white')
        x = 0
        for im in top:
            sheet.paste(im, (x, 0))
            x += im.width + GAP
        x = 0
        for im in bot:
            sheet.paste(im, (x, ht + GAP))
            x += im.width + GAP
        cap = (f"[{rid}] kind={r['kind']}  roi_delta={r.get('roi_delta')}\n"
               f"PROMPT: {r['edit_prompt'][:110]}\n"
               f"OBS: {r['obs'][:55]}  ->  {r['obs_new'][:55]}")
        d = ImageDraw.Draw(sheet)
        d.rectangle([0, ht + hb + 2 * GAP, W, ht + hb + 2 * GAP + CAP_H], fill='black')
        d.text((6, ht + hb + 2 * GAP + 4), cap, fill='white')
        sheet.save(os.path.join(args.out, f'{rid}.png'))
        made += 1
        print(f'{rid}: {r["edit_prompt"][:80]}')
    print(f'\n{made} sheets -> {args.out}')


if __name__ == '__main__':
    main()
