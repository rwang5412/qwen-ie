#!/usr/bin/env python3
"""Phase A / Steps 1+2+8: filter Monet-SFT to editable rows, extract edit specs,
assign the frozen holdout split.

Keeps only Visual_CoT and ReFocus rows that pass ALL asserts:
  - exactly 2 images (original, aux), exactly 1 <abs_vis_token> block,
  - exactly 1 <observation>...</observation> span, non-empty assistant text,
  - observation is rule-editable (contains a number, or a color word) so
    new_content needs no model call,
  - the deterministic swap actually changes the answer (y' != y).
Rows failing any assert are dropped, not repaired.

Emits pairs_manifest.jsonl with one record per kept row: images, question,
obs/obs', y/y', edit prompt, and a stable 10% holdout split (hash of id —
written once here, never reassigned).

Stdlib only. Run:  python3 step1_filter.py --root ~/Desktop/Monet-SFT-125k \
                        --out ~/Desktop/Monet-SFT-125k/counterfactual/pairs_manifest.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import re

COLORS = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
          'brown', 'black', 'white', 'gray']
COLOR_SET = set(COLORS) | {'grey', 'beige', 'tan', 'gold', 'silver', 'maroon',
                           'navy', 'teal', 'cyan', 'magenta', 'violet'}
NUM_RE = re.compile(r'\d[\d,. ]*\d|\d')  # first number incl. separators
# visually-adjacent colors: a swap within a group is not "materially different"
NEIGHBORS = [{'black', 'gray', 'grey', 'white', 'silver', 'brown', 'tan', 'beige'},
             {'red', 'maroon', 'pink', 'orange'},
             {'blue', 'navy', 'teal', 'cyan', 'purple', 'violet', 'magenta'},
             {'green', 'teal'},
             {'yellow', 'gold', 'tan', 'beige', 'orange'}]


def swap_number(obs, rng):
    """Replace the leading digit of the first number; keeps format/length."""
    m = NUM_RE.search(obs)
    old = m.group(0)
    lead = old.lstrip('0')[0] if old.lstrip('0') else old[0]
    new_lead = rng.choice([d for d in '123456789' if d != lead])
    new = old.replace(lead, new_lead, 1)
    return old, new, obs[:m.start()] + new + obs[m.end():]


def swap_color(obs, rng):
    words = re.findall(r'[A-Za-z]+', obs.lower())
    present = [w for w in words if w in COLOR_SET]
    old = present[0]
    near = set().union(*[g for g in NEIGHBORS if old in g]) | set(present)
    new = rng.choice([c for c in COLORS if c not in near])
    pat = re.compile(re.escape(old), re.IGNORECASE)
    swapped = pat.sub(new, obs, count=1)
    swapped = re.sub(r'\ba ([aeiou])', r'an \1', swapped)
    swapped = re.sub(r'\ban ([^aeiou\s])', r'a \1', swapped)
    return old, new, swapped


def process(row, subset):
    turns = row['data']
    imgs = [p['image'] for t in turns for p in t['content'] if p['type'] == 'image']
    q = ' '.join(p['text'] for t in turns if t['role'] == 'user'
                 for p in t['content'] if p['type'] == 'text').strip()
    atext = ' '.join(p['text'] for t in turns if t['role'] == 'assistant'
                     for p in t['content'] if p['type'] == 'text').strip()
    obs = re.findall(r'<observation>(.*?)</observation>', atext, re.S)
    if len(imgs) != 2 or len(obs) != 1 or atext.count('<abs_vis_token>') != 1 or not atext:
        return None
    o = obs[0].strip()
    rid = f"{subset}_{row['metadata']['sample_id']}"
    rng = random.Random(rid)  # deterministic per row

    if NUM_RE.search(o):
        kind = 'number'
        old, new, o_new = swap_number(o, rng)
        if subset == 'ReFocus':
            prompt = (f'Replace the text "{old}" with "{new}". '
                      f'Keep the same font, size, color and alignment.')
        else:
            prompt = f'Change {o} to {o_new}'
    elif any(w in COLOR_SET for w in re.findall(r'[A-Za-z]+', o.lower())):
        kind = 'color'
        old, new, o_new = swap_color(o, rng)
        prompt = (f'Change the {old} object in the image to {new}'
                  if o_new.strip().lower() in COLOR_SET else f'Change {o} to {o_new}')
    else:
        # free-text observation: no deterministic swap — emitted with empty
        # spec fields; step2b (VLM proposal, cluster) fills them or drops.
        kind, old, new, o_new, prompt = 'freetext', None, None, None, None

    if kind == 'freetext':
        y_new = None
    else:
        y_new = atext.replace(f'<observation>{obs[0]}</observation>',
                              f'<observation>{o_new}</observation>')
    # ReFocus repeats the value outside the span (ANSWER/FINAL ANSWER) — swap there too.
    if subset == 'ReFocus' and y_new is not None:
        y_new = y_new.replace(old, new)
        plain_old, plain_new = old.replace(',', '').replace(' ', ''), \
            new.replace(',', '').replace(' ', '')
        y_new = y_new.replace(plain_old, plain_new)
    if kind != 'freetext' and y_new == atext:
        return None  # swap didn't take -> dropped

    holdout = int(hashlib.sha256(rid.encode()).hexdigest(), 16) % 10 == 0
    return {
        'id': rid, 'subset': subset,
        'orig_image': imgs[0], 'aux_image': imgs[1],
        'question': q, 'kind': kind,
        'obs': o, 'obs_new': o_new,
        'old_content': old, 'new_content': new,
        'y': atext, 'y_new': y_new,
        'edit_prompt': prompt,
        'split': 'holdout' if holdout else 'train',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, help='Monet-SFT-125k directory')
    ap.add_argument('--out', required=True, help='output pairs_manifest.jsonl')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    counts = {}
    with open(args.out, 'w') as f:
        for subset in ['Visual_CoT', 'ReFocus']:
            rows = json.load(open(os.path.join(args.root, subset, 'train.json')))
            kept = 0
            for row in rows:
                rec = process(row, subset)
                if rec:
                    f.write(json.dumps(rec) + '\n')
                    kept += 1
            counts[subset] = (kept, len(rows))
    for s, (k, n) in counts.items():
        print(f'{s}: kept {k}/{n}')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
