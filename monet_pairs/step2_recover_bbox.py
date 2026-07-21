#!/usr/bin/env python3
"""Phase A / bbox recovery: locate each pair's edit region, since Monet-SFT
stores no coordinates.

  - Visual_CoT: aux is an unscaled crop of the original -> FFT normalized
    cross-correlation finds its exact location. Accept iff NCC >= --min-ncc
    and pixel MAE at the match <= --max-mae; otherwise the row is dropped.
  - ReFocus: aux is the full original plus a drawn highlight rectangle ->
    the bbox is the bounding rect of |aux - orig| above threshold. Accept iff
    the diff region is a plausible box (non-degenerate, <30% of image area).

Reads images straight from each subset's images.zip. Writes the input records
augmented with bbox=[x1,y1,x2,y2] (pixels, original's frame) + match scores to
--out, and (optionally) exports every kept pair's images to --export-dir for
transfer to the cluster.

Run (viz venv has numpy+PIL):
  ~/Desktop/Monet-SFT-125k/_viz_work/venv/bin/python step2_recover_bbox.py \
      --root ~/Desktop/Monet-SFT-125k \
      --manifest ~/Desktop/Monet-SFT-125k/counterfactual/pairs_manifest.jsonl \
      --out ~/Desktop/Monet-SFT-125k/counterfactual/pairs_bbox.jsonl \
      --export-dir ~/Desktop/Monet-SFT-125k/counterfactual/images
"""
import argparse
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image


def find_crop(orig, aux):
    """(x, y, ncc) of aux's top-left inside orig via FFT NCC on grayscale."""
    O = np.asarray(orig.convert('L'), dtype=np.float32)
    A = np.asarray(aux.convert('L'), dtype=np.float32)
    oh, ow = O.shape
    ah, aw = A.shape
    if ah > oh or aw > ow:
        return None
    A0 = A - A.mean()
    F = np.fft.rfft2(O, s=(oh + ah - 1, ow + aw - 1))
    G = np.fft.rfft2(A0[::-1, ::-1], s=(oh + ah - 1, ow + aw - 1))
    corr = np.fft.irfft2(F * G, s=(oh + ah - 1, ow + aw - 1))[ah - 1:oh, aw - 1:ow]
    ii = np.cumsum(np.cumsum(np.pad(O, ((1, 0), (1, 0))), 0), 1)
    ii2 = np.cumsum(np.cumsum(np.pad(O ** 2, ((1, 0), (1, 0))), 0), 1)

    def wsum(I):
        return I[ah:, aw:] - I[:-ah, aw:] - I[ah:, :-aw] + I[:-ah, :-aw]

    var = wsum(ii2) - wsum(ii) ** 2 / (ah * aw)
    ncc = corr / np.sqrt(np.maximum(var, 1e-6) * (A0 ** 2).sum())
    y, x = np.unravel_index(np.argmax(ncc), ncc.shape)
    return int(x), int(y), float(ncc[y, x])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--export-dir', default=None)
    ap.add_argument('--min-ncc', type=float, default=0.95)
    ap.add_argument('--max-mae', type=float, default=15.0)
    ap.add_argument('--diff-thresh', type=int, default=40)
    ap.add_argument('--limit', type=int, default=0, help='first N rows only (0=all)')
    args = ap.parse_args()

    zips = {s: zipfile.ZipFile(os.path.join(args.root, s, 'images.zip'))
            for s in ['Visual_CoT', 'ReFocus']}
    names = {s: set(z.namelist()) for s, z in zips.items()}

    def load(subset, relpath):
        base = os.path.basename(relpath)  # zips store flat basenames
        member = base if base in names[subset] else relpath
        return Image.open(io.BytesIO(zips[subset].read(member))).convert('RGB')

    if args.export_dir:
        os.makedirs(args.export_dir, exist_ok=True)

    n_in = n_ok = n_rej = n_err = 0
    with open(args.out, 'w') as out_f:
        for line in open(args.manifest):
            if args.limit and n_in >= args.limit:
                break
            n_in += 1
            r = json.loads(line)
            try:
                orig = load(r['subset'], r['orig_image'])
                aux = load(r['subset'], r['aux_image'])
                if r['subset'] == 'Visual_CoT':
                    x, y, ncc = find_crop(orig, aux)
                    aw, ah = aux.size
                    O = np.asarray(orig, dtype=np.int16)
                    A = np.asarray(aux, dtype=np.int16)
                    mae = float(np.abs(O[y:y + ah, x:x + aw] - A).mean())
                    r['bbox'], r['ncc'], r['mae'] = [x, y, x + aw, y + ah], round(ncc, 4), round(mae, 2)
                    if ncc < args.min_ncc or mae > args.max_mae:
                        n_rej += 1
                        continue
                else:  # ReFocus: diff the drawn overlay
                    if orig.size != aux.size:
                        n_rej += 1
                        continue
                    d = np.abs(np.asarray(orig, np.int16) - np.asarray(aux, np.int16)).sum(2)
                    ys, xs = np.where(d > args.diff_thresh)
                    if len(ys) < 50:
                        n_rej += 1
                        continue
                    x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                    area_frac = (x2 - x1) * (y2 - y1) / (orig.size[0] * orig.size[1])
                    if x2 - x1 < 8 or y2 - y1 < 8 or area_frac > 0.6:
                        n_rej += 1
                        continue
                    r['bbox'] = [x1, y1, x2 + 1, y2 + 1]
                    r['diff_frac'] = round(float(len(ys)) / d.size, 4)
                out_f.write(json.dumps(r) + '\n')
                n_ok += 1
                if args.export_dir:
                    for key in ['orig_image', 'aux_image']:
                        dst = os.path.join(args.export_dir, r[key])
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if not os.path.exists(dst):
                            (orig if key == 'orig_image' else aux).save(dst, quality=95)
            except Exception as e:
                n_err += 1
                print(f"ERR {r['id']}: {type(e).__name__} {e}", flush=True)
            if n_in % 500 == 0:
                print(f'{n_in} scanned, {n_ok} ok, {n_rej} rejected', flush=True)

    print(f'done: scanned={n_in} ok={n_ok} rejected={n_rej} errors={n_err}')
    print('wrote', args.out)


if __name__ == '__main__':
    main()
