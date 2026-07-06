#!/usr/bin/env python
"""Generate bbox-localized edited images for complementary VQA pairs.

One responsibility: read an edit manifest and, for each row, produce an image
where only the bbox region (plus a small context margin) is changed per the
`edit` instruction — everything outside that padded crop stays pixel-identical.
Writes the edited PNGs at deterministic id-based paths and appends an augmented
manifest (JSONL) linking each source row to its edited image.

Shardable for sbatch job arrays via --start/--end; resumable via skip-existing.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"


def example_id(row):
    # question_id restarts per dataset, so the dataset+split prefix makes it unique.
    return f"{row['dataset']}_{row['split']}_{row['question_id']}"


def denorm_bbox(norm, w, h):
    """Normalized [0,1] xyxy -> pixel xyxy, matching lvr_sft_dataset.py exactly."""
    x1, y1, x2, y2 = norm
    px = [int(round(x1 * w)), int(round(y1 * h)),
          int(round(x2 * w)), int(round(y2 * h))]
    px[0], px[1] = max(px[0], 0), max(px[1], 0)
    px[2], px[3] = min(px[2], w), min(px[3], h)
    assert px[0] < px[2] and px[1] < px[3], f"degenerate bbox {px} in {w}x{h}"
    return px


def edit_region(pipe, image, bbox_px, instruction, margin, target_long, steps, cfg):
    """Edit only the padded crop around bbox_px; paste it back over the original.

    Returns (edited_full_image, roi_delta) where roi_delta is the mean absolute
    pixel change inside the tight bbox (a cheap no-op detector).
    """
    W, H = image.size
    x1, y1, x2, y2 = bbox_px
    bw, bh = x2 - x1, y2 - y1

    # 1. Crop the box plus a context margin (used for coherence, kept in the edit).
    mx, my = int(bw * margin), int(bh * margin)
    cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
    cx2, cy2 = min(W, x2 + mx), min(H, y2 + my)
    crop = image.crop((cx1, cy1, cx2, cy2))
    cw, ch = crop.size

    # 2. Adaptive resize: bring the long side to target (down for big scans, up
    #    for tiny boxes) so the model works near its ~1 MP sweet spot.
    scale = target_long / max(cw, ch)
    work = crop.resize((max(1, round(cw * scale)), max(1, round(ch * scale))),
                       Image.LANCZOS)

    # 3. The model only ever sees this crop — it cannot touch pixels outside it.
    edited = pipe(image=[work], prompt=instruction, true_cfg_scale=cfg,
                  negative_prompt=" ", num_inference_steps=steps).images[0]
    edited = edited.resize((cw, ch), Image.LANCZOS)

    # 4. Feathered paste of the whole crop back over a copy of the original.
    #    Outside the crop is bitwise-identical; a soft inner border avoids a seam.
    feather = max(2, round(0.04 * min(cw, ch)))
    mask = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(mask).rectangle(
        [feather, feather, cw - feather, ch - feather], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(feather))
    out = image.copy()
    out.paste(edited, (cx1, cy1), mask)

    # 5. No-op detector: how much did the tight bbox actually change?
    a = np.asarray(image.crop((x1, y1, x2, y2)), dtype=np.int16)
    b = np.asarray(out.crop((x1, y1, x2, y2)), dtype=np.int16)
    roi_delta = float(np.abs(a - b).mean())
    return out, roi_delta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="edit-dataset JSON (list of rows)")
    ap.add_argument("--image-root", required=True, help="root for row['image'][0]")
    ap.add_argument("--out-dir", required=True, help="where edited PNGs are written")
    ap.add_argument("--out-manifest", required=True, help="augmented records (JSONL, appended)")
    ap.add_argument("--start", type=int, default=0, help="shard start index (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="shard end index (exclusive)")
    ap.add_argument("--margin", type=float, default=0.2, help="context margin as fraction of box size")
    ap.add_argument("--target-long", type=int, default=1024, help="crop long-side working resolution")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA visible — wrong node"

    rows = json.load(open(args.manifest))
    end = args.end if args.end is not None else len(rows)
    shard = rows[args.start:end]
    os.makedirs(args.out_dir, exist_ok=True)

    # Resume: skip ids already recorded in this shard's manifest.
    done = set()
    if os.path.exists(args.out_manifest):
        with open(args.out_manifest) as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")

    n_ok = n_skip = n_fail = 0
    with open(args.out_manifest, "a") as out_f:
        for i, row in enumerate(shard, start=args.start):
            eid = example_id(row)
            out_png = os.path.join(args.out_dir, f"{eid}.png")
            if eid in done or os.path.exists(out_png):
                n_skip += 1
                continue
            try:
                src = os.path.join(args.image_root, row["image"][0])
                image = Image.open(src).convert("RGB")
                w, h = image.size
                bbox_px = denorm_bbox(row["bboxes"][0], w, h)
                edited, roi_delta = edit_region(
                    pipe, image, bbox_px, row["edit"],
                    args.margin, args.target_long, args.steps, args.cfg)
                edited.save(out_png)
                rec = dict(row)
                rec["id"] = eid
                rec["edited_image"] = out_png
                rec["roi_delta"] = roi_delta
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                n_ok += 1
                print(f"[{i}] {eid} ok delta={roi_delta:.1f}", flush=True)
            except Exception as e:  # isolate one bad row from a multi-hour shard
                n_fail += 1
                print(f"[{i}] {eid} FAIL {type(e).__name__}: {e}", flush=True)

    print(f"done: ok={n_ok} skip={n_skip} fail={n_fail} "
          f"(shard {args.start}:{end} of {len(rows)})", flush=True)


if __name__ == "__main__":
    main()
