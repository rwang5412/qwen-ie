#!/usr/bin/env python
"""Phase A / Steps 3+4 (GPU): edit each pair's region, build (original', aux').

  - Visual_CoT: the aux IS the crop. Edit it (adaptive resize -> model ->
    resize back) = aux'; paste aux' into the original at the recovered bbox
    with a light feathered seam = original'.
  - ReFocus: aux is original + drawn highlight box. Edit the ORIGINAL inside
    the box interior (stroke inset) = original'; then transplant the stroke
    pixels from the old aux onto original' = aux'. Both deterministic.

Reads pairs_bbox.jsonl; writes edited/{id}_orig.png + edited/{id}_aux.png and
appends full pair records (with roi_delta) to --out-manifest. Shardable via
--start/--end, resumable via skip-existing. Same Lightning flags as generate.py.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
STROKE_INSET = 8  # px inside the ReFocus diff box, past the drawn stroke


def edit_patch(pipe, patch, prompt, target_long, steps, cfg):
    """Run the editor on one image patch at working resolution; return same size."""
    w, h = patch.size
    scale = target_long / max(w, h)
    work = patch.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                        Image.LANCZOS)
    out = pipe(image=[work], prompt=prompt, true_cfg_scale=cfg,
               negative_prompt=" ", num_inference_steps=steps).images[0]
    return out.resize((w, h), Image.LANCZOS)


def feather_paste(base, patch, box):
    """Paste patch at box=[x1,y1,x2,y2] with a small feathered border."""
    w, h = patch.size
    f = max(2, round(0.04 * min(w, h)))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([f, f, w - f, h - f], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(f))
    out = base.copy()
    out.paste(patch, (box[0], box[1]), mask)
    return out


def roi_delta(a, b, box):
    A = np.asarray(a.crop(box), dtype=np.int16)
    B = np.asarray(b.crop(box), dtype=np.int16)
    return float(np.abs(A - B).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="pairs_bbox.jsonl")
    ap.add_argument("--image-root", required=True, help="exported pair images root")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--target-long", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-weight", default=None)
    ap.add_argument("--kind", default=None, choices=["color", "number"],
                    help="restrict to one edit kind; --start/--end index the filtered list")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA visible — wrong node"

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.kind:
        rows = [r for r in rows if r["kind"] == args.kind]
        print(f"kind={args.kind}: {len(rows)} rows", flush=True)
    end = args.end if args.end is not None else len(rows)
    shard = rows[args.start:end]
    os.makedirs(args.out_dir, exist_ok=True)

    done = set()
    if os.path.exists(args.out_manifest):
        with open(args.out_manifest) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
    if args.lora:
        pipe.load_lora_weights(args.lora, weight_name=args.lora_weight)
        pipe.fuse_lora()
        print(f"fused LoRA {args.lora} ({args.lora_weight})", flush=True)

    n_ok = n_skip = n_fail = 0
    with open(args.out_manifest, "a") as out_f:
        for i, r in enumerate(shard, start=args.start):
            rid = r["id"]
            p_orig = os.path.join(args.out_dir, f"{rid}_orig.png")
            p_aux = os.path.join(args.out_dir, f"{rid}_aux.png")
            if rid in done or (os.path.exists(p_orig) and os.path.exists(p_aux)):
                n_skip += 1
                continue
            try:
                orig = Image.open(os.path.join(args.image_root, r["orig_image"])).convert("RGB")
                aux = Image.open(os.path.join(args.image_root, r["aux_image"])).convert("RGB")
                box = r["bbox"]

                if r["subset"] == "Visual_CoT":
                    aux_new = edit_patch(pipe, aux, r["edit_prompt"],
                                         args.target_long, args.steps, args.cfg)
                    orig_new = feather_paste(orig, aux_new, box)
                else:  # ReFocus
                    x1, y1, x2, y2 = box
                    ix1, iy1 = x1 + STROKE_INSET, y1 + STROKE_INSET
                    ix2, iy2 = x2 - STROKE_INSET, y2 - STROKE_INSET
                    assert ix2 - ix1 >= 8 and iy2 - iy1 >= 8, "box too thin after inset"
                    patch = edit_patch(pipe, orig.crop((ix1, iy1, ix2, iy2)),
                                       r["edit_prompt"], args.target_long,
                                       args.steps, args.cfg)
                    orig_new = feather_paste(orig, patch, [ix1, iy1, ix2, iy2])
                    stroke = np.abs(np.asarray(orig, np.int16)
                                    - np.asarray(aux, np.int16)).sum(2) > 40
                    a_new = np.asarray(orig_new).copy()
                    a_new[stroke] = np.asarray(aux)[stroke]
                    aux_new = Image.fromarray(a_new)

                orig_new.save(p_orig)
                aux_new.save(p_aux)
                rec = dict(r)
                rec["orig_new"] = p_orig
                rec["aux_new"] = p_aux
                rec["roi_delta"] = round(roi_delta(orig, orig_new, box), 2)
                out_f.write(json.dumps(rec) + "\n")
                out_f.flush()
                n_ok += 1
                print(f"[{i}] {rid} ok delta={rec['roi_delta']}", flush=True)
            except Exception as e:
                n_fail += 1
                print(f"[{i}] {rid} FAIL {type(e).__name__}: {e}", flush=True)

    print(f"done: ok={n_ok} skip={n_skip} fail={n_fail} (shard {args.start}:{end})",
          flush=True)


if __name__ == "__main__":
    main()
