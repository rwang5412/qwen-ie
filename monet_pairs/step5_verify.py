#!/usr/bin/env python
"""Phase A / Step 5 (GPU): verify each edited pair from original' ALONE, or
reject it. No retry loop, no fallback editor.

Verifier: Qwen2.5-VL-7B-Instruct. Two checks per row:
  (a) legible + unambiguous: ask the row's original question against
      original'; the answer must contain new_content and NOT old_content.
  (b) internally consistent: a direct yes/no probe on the edited image.
Both must pass -> verified=true; anything else -> rejected. Expect 20-40%
rejection.

Also enforces Step 6's assert y' != y (already guaranteed upstream; rows where
the verifier's answer equals the old answer are exactly the flip failures).

Writes every input record + verified flag + verifier answers to --out-manifest;
the verified subset is your pair set, verified & split=='holdout' is the flip
eval (Step 8's split was assigned in step1 and is never reassigned).

Needs: pip install qwen-vl-utils   and   hf download Qwen/Qwen2.5-VL-7B-Instruct
"""
import argparse
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

VERIFIER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


def norm(s):
    return re.sub(r'[,\s]', '', s.lower())


def ask(model, processor, image, text, max_new=64):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="step3 out-manifest")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA visible — wrong node"

    rows = [json.loads(l) for l in open(args.manifest)]
    end = args.end if args.end is not None else len(rows)
    shard = rows[args.start:end]

    done = set()
    if os.path.exists(args.out_manifest):
        with open(args.out_manifest) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VERIFIER_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(VERIFIER_ID)

    n_pass = n_rej = 0
    with open(args.out_manifest, "a") as out_f:
        for i, r in enumerate(shard, start=args.start):
            if r["id"] in done:
                continue
            img = Image.open(r["orig_new"]).convert("RGB")

            ans = ask(model, processor, img, r["question"])
            has_new = norm(r["new_content"]) in norm(ans)
            has_old = norm(r["old_content"]) in norm(ans)
            legible = has_new and not has_old

            cons = ask(model, processor, img,
                       "Does this image look natural and internally consistent, "
                       "with no garbled text, artifacts, or contradictory "
                       "content? Answer strictly yes or no.")
            consistent = cons.lower().lstrip().startswith("yes")

            # seam gate for VLM-proposed edits: inspect the edited region plus
            # margin for boundary breaks (misaligned limbs, cut poses, seams).
            seam_ok, seam = True, ""
            if r.get("kind") == "freetext" and consistent:
                x1, y1, x2, y2 = r["bbox"]
                mx, my = int((x2 - x1) * 0.4), int((y2 - y1) * 0.4)
                region = img.crop((max(0, x1 - mx), max(0, y1 - my),
                                   min(img.width, x2 + mx), min(img.height, y2 + my)))
                seam = ask(model, processor, region,
                           "Look closely at this image region. Is it physically "
                           "coherent — no misaligned or truncated body parts, no "
                           "visible editing seam, no content that breaks abruptly "
                           "at an invisible boundary? Answer strictly yes or no.")
                seam_ok = seam.lower().lstrip().startswith("yes")

            r2 = dict(r)
            r2["verifier_answer"] = ans
            r2["verifier_consistent"] = cons[:80]
            r2["verifier_seam"] = seam[:80]
            r2["verified"] = bool(legible and consistent and seam_ok)
            out_f.write(json.dumps(r2) + "\n")
            out_f.flush()
            n_pass += r2["verified"]
            n_rej += not r2["verified"]
            print(f"[{i}] {r['id']} verified={r2['verified']} "
                  f"(new={has_new} old={has_old} cons={consistent})", flush=True)

    tot = n_pass + n_rej
    print(f"done: verified={n_pass} rejected={n_rej} "
          f"({100 * n_rej / max(tot, 1):.0f}% rejection)", flush=True)


if __name__ == "__main__":
    main()
