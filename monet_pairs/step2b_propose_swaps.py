#!/usr/bin/env python
"""Step 2b (GPU): fill the edit spec for kind=='freetext' rows via VLM proposal.

For each free-text row, Qwen2.5-VL-7B sees the aux crop + the question + the
current observation and proposes a replacement observation: same category,
short, visually renderable, materially different (e.g. dog -> cat,
smiling -> frowning). Failed/degenerate proposals drop the row — no retry.

Reads pairs_bbox.jsonl (all kinds). Writes --out: color/number rows passed
through UNCHANGED; freetext rows completed with obs_new / new_content /
edit_prompt / y_new. The output is what step3 consumes. Sharded/resumable.

Needs the same model as step5: Qwen/Qwen2.5-VL-7B-Instruct (+ qwen-vl-utils).
"""
import argparse
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROPOSER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = """This image crop answers the question: "{q}"
The current grounded observation is: "{obs}"

Propose ONE replacement observation that is:
- the same kind of thing (object->different object, action->different action),
- clearly and materially different from the current observation,
- short (at most 6 words), concrete, and physically renderable by an image
  editor changing ONLY this crop.

Reply with ONLY the replacement phrase, nothing else."""


def ask(model, processor, image, text, max_new=24):
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image],
                       return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)[0].strip()


def valid(obs, prop):
    a, b = obs.lower().strip(), prop.lower().strip()
    if not b or len(prop.split()) > 6 or '"' in prop or '\n' in prop:
        return False
    return a != b and a not in b and b not in a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="pairs_bbox.jsonl")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True, help="completed manifest for step3")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA visible — wrong node"

    rows = [json.loads(l) for l in open(args.manifest)]
    end = args.end if args.end is not None else len(rows)
    shard = rows[args.start:end]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        PROPOSER_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(PROPOSER_ID)

    n_pass = n_prop = n_drop = 0
    with open(args.out, "a") as out_f:
        for i, r in enumerate(shard, start=args.start):
            if r["id"] in done:
                continue
            if r["kind"] != "freetext":
                out_f.write(json.dumps(r) + "\n")
                n_pass += 1
                continue
            img = Image.open(os.path.join(args.image_root, r["aux_image"])).convert("RGB")
            prop = ask(model, processor, img,
                       PROMPT.format(q=r["question"], obs=r["obs"]))
            prop = re.sub(r'^["\']|["\'.]$', '', prop).strip()
            if not valid(r["obs"], prop):
                n_drop += 1
                print(f"[{i}] {r['id']} DROP proposal={prop!r}", flush=True)
                continue
            r2 = dict(r)
            r2["obs_new"] = prop
            r2["old_content"], r2["new_content"] = r["obs"], prop
            r2["edit_prompt"] = f"Change {r['obs']} to {prop}"
            y_new = r["y"].replace(f"<observation>{r['obs']}</observation>",
                                   f"<observation>{prop}</observation>")
            if y_new == r["y"]:
                n_drop += 1
                continue
            r2["y_new"] = y_new
            out_f.write(json.dumps(r2) + "\n")
            out_f.flush()
            n_prop += 1
            if n_prop % 50 == 0:
                print(f"[{i}] proposed={n_prop} dropped={n_drop}", flush=True)

    print(f"done: passthrough={n_pass} proposed={n_prop} dropped={n_drop}",
          flush=True)


if __name__ == "__main__":
    main()
