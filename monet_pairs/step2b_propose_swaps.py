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
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

PROPOSER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = """This image crop answers the question: "{q}"
The current grounded observation is: "{obs}"

Propose ONE replacement observation, subject to ALL of these rules:
- The change must be FULLY renderable by repainting ONLY this crop: the
  entire thing being changed is visible inside the crop, and nothing outside
  the crop would need to change for the result to look coherent.
- Allowed: swapping an object's identity, its color/texture/material, text or
  logos, clothing or held items that are fully visible in the crop.
- Forbidden: changes to pose, body position, motion, location, or anything
  abstract (moods, relationships, vibes). Forbidden if the subject is only
  partially visible in the crop.
- The replacement must be the same kind of thing, clearly and materially
  different, short (at most 6 words), and concrete.

If no replacement satisfies every rule, reply with exactly: SKIP
Otherwise reply with ONLY the replacement phrase, nothing else."""

CONTAIN_PROMPT = """The red box in this image marks a region that will be
repainted by an image editor to change "{obs}" into "{prop}". The editor
blends the box border, so SMALL spill past the edges (a sleeve tip, a few
hair strands, a sliver of an object grazing the boundary) does NOT matter.

Answer NOT_CONTAINED only if one of these clearly holds:
- a SUBSTANTIAL part of the thing being changed (roughly a quarter of it or
  more) lies outside the red box, so the unedited remainder would visibly
  contradict the edit,
- the change cannot look right without also altering things outside the box
  (a mid-air pose continuing outside, contact with ground/water, a large
  mirror reflection),
- the box cuts through the middle of the subject mid-action.

If the thing being changed is essentially inside the box, answer CONTAINED.
Reply with exactly one word: CONTAINED or NOT_CONTAINED."""

JUDGE_HEADER = """You are shown the ORIGINAL image. The red box marks the ONLY
region an image editor will repaint. The PLAN is to change "{obs}" into
"{prop}".

{question}

Answer YES or NO only."""

# each judge question asked in its OWN call — multi-question formats collapse
# to uniform answers on small models
JUDGE_QS = [
    ('pose', 'Does changing "{obs}" into "{prop}" alter a body pose, an '
             'action in progress, motion, or location (e.g. jumping -> '
             'running, sitting -> standing)?'),
    ('plural-question', 'Consider only this question text: "{q}". Does it '
             'ask about MULTIPLE people or objects, or a group, or the '
             'scene in general (e.g. "the participants", "the men", '
             '"everyone") rather than one single specific subject?'),
    ('others-outside', 'Look at the parts of the image OUTSIDE the red box. '
             'Do you see more of the same kind of thing as "{obs}" out '
             'there (other instances that repainting the box would not '
             'change)?'),
    ('nonsense', 'Would "{prop}" be physically absurd at the exact spot of '
             'the red box — a thing that cannot exist there (e.g. a shirt '
             'where someone\'s legs are)?'),
    ('partial', 'Look at "{obs}" in the image. Does part of it stick out '
             'PAST the edges of the red box (hair, garment, or object '
             'continuing outside the box)?'),
]


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
    if b == 'skip' or not b or len(prop.split()) > 6 or '"' in prop or '\n' in prop:
        return False
    return a != b and a not in b and b not in a


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="pairs_bbox.jsonl")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out", required=True, help="completed manifest for step3")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--model", default=PROPOSER_ID,
                    help="VLM for propose+contain+judge (e.g. Qwen/Qwen2.5-VL-32B-Instruct)")
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
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"filter model: {args.model}", flush=True)

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
            # containment gate: judge on the full original with the bbox drawn —
            # subjects crossing the box edge (pose, contact, reflections) drop here.
            boxed = Image.open(os.path.join(args.image_root, r["orig_image"])).convert("RGB")
            ImageDraw.Draw(boxed).rectangle(r["bbox"], outline=(255, 0, 0),
                                            width=max(3, boxed.width // 200))
            verdict = ask(model, processor, boxed,
                          CONTAIN_PROMPT.format(obs=r["obs"], prop=prop), max_new=8)
            if not verdict.strip().upper().startswith("CONTAINED"):
                n_drop += 1
                print(f"[{i}] {r['id']} DROP not-contained ({prop!r})", flush=True)
                continue
            # adversarial judge: sees the question too — kills multi-instance
            # (answer wouldn't flip), pose/action slips, and physical nonsense.
            q_text = r["question"].replace("<image>", "").strip()
            yes = set()
            for tag, qt in JUDGE_QS:
                a = ask(model, processor, boxed,
                        JUDGE_HEADER.format(
                            obs=r["obs"], prop=prop,
                            question=qt.format(obs=r["obs"], prop=prop, q=q_text)),
                        max_new=4)
                if a.strip().upper().startswith("YES"):
                    yes.add(tag)
            # multi-instance needs BOTH halves: plural question AND identical
            # instances outside the box; alone, neither blocks the edit.
            fatal = ({'pose', 'nonsense', 'partial'} & yes) or \
                    ({'plural-question', 'others-outside'} <= yes)
            if fatal:
                n_drop += 1
                print(f"[{i}] {r['id']} DROP judge ({prop!r}) :: {','.join(sorted(yes))}",
                      flush=True)
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
