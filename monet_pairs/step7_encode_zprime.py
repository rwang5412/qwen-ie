#!/usr/bin/env python
"""Phase A / Step 7 (GPU, Monet repo required): encode Z' for every VERIFIED pair.

For each verified row, run the frozen Monet-SFT Stage-2 teacher path on
(question, original', aux'): teacher-force the trajectory up to the latent
block with aux' visible to the latents, capture the K latent vectors, and
cache Z_prime [K, H] to disk keyed by row id.

This script is a STUB with the exact interface — the two marked calls must be
wired to the Monet codebase (github.com/NOVAglow646/Monet), which is not part
of this repo. Everything else (I/O, filtering, caching, resume) is complete.

Run inside the Monet environment with the Stage-2 checkpoint:
  python step7_encode_zprime.py --manifest verified.jsonl \
      --checkpoint /path/to/monet_sft_stage2 --out-dir zprime_cache
"""
import argparse
import json
import os

import torch


def load_teacher(checkpoint):
    """WIRE ME: load the frozen Monet Stage-2 teacher (model + processor)."""
    raise NotImplementedError(
        "import from the Monet repo, e.g. "
        "monet.modeling.load_stage2_teacher(checkpoint)")


def encode_latents(model, processor, question, orig_img, aux_img):
    """WIRE ME: teacher-forced forward to the latent block, aux' visible.
    Must return the K latent vectors as a [K, H] tensor."""
    raise NotImplementedError(
        "teacher-force (question, original') with aux' as the latent-block "
        "target; capture hidden states at the <abs_vis_token> block")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="step5 out-manifest")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "no CUDA visible — wrong node"
    os.makedirs(args.out_dir, exist_ok=True)

    rows = [json.loads(l) for l in open(args.manifest)]
    verified = [r for r in rows if r.get("verified")]
    print(f"{len(verified)} verified rows of {len(rows)}")

    model, processor = load_teacher(args.checkpoint)

    n_ok = n_skip = 0
    for r in verified:
        out_path = os.path.join(args.out_dir, f"{r['id']}.pt")
        if os.path.exists(out_path):
            n_skip += 1
            continue
        from PIL import Image
        orig_new = Image.open(r["orig_new"]).convert("RGB")
        aux_new = Image.open(r["aux_new"]).convert("RGB")
        z = encode_latents(model, processor, r["question"], orig_new, aux_new)
        assert z.dim() == 2, f"expected [K, H], got {tuple(z.shape)}"
        torch.save({"id": r["id"], "z_prime": z.cpu(), "split": r["split"]},
                   out_path)
        n_ok += 1
    print(f"done: encoded={n_ok} skipped={n_skip}")


if __name__ == "__main__":
    main()
