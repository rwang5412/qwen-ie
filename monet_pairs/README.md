# Monet-SFT counterfactual pairs (Phase A)

Builds the swap-loss pair set from Monet-SFT-125k: for each editable row,
produce (original', aux', obs', y') where ONLY the observation region changed,
plus a frozen 10% holdout for the flip eval.

## Plan amendments discovered from the data (why the scripts differ from the plan)

1. **Monet-SFT has NO bbox coordinates anywhere.** The plan's Step 1 assert
   would drop everything. Recovery is deterministic instead (step2):
   Visual_CoT aux = unscaled crop -> FFT template match (accept NCC>=0.95,
   MAE<=15); ReFocus aux = full image + drawn highlight box -> pixel-diff
   bounding rect. Rows failing recovery are dropped, not repaired.
2. **ReFocus Steps 3/4 are inverted.** Its aux is not a crop, so we edit the
   ORIGINAL inside the recovered box and rebuild aux' by transplanting the
   drawn-box stroke pixels. Visual_CoT follows the plan verbatim
   (edit aux crop -> paste at bbox).
3. **"No model call" new_content limits the set to number/color observations.**
   Free-text observations (~92k rows) have no deterministic "materially
   different" swap and are dropped by step1. Yield: 13,138 pairs
   (12,848 Visual_CoT + 290 ReFocus; 10,284 color / 2,854 number).
4. **Step 6 needs no captioner VLM for this set.** obs'/y' are deterministic
   string swaps computed in step1 (y' != y asserted there). The Step 5
   verifier still gates legibility/consistency of the rendered edit.

## Pipeline (steps 1,2 local Mac; 3,5,7 GPU cluster)

| script | plan steps | where | in -> out |
|---|---|---|---|
| step1_filter.py | 1,2,6-text,8 | Mac (done) | train.json -> pairs_manifest.jsonl (id, obs/obs', y/y', edit_prompt, split) |
| step2_recover_bbox.py | bbox recovery | Mac (done) | + bbox, scores; exports kept images to counterfactual/images/ |
| step2b_propose_swaps.py | 3-spec for freetext | cluster GPU | Qwen2.5-VL proposes obs' for kind=='freetext' (dog->cat, smiling->frowning); color/number pass through unchanged -> pairs_ready.jsonl (PLAN CHANGE: model call for new_content, approved for diversity) |
| step3_edit_composite.py | 3,4 | cluster GPU | -> edited/{id}_orig.png, {id}_aux.png, roi_delta |
| step5_verify.py | 5,6-assert | cluster GPU | + verified flag (Qwen2.5-VL-7B answers Q from original' alone) |
| step7_encode_zprime.py | 7 | cluster GPU + Monet repo | verified rows -> zprime_cache/{id}.pt [K,H] (stub: wire 2 calls) |
| make_base_rows.py | training format | Mac (done) | monet125k -> data/base_rows.json (trainer-native interleaved; VERIFY vs Monet loader's three-row dump) |
| emit_sidecar.py | training format | cluster, LAST | verified.jsonl + Z' cache -> data/cf/pairs.jsonl + cf/images/ + cf/z/ (fp16) + splits/eval_flip_ids.txt |

Step 8 split is assigned in step1 by stable hash (10.0% holdout), written once.
Flip eval set = verified AND split=="holdout"; emitted as split=="eval_flip"
in the sidecar and frozen in splits/eval_flip_ids.txt.

## Training-time layout (the contract)

```
data/
  base_rows.json            # 125,072 rows, trainer-native (schema UNVERIFIED
                            #   until checked against Monet's loader dump)
  cf/
    pairs.jsonl             # final sidecar: row_id join key, obs/obs', y/y',
                            #   bbox_norm, z_prime_path, verifier_pass, split
    images/{id}_aux.png, {id}_full.png
    z/{id}.pt               # [K, H] fp16
  splits/eval_flip_ids.txt
```
The collator joins sidecar->base by row_id at load time; no merged format.

## Cluster run

Transfer `counterfactual/` (manifest + images, ~a few GB) to
`/scratch/$USER/monet-data/`, pull this repo, then per GPU shard:

```bash
source /scratch/$USER/qwen-venv/bin/activate
export HF_HOME=/scratch/$USER/hf HF_HUB_OFFLINE=1
python monet_pairs/step3_edit_composite.py \
  --manifest /scratch/$USER/monet-data/pairs_bbox.jsonl \
  --image-root /scratch/$USER/monet-data/images \
  --out-dir /scratch/$USER/monet-data/edited \
  --out-manifest /scratch/$USER/monet-data/edited.jsonl \
  --start 0 --end 20          # pilot first; then shard 0..13138
  # optional Lightning: --lora lightx2v/Qwen-Image-Edit-2511-Lightning \
  #   --lora-weight Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
  #   --steps 8 --cfg 1.0

# verifier needs (login node): pip install qwen-vl-utils
# and (DTN): hf download Qwen/Qwen2.5-VL-7B-Instruct
python monet_pairs/step5_verify.py \
  --manifest /scratch/$USER/monet-data/edited.jsonl \
  --out-manifest /scratch/$USER/monet-data/verified.jsonl
```

step7 runs inside the Monet environment (checkpoint + repo) — wire the two
NotImplementedError calls first.
