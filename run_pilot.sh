#!/usr/bin/env bash
# Pilot run: 10 document/text edits (sroie/dude) + 10 natural-image edits
# (flickr30k), timed. Run INSIDE a GPU salloc, after the venv + weights exist.
#
#   salloc --gpus-per-node=a100:1 --mem=64G --cpus-per-task=8 --time=01:00:00
#   bash /scratch/$USER/qwen-code/run_pilot.sh
#
# Then eyeball the PNGs (scp them to your Mac) and read the `time` lines to
# estimate the full-run cost before launching the sbatch array.
set -eo pipefail

DATA=/scratch/$USER/qwen-data
GEN="$(cd "$(dirname "$0")" && pwd)/generate.py"

source /scratch/$USER/qwen-venv/bin/activate
export HF_HOME=/scratch/$USER/hf HF_HUB_OFFLINE=1

echo "== document/text edits (sroie/dude): rows 0-10 =="
time python "$GEN" \
  --manifest     "$DATA/viscot_sroie_dude_lvr_unique_q_with_edits.json" \
  --image-root   "$DATA/images" \
  --out-dir      "$DATA/edited" \
  --out-manifest "$DATA/pilot_doc.jsonl" \
  --start 0 --end 10

echo "== natural-image edits (flickr30k): rows 0-10 =="
time python "$GEN" \
  --manifest     "$DATA/viscot_363k_lvr_unique_q_with_edits.json" \
  --image-root   "$DATA/images" \
  --out-dir      "$DATA/edited" \
  --out-manifest "$DATA/pilot_nat.jsonl" \
  --start 0 --end 10

echo
echo "Edited images -> $DATA/edited/{id}.png"
echo "Pilot records -> $DATA/pilot_doc.jsonl , $DATA/pilot_nat.jsonl"
