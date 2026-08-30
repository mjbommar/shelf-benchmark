#!/usr/bin/env bash
# Re-run every analysis that reads the results directories, in dependency
# order, after the model set changes.
#
# These artifacts are derived, not authored. Adding a model to config.yaml
# invalidates all of them at once, and hand-editing a number in the paper
# without re-running the script that produced it is how a table drifts from
# its evidence. Run this, then verify_paper_numbers.py, then write prose.
#
# GPU work (the clustering stability gate) runs one corpus at a time on the
# single free device, and only for models not already measured: a model's
# median ARI over seeds does not depend on which other models are in the
# sweep, so previously stored per-seed values are reused and only the rank
# statistics are recomputed. Nothing here runs concurrently: three jobs on one card
# is what OOM'd this project before.
#
# Usage:
#   bash scripts/refresh_analyses.sh            # everything
#   bash scripts/refresh_analyses.sh --no-gpu   # skip the clustering gate
set -u
cd "$(dirname "$0")/.."

GPU_OK=1
[ "${1:-}" = "--no-gpu" ] && GPU_OK=0

# The weight duplicate: ogbert-110m-base and ogbert-110m-sentence are one
# safetensors blob with one SHA-256. Also gte_modernbert_8k, which shares
# weights with the 2k entry; it belongs to the truncation experiment, not to
# the main model set, and letting both into a rank correlation double-counts.
DEDUP="ogbert_110m_sentence,gte_modernbert_8k"

# Models no practitioner would deploy, for the restricted-range column.
DEGEN="$DEDUP,ogbert_110m_base,ogbert_2m_sentence,ogbert_v1_mlm,roberta"

CL="--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=CUDA_VISIBLE_DEVICES=1"
step() { echo; echo "=== $* ==="; }
fail=0
guard() { "$@" || { echo "FAILED: $*"; fail=1; }; }

step "1. clustering stability, five seeds, new models only"
if [ "$GPU_OK" = "1" ]; then
  MODELS=$(uv run python -c "
import yaml
c = yaml.safe_load(open('scripts/baselines/config.yaml'))['models']
print(' '.join(k for k, v in c.items()
                if 'finetune' not in k and 'clustering' in (v.get('supports') or [])))
" 2>/dev/null)
  for pair in all:all transfer_gutenberg:transfer_gutenberg transfer_lcshbench:transfer_lcshbench; do
    corpus="${pair%%:*}"; tag="${pair##*:}"
    out="results/transfer/clustering_stability_${tag}.json"
    echo "  $corpus -> $out"
    guard systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 \
      $CL --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --setenv=SHELF_NUM_THREADS=8 --setenv=PYTHONHASHSEED=0 \
      --setenv=SHELF_DATA_DIR="data/hf_dataset/$corpus" --setenv=COLUMNS=200 \
      uv run python scripts/clustering_stability_merge.py \
        --existing "$out" --models $MODELS --task lcc_clustering --output "$out"
  done
else
  echo "  skipped (--no-gpu); existing medians will be reused"
fi

step "2. rank agreement, pooled and the earlier build"
guard uv run python scripts/rank_agreement.py \
  --corpus "shelf_pooled=results/pooled/baselines" \
  --corpus "gutenberg=results/transfer_gutenberg/baselines" \
  --corpus "lcshbench=results/transfer_lcshbench/baselines" \
  --exclude-models "$DEDUP" \
  --output results/transfer/rank_agreement_pooled_dedup.json

step "3. cross-formulation, pre-registered intersection"
guard uv run python scripts/cross_formulation_agreement.py \
  --shelf results/pooled/baselines \
  --shelf-pairs results/all_pairs/baselines \
  --natural "gutenberg=results/transfer_gutenberg/baselines" \
  --natural "lcshbench=results/transfer_lcshbench/baselines" \
  --clustering-medians "shelf=results/transfer/clustering_stability_all.json" \
  --clustering-medians "gutenberg=results/transfer/clustering_stability_transfer_gutenberg.json" \
  --clustering-medians "lcshbench=results/transfer/clustering_stability_transfer_lcshbench.json" \
  --exclude-models "$DEDUP" --n-boot 2000 \
  --output results/transfer/cross_formulation.json

step "4. cross-formulation, restricted to models worth deploying"
guard uv run python scripts/cross_formulation_agreement.py \
  --shelf results/pooled/baselines \
  --shelf-pairs results/all_pairs/baselines \
  --natural "gutenberg=results/transfer_gutenberg/baselines" \
  --natural "lcshbench=results/transfer_lcshbench/baselines" \
  --clustering-medians "shelf=results/transfer/clustering_stability_all.json" \
  --clustering-medians "gutenberg=results/transfer/clustering_stability_transfer_gutenberg.json" \
  --clustering-medians "lcshbench=results/transfer/clustering_stability_transfer_lcshbench.json" \
  --exclude-models "$DEGEN" --n-boot 2000 \
  --output results/transfer/cross_formulation_restricted.json

step "5. coarse against fine subject task"
guard uv run python scripts/task_rank_divergence.py \
  --exclude-models "$DEDUP" \
  --output results/transfer/task_rank_divergence.json

step "6. gate: does the paper still match its evidence?"
uv run python scripts/verify_paper_numbers.py
vp=$?

echo
if [ "$fail" = "0" ] && [ "$vp" = "0" ]; then
  echo "All analyses refreshed and the paper matches. Safe to write prose."
else
  echo "Refresh incomplete (analysis fail=$fail, verifier exit=$vp)."
  echo "The verifier is EXPECTED to fail here until the tables are updated:"
  echo "its job is to name every number that moved."
fi
exit $(( fail ))
