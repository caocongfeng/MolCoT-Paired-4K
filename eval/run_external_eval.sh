#!/bin/bash
# Evaluate a trained adapter (or a base model) on an external test set with the evaluator of sft/train_sft.py:
#
#   GPU=0 bash eval/run_external_eval.sh <testset_dir> <model> <name> [run_id]
#
#   testset_dir  a folder written by eval/prepare_external_testsets.py, e.g. data_external/chebi20
#   model        an SFT adapter directory (e.g. runs/qwen3.5-4b-base/curated/run_000/model/checkpoint-844),
#                a GRPO adapter (runs_grpo/.../adapter_final) or a Hugging Face model id (Qwen/Qwen3.5-4B-Base)
#   name         label of the output folder, e.g. sft_curated
#   run_id       used for the evaluation seed 2000 + run_id (default 0)
#
# Decoding as in the paper: greedy, at most 768 new tokens, chat template (Qwen3.5). Greedy only (best-of-n 1).
# Output: runs_external/<testset>/<name>/<variant>/run_<id>/test_predictions.json
set -eo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TESTSET=${1:?testset dir}; MODEL=${2:?model}; NAME=${3:?name}; RUN=${4:-0}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
VARIANT=original
[[ "$NAME" == *curated* ]] && VARIANT=curated
cd "$REPO/sft"
exec python -u train_sft.py --eval-only --no-wandb \
  --model "$MODEL" --experiment-name "$NAME" --variant "$VARIANT" --run-id "$RUN" --eval-seed $((2000 + RUN)) \
  --format-style chat --data-dir "$(cd "$REPO" && realpath "$TESTSET")" --output-dir "$REPO/runs_external/$(basename "$TESTSET")" \
  --max-seq-length 1280 --max-new-tokens 768 --best-of-n 1 \
  --eval-batch-size "${EVAL_BS:-64}" --gen-batch-size "${GEN_BS:-256}"
