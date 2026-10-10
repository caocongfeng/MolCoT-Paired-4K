#!/bin/bash
# One GRPO run on one GPU (Qwen3.5-4B-Base, settings of the paper):
#
#   GPU=0 bash grpo/run_grpo.sh <start> <run_id> [extra train_grpo.py arguments]
#
#   start   curated  : continue the Curated-CoT SFT adapter  (runs/qwen3.5-4b-base/curated/run_<id>/model/checkpoint-844)
#           original : continue the Original-CoT SFT adapter (runs/qwen3.5-4b-base/original/run_<id>/model/checkpoint-844)
#           base     : Qwen3.5-4B-Base without SFT, fresh LoRA adapter
#   run_id  0-2 in the paper (seed 1000 + id, evaluation seed 2000 + id, as in SFT)
#
# The SFT adapters come from:  bash sft/run_sft.sh scales qwen3.5-4b <variant> <run_id>
# Smoke test:  GPU=0 bash grpo/run_grpo.sh curated 0 --max-steps 2 --limit-test 32 --tag _smoke --no-wandb
# Output:      runs_grpo/qwen3.5-4b-base/<start>/run_<id>/ (evaluations before SFT, after SFT, after each GRPO epoch).
# Environment: requirements-grpo.txt (vLLM 0.29, TRL 1.13); vLLM runs colocated on the same GPU.
set -eo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
START=${1:?start: curated | original | base}; RUN_ID=${2:?run id}; shift 2

export CUDA_VISIBLE_DEVICES=${GPU:-0}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WANDB_PROJECT=${WANDB_PROJECT:-molcot-grpo}
# TRL's colocate mode starts a one-process torch.distributed group; give every GPU its own port so that
# several runs can start at the same moment on one machine.
export MASTER_ADDR=localhost
export MASTER_PORT=${MASTER_PORT:-$((29600 + 10 * ${GPU:-0}))}

exec python -u "$REPO/grpo/train_grpo.py" --variant "$START" --run-id "$RUN_ID" "$@"
