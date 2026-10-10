#!/bin/bash
# One SFT run (one base model x one CoT variant x one seed) with the settings of the paper.
#
#   bash sft/run_sft.sh <experiment> <model> <variant> <run_id> [extra train_sft.py arguments]
#
#   experiment  families : five base models of comparable size, run_id 0-9 in the paper
#               scales   : six Qwen3.5 sizes, run_id 0-2 in the paper
#   model       families : qwen3-4b | ministral-3b | llama-3.2-3b | gemma-4-e4b | qwen3.5-4b
#               scales   : qwen3.5-0.8b | qwen3.5-2b | qwen3.5-4b | qwen3.5-9b | qwen3.5-27b | qwen3.5-35b-a3b
#   variant     original | curated   (the No-CoT baseline uses --data-dir from scripts/make_nocot_data.py and variant original)
#   run_id      seed = 1000 + run_id (training and data order), evaluation seed = 2000 + run_id
#
# Environment: GPU (default 0), DATA_DIR (default <repo>/data), OUTPUT_DIR (default <repo>/runs),
#              NO_WANDB=1 to switch off Weights & Biases logging (otherwise WANDB_PROJECT / WANDB_ENTITY are used).
# Example:     GPU=0 bash sft/run_sft.sh scales qwen3.5-4b curated 0
# Output:      $OUTPUT_DIR/<model slug>/<variant>/run_<id>/ with test_predictions_epoch_000..004.json,
#              test_predictions.json (checkpoint with the lowest validation loss), model/ (LoRA checkpoints, the
#              epoch-4 adapter is model/checkpoint-<last step>), run_config.json, runtime_metrics.json.
set -eo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT=${1:?experiment: families | scales}; MODEL_KEY=${2:?model}; VARIANT=${3:?variant: original | curated}; RUN_ID=${4:?run id}
shift 4

export CUDA_VISIBLE_DEVICES=${GPU:-0}                       # train_sft.py uses one GPU (cuda:0)
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export WANDB_PROJECT=${WANDB_PROJECT:-molcot-sft-$EXPERIMENT}
DATA_DIR=${DATA_DIR:-$REPO/data}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/runs}

# Common recipe (both experiments): LoRA r=16, alpha=32, dropout 0.05 on all linear layers, AdamW (fused), lr 2e-4,
# cosine schedule with 3% warmup, gradient clipping 0.3, 4 epochs, effective batch 16, max length 1280 tokens;
# evaluation: greedy + 16 samples (T=1.0, top-p 0.9, top-k 50), at most 768 new tokens.
EPOCHS=4; MAX_SEQ_LENGTH=1280; MAX_NEW_TOKENS=768; LR=2e-4; BEST_OF_N=16
PRECISION=bf16

case "$EXPERIMENT:$MODEL_KEY" in
  # Model families. Micro-batch 4 x accumulation 4 = 16; Qwen3.5 uses its chat template, the others plain text.
  families:qwen3-4b)        MODEL=Qwen/Qwen3-4B-Base;                SLUG=qwen3-4b-base;            FORMAT=plain; BS=4; GA=4; EBS=8 ;;
  families:ministral-3b)    MODEL=mistralai/Ministral-3-3B-Base-2512; SLUG=ministral-3-3b-base-2512; FORMAT=plain; BS=4; GA=4; EBS=8 ;;
  families:llama-3.2-3b)    MODEL=meta-llama/Llama-3.2-3B;           SLUG=llama-3.2-3b;             FORMAT=plain; BS=4; GA=4; EBS=8 ;;
  families:gemma-4-e4b)     MODEL=google/gemma-4-E4B;                SLUG=gemma-4-e4b;              FORMAT=plain; BS=4; GA=4; EBS=8 ;;
  families:qwen3.5-4b)      MODEL=Qwen/Qwen3.5-4B-Base;              SLUG=qwen3.5-4b-base;          FORMAT=chat;  BS=4; GA=4; EBS=8 ;;
  # Model scales (all Qwen3.5, chat template). The effective batch stays 16; larger micro-batches only fill the GPU.
  scales:qwen3.5-0.8b)      MODEL=Qwen/Qwen3.5-0.8B-Base;            SLUG=qwen3.5-0.8b-base;        FORMAT=chat;  BS=16; GA=1; EBS=32 ;;
  scales:qwen3.5-2b)        MODEL=Qwen/Qwen3.5-2B-Base;              SLUG=qwen3.5-2b-base;          FORMAT=chat;  BS=16; GA=1; EBS=32 ;;
  scales:qwen3.5-4b)        MODEL=Qwen/Qwen3.5-4B-Base;              SLUG=qwen3.5-4b-base;          FORMAT=chat;  BS=16; GA=1; EBS=16 ;;
  scales:qwen3.5-9b)        MODEL=Qwen/Qwen3.5-9B-Base;              SLUG=qwen3.5-9b-base;          FORMAT=chat;  BS=16; GA=1; EBS=16 ;;
  scales:qwen3.5-27b)       MODEL=Qwen/Qwen3.5-27B;                  SLUG=qwen3.5-27b-posttrained-nf4; FORMAT=chat; BS=8; GA=2; EBS=16; PRECISION=nf4 ;;
  scales:qwen3.5-35b-a3b)   MODEL=Qwen/Qwen3.5-35B-A3B;              SLUG=qwen3.5-35b-a3b-posttrained; FORMAT=chat; BS=16; GA=1; EBS=16 ;;
  *) echo "unknown experiment/model: $EXPERIMENT $MODEL_KEY" >&2; exit 2 ;;
esac
case "$VARIANT" in original|curated) ;; *) echo "variant must be original or curated" >&2; exit 2 ;; esac
GEN_BS=$((BEST_OF_N * EBS))                                  # all 16 samples of an evaluation batch in one generate call

WANDB_ARGS=()
[[ "${NO_WANDB:-0}" == 1 ]] && WANDB_ARGS+=(--no-wandb)

ARGS=(
  --variant "$VARIANT" --run-id "$RUN_ID" --seed $((1000 + RUN_ID)) --eval-seed $((2000 + RUN_ID))
  --model "$MODEL" --experiment-name "$SLUG" --format-style "$FORMAT"
  --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS" --max-seq-length "$MAX_SEQ_LENGTH" --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BS" --grad-accum "$GA" --eval-batch-size "$EBS" --gen-batch-size "$GEN_BS"
  --lr "$LR" --best-of-n "$BEST_OF_N" --lora-rank 16 --lora-dropout 0.05
  "${WANDB_ARGS[@]}"
)
echo "experiment=$EXPERIMENT model=$MODEL variant=$VARIANT run_id=$RUN_ID precision=$PRECISION gpu=$CUDA_VISIBLE_DEVICES"
cd "$REPO/sft"
if [[ "$PRECISION" == nf4 ]]; then
  exec python -u train_sft_nf4.py "${ARGS[@]}" "$@"           # Qwen3.5-27B: 4-bit NF4 base weights, BF16 compute
fi
exec python -u train_sft.py "${ARGS[@]}" "$@"
