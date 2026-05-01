#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_REPORT="${OUTPUT_REPORT:-reports/hetero_gnn_full.md}"
PLOTS_DIR="${PLOTS_DIR:-reports/plots/hetero_gnn_full}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-reports/artifacts/hetero_gnn_full}"
LOG_FILE="${LOG_FILE:-run.txt}"

EPOCHS="${EPOCHS:-120}"
PATIENCE="${PATIENCE:-20}"
EVAL_EVERY="${EVAL_EVERY:-2}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
NUM_HEADS="${NUM_HEADS:-4}"
NUM_GLOBAL_TOKENS="${NUM_GLOBAL_TOKENS:-8}"
KNN_K="${KNN_K:-20}"
MIN_SIMILARITY="${MIN_SIMILARITY:-0.5}"
MIN_SHARED_HASHTAGS="${MIN_SHARED_HASHTAGS:-2}"
MIN_SHARED_URLS="${MIN_SHARED_URLS:-1}"
MAX_USERS_PER_TOKEN="${MAX_USERS_PER_TOKEN:-200}"

mkdir -p "$(dirname "$OUTPUT_REPORT")" "$PLOTS_DIR" "$ARTIFACTS_DIR"

echo "Running full hetero-GNN training on GPU ${GPU_ID}"
echo "Report: ${OUTPUT_REPORT}"
echo "Plots: ${PLOTS_DIR}"
echo "Artifacts: ${ARTIFACTS_DIR}"
echo "Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/train_hetero_gnn.py \
  --epochs "${EPOCHS}" \
  --patience "${PATIENCE}" \
  --eval-every "${EVAL_EVERY}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --num-heads "${NUM_HEADS}" \
  --num-global-tokens "${NUM_GLOBAL_TOKENS}" \
  --knn-k "${KNN_K}" \
  --min-similarity "${MIN_SIMILARITY}" \
  --min-shared-hashtags "${MIN_SHARED_HASHTAGS}" \
  --min-shared-urls "${MIN_SHARED_URLS}" \
  --max-users-per-token "${MAX_USERS_PER_TOKEN}" \
  --output "${OUTPUT_REPORT}" \
  --plots-dir "${PLOTS_DIR}" \
  --artifacts-dir "${ARTIFACTS_DIR}" \
  "$@" | tee "${LOG_FILE}"
