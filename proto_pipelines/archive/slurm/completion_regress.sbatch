#!/bin/bash
#SBATCH --job-name=completion_regress
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=evo_gpu_priority,gpu,gpu_high_mem,gpu_batch,preemptible
#SBATCH --requeue
#SBATCH --output=logs/completion_regress_%j.out
#SBATCH --error=logs/completion_regress_%j.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"; mkdir -p logs
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="$PROTO_HOME/proto_model_cache"
export PROTO_DATABASES_DIR="$PROTO_MODEL_CACHE/databases"
for p in gene_completion operon_completion; do
  echo; echo "=== $p ==="
  "${PYTHON:?}" -m proto_pipelines.pipelines.$p --config proto_pipelines/configs/smoke/${p}_smoke.yaml
done
