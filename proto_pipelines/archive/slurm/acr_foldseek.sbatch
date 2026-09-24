#!/bin/bash
#SBATCH --job-name=acr_fseek
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu,cpu_batch,preemptible
#SBATCH --requeue
#SBATCH --output=logs/acr_fseek_%A_%a.out
#SBATCH --error=logs/acr_fseek_%A_%a.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
"${PYTHON:?}" -m proto_pipelines.calibration.score_acr_foldseek \
  --chains proto_pipelines/calibration/data/acr/acr_monomer_chains.csv \
  --structures proto_pipelines/calibration/results/acr/folds \
  --ref-dir proto_pipelines/calibration/data/acr/ref_acr_chains \
  --out proto_pipelines/calibration/results/acr/foldseek_shard${SLURM_ARRAY_TASK_ID}.csv \
  --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 8
