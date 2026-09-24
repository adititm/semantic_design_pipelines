#!/bin/bash
#SBATCH --job-name=acrnet_folds
#SBATCH --array=0-1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --partition=evo_gpu_priority,gpu,gpu_high_mem,gpu_batch,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/acrnet_folds_%A_%a.out
#SBATCH --error=logs/acrnet_folds_%A_%a.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"; mkdir -p logs
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="$PROTO_HOME/proto_model_cache"
export PROTO_DATABASES_DIR="$PROTO_MODEL_CACHE/databases"
export PROTO_ALPHAFOLD3_WEIGHTS_DIR="$PROTO_MODEL_CACHE/alphafold3"
PYTHON="${PYTHON:-python}"
compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin"* >/dev/null || { echo "ERROR: no AF3 weights" >&2; exit 1; }
# Single-sequence: this mirrors the monomer screen, and an MSA for 316 chains
# would dominate wall time without changing what Foldseek needs (a fold).
"$PYTHON" -m proto_pipelines.calibration.run_monomer_folds \
    --chains proto_pipelines/calibration/data/acr/acrnet_testset_chains.csv \
    --out-dir proto_pipelines/calibration/results/acr/acrnet_folds \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 2 \
    --num-recycles 3 --num-diffusion-samples 1
