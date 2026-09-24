#!/bin/bash
#SBATCH --job-name=acr_esm1b
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=evo_gpu_priority,gpu,gpu_high_mem,gpu_batch,preemptible
#SBATCH --requeue
#SBATCH --output=logs/acr_esm1b_%j.out
#SBATCH --error=logs/acr_esm1b_%j.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export TORCH_HOME="${PROTO_HOME:?}/torch_home"
"${PYTHON:?}" .scratch_acr/embed_esm1b_div.py
