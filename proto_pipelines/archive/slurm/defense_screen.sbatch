#!/bin/bash
#SBATCH --job-name=df_screen
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --partition=cpu,cpu_batch,preemptible
#SBATCH --output=logs/df_screen_%j.out
#SBATCH --error=logs/df_screen_%j.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
"${PYTHON:?}" .scratch_acr/screen_defense.py
