#!/bin/bash
#SBATCH --job-name=acrnet_heldout
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=evo_gpu_priority,gpu,gpu_high_mem,gpu_batch,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/acrnet_heldout_%j.out
#SBATCH --error=logs/acrnet_heldout_%j.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"; mkdir -p logs
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
"${PYTHON:?}" -m proto_pipelines.calibration.score_acrnet \
    --sequences proto_pipelines/calibration/data/acr/heldout_sequences.csv \
    --predict-property "$SEMANTIC_DESIGN_DIR/.scratch_acr/Predict_Property" \
    --psiblast "$SEMANTIC_DESIGN_DIR/.scratch_acr/blast/ncbi-blast-2.17.0+/bin/psiblast" \
    --blast-db "$SEMANTIC_DESIGN_DIR/.scratch_acr/blast/db/uniref30" \
    --checkpoint proto_pipelines/calibration/data/acr/acrnet/model.ckpt \
    --workers 30 --device cuda \
    --dump-features proto_pipelines/calibration/results/acr/acrnet_features_heldout.pkl \
    --out proto_pipelines/calibration/results/acr/acrnet_scores_heldout_regime.csv
