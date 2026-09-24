#!/bin/bash
#SBATCH --job-name=acr_e2e
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=evo_gpu_priority,gpu,gpu_high_mem,gpu_batch,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/acr_e2e_%j.out
#SBATCH --error=logs/acr_e2e_%j.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"; mkdir -p logs
# Keep a clean env clean: ~/.local/lib/python3.11 would otherwise shadow
# the installed proto-language/proto-tools during dependency resolution.
export PYTHONNOUSERSITE=1
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="$PROTO_HOME/proto_model_cache"
export PROTO_DATABASES_DIR="$PROTO_MODEL_CACHE/databases"
export PROTO_ALPHAFOLD3_WEIGHTS_DIR="$PROTO_MODEL_CACHE/alphafold3"
compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin"* >/dev/null || { echo "ERROR: no AF3 weights" >&2; exit 1; }
"${PYTHON:?}" proto_pipelines/tests/test_parity.py
echo; echo "=== acr_sample end to end (with Acr evidence callers) ==="
"$PYTHON" -m proto_pipelines.pipelines.acr_sample --config proto_pipelines/configs/smoke/acr_sample_smoke.yaml
echo; echo "=== outputs ==="
ls -la "${PROTO_PIPELINES_OUTPUT_ROOT:-outputs}/smoke/acr_prescreen"/*.csv "${PROTO_PIPELINES_OUTPUT_ROOT:-outputs}/smoke/acr_prescreen"/*.json 2>/dev/null | awk "{print \"  \"\$9, \$5\"B\"}" || true
