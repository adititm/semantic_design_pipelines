#!/bin/bash
#SBATCH --job-name=t2ta_evo2
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=evo_gpu_priority,gpu,gpu_high_mem,gpu_batch,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/t2ta_evo2_%j.out
#SBATCH --error=logs/t2ta_evo2_%j.err
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
[ -f "$PROTO_DATABASES_DIR/uniref30_2302/uniref30_2302_db.idx_pad.dbtype" ] || { echo "ERROR: no uniref30 GPU index" >&2; exit 1; }
[ -f ./proto_pipelines/calibration/data/ta_families.hmm ] || { echo "ERROR: no TA HMM" >&2; exit 1; }
echo "preflight OK (AF3 weights, uniref30 GPU index, TA HMM)"
"$PYTHON" proto_pipelines/tests/test_parity.py
echo; echo "=== Evo 2 generator path, end to end ==="
"$PYTHON" -m proto_pipelines.pipelines.t2ta_sample --config proto_pipelines/configs/smoke/t2ta_evo2_smoke.yaml
echo; echo "=== outputs written ==="
ls -la "outputs/smoke/t2ta_evo2_final"/*.csv "outputs/smoke/t2ta_evo2_final"/*.json "outputs/smoke/t2ta_evo2_final"/*.fasta 2>/dev/null | awk '{print "  "$9, $5"B"}' || true
