#!/bin/bash
#SBATCH --job-name=proto_mono_calib
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --partition=evo_gpu_priority,gpu_batch,gpu_high_mem,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --array=0-3
#SBATCH --output=logs/mono_%A_%a.out
#SBATCH --error=logs/mono_%A_%a.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"
mkdir -p logs
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="$PROTO_HOME/proto_model_cache"
export PROTO_DATABASES_DIR="$PROTO_MODEL_CACHE/databases"
export PROTO_ALPHAFOLD3_WEIGHTS_DIR="$PROTO_MODEL_CACHE/alphafold3"
PYTHON="${PYTHON:-python}"
compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin"* > /dev/null || { echo "ERROR: no AF3 weights" >&2; exit 1; }
# USE_MSA=1 runs the comparison arm (NOT how the screen runs in production).
MSA_FLAG=""
if [ "${USE_MSA:-0}" = "1" ]; then
    [ -f "$PROTO_DATABASES_DIR/uniref30_2302/uniref30_2302_db.idx_pad.dbtype" ] || \
        { echo "ERROR: MSA mode needs the GPU-padded uniref30 index" >&2; exit 1; }
    MSA_FLAG="--use-msa"
fi
echo "preflight OK (msa=${USE_MSA:-0})"
"$PYTHON" -m proto_pipelines.calibration.run_monomer_folds \
    --chains "${CHAINS:?}" --out-dir "${OUTDIR:?}" \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards "${NUM_SHARDS:-4}" $MSA_FLAG
