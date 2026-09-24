#!/bin/bash
#SBATCH --job-name=proto_ta_calib
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu_batch,preemptible,gpu_high_mem
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --array=0-7
#SBATCH --output=logs/calib_%A_%a.out
#SBATCH --error=logs/calib_%A_%a.err

# Score the labelled TA pair set with AlphaFold 3 + pDockQ2, one array task
# per shard. Each shard writes incrementally and skips pairs already scored,
# so a preempted task resumes instead of restarting.
#
#   sbatch --export=ALL,PAIRS=...,OUTDIR=... proto_pipelines/slurm/calibration_folds.sbatch

set -euo pipefail

SEMANTIC_DESIGN_DIR="${SEMANTIC_DESIGN_DIR:-$(pwd)}"
cd "$SEMANTIC_DESIGN_DIR"
mkdir -p logs

export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?set PROTO_HOME}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="${PROTO_MODEL_CACHE_OVERRIDE:-$PROTO_HOME/proto_model_cache}"
export PROTO_DATABASES_DIR="${PROTO_DATABASES_DIR_OVERRIDE:-$PROTO_MODEL_CACHE/databases}"
export PROTO_ALPHAFOLD3_WEIGHTS_DIR="${PROTO_ALPHAFOLD3_WEIGHTS_DIR_OVERRIDE:-$PROTO_MODEL_CACHE/alphafold3}"

PAIRS="${PAIRS:?set PAIRS to the labelled pairs CSV}"
OUTDIR="${OUTDIR:?set OUTDIR}"
NUM_SHARDS="${NUM_SHARDS:-8}"
# "cuda" engages MMseqs2-GPU (~2.2x faster end-to-end, verified to give the
# same scores as the CPU path on an identical pair). Needs the .idx_pad index.
MSA_DEVICE="${MSA_DEVICE:-cpu}"
PYTHON="${PYTHON:-python}"

echo "shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} on $(hostname)"
echo "pairs: $PAIRS -> $OUTDIR"

# Same preflight as the smoke test: fail in seconds, not after hours.
if ! compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin"* > /dev/null; then
    echo "ERROR: no AlphaFold 3 weights in $PROTO_ALPHAFOLD3_WEIGHTS_DIR" >&2; exit 1
fi
if [ ! -f "$PROTO_DATABASES_DIR/uniref30_2302/uniref30_2302_db.dbtype" ]; then
    echo "ERROR: uniref30-2302 not provisioned at $PROTO_DATABASES_DIR" >&2; exit 1
fi
if [ "$MSA_DEVICE" != "cpu" ] && \
   [ ! -f "$PROTO_DATABASES_DIR/uniref30_2302/uniref30_2302_db.idx_pad.dbtype" ]; then
    echo "ERROR: MSA_DEVICE=$MSA_DEVICE needs the GPU-padded index" >&2
    echo "       (uniref30_2302_db.idx_pad.dbtype); use MSA_DEVICE=cpu instead." >&2
    exit 1
fi
echo "preflight OK (msa_device=$MSA_DEVICE)"

"$PYTHON" -m proto_pipelines.calibration.run_folds \
    --pairs "$PAIRS" \
    --out-dir "$OUTDIR" \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --num-shards "$NUM_SHARDS" \
    --msa-device "$MSA_DEVICE"
