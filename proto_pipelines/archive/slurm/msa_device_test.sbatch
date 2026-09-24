#!/bin/bash
#SBATCH --job-name=msa_gpu_test
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu,gpu_high_mem,preemptible
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/msatest_%j.out
#SBATCH --error=logs/msatest_%j.err
set -euo pipefail
cd "${SEMANTIC_DESIGN_DIR:?}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="$PROTO_HOME/proto_model_cache"
export PROTO_DATABASES_DIR="$PROTO_MODEL_CACHE/databases"
export PROTO_ALPHAFOLD3_WEIGHTS_DIR="$PROTO_MODEL_CACHE/alphafold3"

# MMseqs2-GPU needs the padded index, not just the base DB.
IDX="$PROTO_DATABASES_DIR/uniref30_2302/uniref30_2302_db.idx_pad"
if [ ! -f "${IDX}.dbtype" ]; then
    echo "ERROR: MMseqs2-GPU needs ${IDX}.dbtype; not present." >&2; exit 1
fi
echo "GPU-padded index present: ${IDX}.dbtype"

echo "=== timing ONE pair with --msa-device cuda ==="
start=$(date +%s)
"$PYTHON" -m proto_pipelines.calibration.run_folds \
    --pairs proto_pipelines/calibration/data/gpu_msa_test.csv \
    --out-dir proto_pipelines/calibration/results/gpu_msa_test \
    --shard 0 --num-shards 1 --msa-device cuda
echo "ELAPSED_SECONDS=$(( $(date +%s) - start ))"
