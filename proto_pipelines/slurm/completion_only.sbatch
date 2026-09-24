#!/bin/bash
#SBATCH --job-name=proto_semdesign_completion
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --partition=gpu,gpu_high_mem,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/completion_%j.out
#SBATCH --error=logs/completion_%j.err

# Gene- and operon-completion smoke test only. Neither pipeline folds, so this
# needs no AlphaFold 3 weights and no MSA database -- just Evo, Prodigal and
# MAFFT. Useful for iterating on the identity path without paying for a fold.
#
# Submit from the semantic_design directory:
#   sbatch proto_pipelines/slurm/completion_only.sbatch

set -euo pipefail

SEMANTIC_DESIGN_DIR="${SEMANTIC_DESIGN_DIR:-$(pwd)}"
cd "$SEMANTIC_DESIGN_DIR"
mkdir -p logs

export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?set PROTO_HOME to your proto storage root}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"
export PROTO_MODEL_CACHE="${PROTO_MODEL_CACHE_OVERRIDE:-$PROTO_HOME/proto_model_cache}"
export PROTO_DATABASES_DIR="${PROTO_DATABASES_DIR_OVERRIDE:-$PROTO_MODEL_CACHE/databases}"

PYTHON="${PYTHON:-python}"

echo "=== environment ==="
echo "host     : $(hostname)"
echo "PROTO_HOME : $PROTO_HOME"
echo "checkout : ${PROTO_LANGUAGE_ROOT:-<pip install>}"

echo
echo "=== CPU parity checks ==="
"$PYTHON" proto_pipelines/tests/test_parity.py

echo
echo "=== gene_completion ==="
"$PYTHON" -m proto_pipelines.pipelines.gene_completion \
    --config proto_pipelines/configs/smoke/gene_completion_smoke.yaml

echo
echo "=== operon_completion ==="
"$PYTHON" -m proto_pipelines.pipelines.operon_completion \
    --config proto_pipelines/configs/smoke/operon_completion_smoke.yaml

echo
echo "=== outputs ==="
for f in proto_pipelines/outputs/smoke/gene_completion/identity_results.csv \
         proto_pipelines/outputs/smoke/gene_completion/summary_statistics.csv \
         proto_pipelines/outputs/smoke/operon_completion/identity_results.csv \
         proto_pipelines/outputs/smoke/operon_completion/summary_statistics.csv; do
    echo "--- $f"; head -5 "$f" 2>/dev/null || echo "(missing)"
done
