#!/bin/bash
#SBATCH --job-name=proto_semdesign_smoke
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu,gpu_high_mem,preemptible
#SBATCH --requeue
#SBATCH --exclude=GPUCACE
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# End-to-end smoke test over all four workflows, one or two prompts
# each with two samples:
#
#   t2ta_sample        Evo -> Prodigal/QC -> HMM -> AF3 monomer -> cofold
#   acr_sample         Evo -> Prodigal/QC -> AlphaFold 3 monomer screen
#   gene_completion    Evo -> Prodigal/QC -> MAFFT identity (no folding)
#   operon_completion  Evo -> Prodigal/QC -> MAFFT identity (no folding)
#
# This proves the wiring runs. It is far too small to say anything about
# yield, and its AlphaFold 3 confidences are not a calibration.
#
# The first run on a fresh PROTO_HOME builds the evo1 tool environment, which
# compiles flash-attn from source and can take one to three hours. Later runs
# reuse it.
#
# Submit from the semantic_design directory:
#   sbatch proto_pipelines/slurm/smoke_test.sbatch

set -euo pipefail

SEMANTIC_DESIGN_DIR="${SEMANTIC_DESIGN_DIR:-$(pwd)}"
cd "$SEMANTIC_DESIGN_DIR"
mkdir -p logs

# Point at a specific proto-language checkout when the environment has more
# than one, or a stale editable install. Unset for a normal pip install.
export PROTO_LANGUAGE_ROOT="${PROTO_LANGUAGE_ROOT:-}"
export PROTO_HOME="${PROTO_HOME:?set PROTO_HOME to your proto storage root}"
export PYTHONPATH="$SEMANTIC_DESIGN_DIR:${PYTHONPATH:-}"

# Asset lookups have their own overrides that each outrank PROTO_HOME:
#   weights   PROTO_ALPHAFOLD3_WEIGHTS_DIR -> PROTO_MODEL_CACHE -> PROTO_HOME/...
#   databases PROTO_DATABASES_DIR -> PROTO_MODEL_CACHE/databases -> PROTO_HOME/...
# A shell profile that exports any of them (~/.bashrc commonly does) therefore
# silently redirects the lookup even when PROTO_HOME is set correctly here, and
# SLURM's --export=ALL carries it into the job. Re-derive all of them from this
# job's PROTO_HOME so one variable controls the whole run.
export PROTO_MODEL_CACHE="${PROTO_MODEL_CACHE_OVERRIDE:-$PROTO_HOME/proto_model_cache}"
export PROTO_DATABASES_DIR="${PROTO_DATABASES_DIR_OVERRIDE:-$PROTO_MODEL_CACHE/databases}"
export PROTO_ALPHAFOLD3_WEIGHTS_DIR="${PROTO_ALPHAFOLD3_WEIGHTS_DIR_OVERRIDE:-$PROTO_MODEL_CACHE/alphafold3}"

PYTHON="${PYTHON:-python}"

echo "=== environment ==="
echo "host           : $(hostname)"
echo "PROTO_HOME     : $PROTO_HOME"
echo "PROTO_MODEL_CACHE      : $PROTO_MODEL_CACHE"
echo "PROTO_DATABASES_DIR    : $PROTO_DATABASES_DIR"
echo "AF3 weights dir        : $PROTO_ALPHAFOLD3_WEIGHTS_DIR"
echo "checkout       : ${PROTO_LANGUAGE_ROOT:-<pip install>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Fail before spending GPU hours on generation if the screens cannot run.
# Evo sampling plus a cold evo1 environment build can take hours, and an
# AlphaFold 3 weights problem would otherwise only surface after all of it.
echo
echo "=== preflight ==="
preflight_failed=0

if compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin" > /dev/null \
   || compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin.zst" > /dev/null; then
    echo "AlphaFold 3 weights : OK ($(ls -1 "$PROTO_ALPHAFOLD3_WEIGHTS_DIR"/*.bin* | head -1))"
else
    echo "ERROR: no AlphaFold 3 weights (.bin / .bin.zst) in $PROTO_ALPHAFOLD3_WEIGHTS_DIR" >&2
    echo "       Point PROTO_ALPHAFOLD3_WEIGHTS_DIR_OVERRIDE at the directory holding af3.bin.zst." >&2
    preflight_failed=1
fi

# The cofold stage runs AlphaFold 3 with a local MMseqs2 MSA, which needs the
# uniref30 database indexed on disk. Checking the same file the tool checks
# (<db_prefix>.dbtype) rather than just the directory, so a dangling symlink or
# a partial download is caught here instead of after stage 1 has run.
MSA_DB_DIR="$PROTO_DATABASES_DIR/uniref30_2302"
if [ -f "$MSA_DB_DIR/uniref30_2302_db.dbtype" ]; then
    echo "MMseqs2 uniref30    : OK ($MSA_DB_DIR)"
else
    echo "ERROR: uniref30-2302 not provisioned at $MSA_DB_DIR" >&2
    echo "       Point PROTO_DATABASES_DIR_OVERRIDE at the databases root that holds it," >&2
    echo "       or set af3_use_msa: false in the cofold config to skip the MSA." >&2
    preflight_failed=1
fi

if [ "$preflight_failed" -ne 0 ]; then
    echo "ERROR: preflight failed; refusing to spend GPU time on generation." >&2
    exit 1
fi

echo
echo "=== CPU parity checks ==="
"$PYTHON" proto_pipelines/tests/test_parity.py

echo
echo "=== stage 1: t2ta_sample ==="
"$PYTHON" -m proto_pipelines.pipelines.t2ta_sample \
    --config proto_pipelines/configs/smoke/t2ta_sample_smoke.yaml

echo
echo "=== stage 2: acr_sample ==="
"$PYTHON" -m proto_pipelines.pipelines.acr_sample \
    --config proto_pipelines/configs/smoke/acr_sample_smoke.yaml

echo
echo "=== stage 3: gene_completion ==="
"$PYTHON" -m proto_pipelines.pipelines.gene_completion \
    --config proto_pipelines/configs/smoke/gene_completion_smoke.yaml

echo
echo "=== stage 4: operon_completion ==="
"$PYTHON" -m proto_pipelines.pipelines.operon_completion \
    --config proto_pipelines/configs/smoke/operon_completion_smoke.yaml

echo
echo "=== outputs ==="
find proto_pipelines/outputs/smoke -maxdepth 2 -type f -name '*.csv' -print -exec head -3 {} \;
