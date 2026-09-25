#!/usr/bin/env bash
# Run one pipeline, anywhere.
#
# Works unchanged on a workstation, under SLURM, or against connected compute
# -- the only difference is the `device` key in your config (see
# "Execution modes" in the README). Nothing here is cluster-specific.
#
#   utils/run_pipeline.sh acr_sample configs/smoke/acr_sample_smoke.yaml
#
# Environment:
#   PROTO_HOME    required. Asset root you own; tool envs are built under it.
#   PYTHON        interpreter to use (default: python)
#   OUTPUT_ROOT   where run outputs land (default: <repo>/outputs)
#   SKIP_CHECKS   set to 1 to skip the parity suite
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <pipeline> <config>" >&2
    echo "  pipeline: acr_sample | t2ta_sample | gene_completion | operon_completion" >&2
    exit 2
fi
PIPELINE="$1"
CONFIG="$2"

# Resolve the repo root from this script's location, so the script works from
# any working directory and needs no SEMANTIC_DESIGN_DIR-style variable.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

: "${PROTO_HOME:?set PROTO_HOME to an asset root you own (see README Setup)}"
# Derive all of these FROM PROTO_HOME rather than honouring whatever is
# already exported. Each one outranks PROTO_HOME inside proto-tools, shell
# profiles commonly set them, and SLURM's --export=ALL carries them into a
# job -- so an inherited value silently redirects assets to another root even
# when PROTO_HOME is correct. Anything inherited and inconsistent is reported
# and replaced, not obeyed.
for _var in PROTO_MODEL_CACHE PROTO_DATABASES_DIR PROTO_ALPHAFOLD3_WEIGHTS_DIR; do
    _inherited="${!_var:-}"
    case "$_var" in
        PROTO_MODEL_CACHE)            _derived="$PROTO_HOME/proto_model_cache" ;;
        PROTO_DATABASES_DIR)          _derived="$PROTO_HOME/proto_model_cache/databases" ;;
        PROTO_ALPHAFOLD3_WEIGHTS_DIR) _derived="$PROTO_HOME/proto_model_cache/alphafold3" ;;
    esac
    if [ -n "$_inherited" ] && [ "$_inherited" != "$_derived" ]; then
        echo "WARNING: $_var was inherited as $_inherited, which does not match" >&2
        echo "         PROTO_HOME=$PROTO_HOME. Using $_derived instead." >&2
        echo "         Export ${_var}_FORCE to keep the inherited value." >&2
    fi
    _force="${_var}_FORCE"
    if [ -n "${!_force:-}" ]; then
        export "$_var=${!_force}"
    else
        export "$_var=$_derived"
    fi
done
unset _var _inherited _derived _force
[ -n "${OUTPUT_ROOT:-}" ] && export PROTO_PIPELINES_OUTPUT_ROOT="$OUTPUT_ROOT"

# Keep a clean environment clean: a user site-packages directory would
# otherwise shadow the installed proto-language during resolution.
export PYTHONNOUSERSITE=1

echo "=== run ==="
echo "  host      : $(hostname)"
echo "  repo      : $REPO_ROOT"
echo "  pipeline  : $PIPELINE"
echo "  config    : $CONFIG"
echo "  PROTO_HOME: $PROTO_HOME"
echo "  device    : $(grep -m1 '^device:' "$CONFIG" 2>/dev/null || echo '(unset in config)')"

# Only the folding pipelines need AF3 weights, and only when running locally.
# A remote device fetches them on the worker, so absence is not an error then.
DEVICE="$(sed -n 's/^device:[[:space:]]*//p' "$CONFIG" | head -1 | tr -d '"'"'"' ')"
NEEDS_AF3=0
case "$PIPELINE" in acr_sample|t2ta_sample) NEEDS_AF3=1 ;; esac
if [ "$NEEDS_AF3" = 1 ] && [ "$DEVICE" != "proto" ] && [ "$DEVICE" != "modal" ]; then
    if ! compgen -G "$PROTO_ALPHAFOLD3_WEIGHTS_DIR/*.bin"* >/dev/null; then
        echo "ERROR: no AlphaFold 3 weights under $PROTO_ALPHAFOLD3_WEIGHTS_DIR." >&2
        echo "       $PIPELINE folds structures. Either provide the weights, or" >&2
        echo "       set 'device: proto' / 'device: modal' in $CONFIG to fold remotely." >&2
        exit 1
    fi
fi

# Fail here, not 30 lines into proto-tools' device manager, when the config
# asks for a GPU that this host does not have. The common cases are running
# on a login node and forgetting to request a GPU from the scheduler.
case "$DEVICE" in
    cuda*)
        # nvidia-smi -L prints "No devices found." and exits 0 when there are
        # none, so count real entries rather than testing output emptiness.
        if [ "$(nvidia-smi -L 2>/dev/null | grep -c '^GPU')" -eq 0 ]; then
            echo "ERROR: $CONFIG asks for device='$DEVICE' but no GPU is visible on $(hostname)." >&2
            echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)}" >&2
            echo "       Options:" >&2
            echo "         - request a GPU from your scheduler (sbatch --gpus=1 ...)" >&2
            echo "         - set 'device: modal' or 'device: proto' to use connected compute" >&2
            echo "         - set 'device: cpu' (works, but Evo generation will be very slow)" >&2
            exit 1
        fi
        ;;
esac

if [ "${SKIP_CHECKS:-0}" != "1" ]; then
    echo; echo "=== parity checks (CPU, no weights needed) ==="
    "$PYTHON" semantic_design_pipelines/tests/test_parity.py
fi

echo; echo "=== $PIPELINE ==="
exec "$PYTHON" -m "semantic_design_pipelines.pipelines.$PIPELINE" --config "$CONFIG"
