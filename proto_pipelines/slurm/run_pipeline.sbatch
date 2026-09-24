#!/bin/bash
#SBATCH --job-name=proto_pipeline
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --time=14:00:00
#SBATCH --partition=gpu,gpu_high_mem,preemptible
#SBATCH --requeue
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Generic Slurm wrapper. Takes the pipeline and config as arguments, so one
# template covers every workflow instead of one file per pipeline-config pair:
#
#   sbatch --export=ALL,PROTO_HOME=$HOME/proto_home,PYTHON=$(which python) \
#          proto_pipelines/slurm/run_pipeline.sbatch \
#          acr_sample proto_pipelines/configs/smoke/acr_sample_smoke.yaml
#
# The partition names above are one cluster's. Override without editing:
#   sbatch --partition=your_partition --exclude= ...
# On a cluster with small-GPU nodes, add --exclude=<node> for 7B+ models.
#
# The build itself is scripts/run_pipeline.sh, which needs no cluster.
set -euo pipefail
cd "${REPO_ROOT:-$(pwd)}"; mkdir -p logs
exec proto_pipelines/scripts/run_pipeline.sh "$@"
