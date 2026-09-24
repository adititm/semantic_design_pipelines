#!/bin/bash
#SBATCH --job-name=blastdb
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=cpu,cpu_batch,preemptible
#SBATCH --output=logs/blastdb_%j.out
#SBATCH --error=logs/blastdb_%j.err
#
# Thin wrapper. The build itself is scripts/build_blastdb.sh, which needs no
# cluster -- run that directly if you do not have Slurm. Partition names here
# are one cluster's; override with `sbatch --partition=...`.
set -euo pipefail
cd "${REPO_ROOT:-$(pwd)}"; mkdir -p logs
THREADS="${SLURM_CPUS_PER_TASK:-}" exec proto_pipelines/scripts/build_blastdb.sh
