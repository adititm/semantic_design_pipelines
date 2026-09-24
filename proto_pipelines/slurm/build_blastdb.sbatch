#!/bin/bash
#SBATCH --job-name=blastdb
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --partition=cpu,cpu_batch,preemptible
#SBATCH --output=logs/blastdb_%j.out
#SBATCH --error=logs/blastdb_%j.err
set -euo pipefail
cd "${REPO_ROOT:-${SEMANTIC_DESIGN_DIR:?}}"
# Both are resolved from the asset roots rather than hard-coded; override
# either if your layout differs.
MM="${MMSEQS_BIN:-$PROTO_HOME/proto_tool_envs/colabfold_search_env/bin/mmseqs}"
DB="${UNIREF30_DB:-${PROTO_DATABASES_DIR:-$PROTO_HOME/proto_model_cache/databases}/uniref30_2302/uniref30_2302_db}"
[ -x "$MM" ] || { echo "ERROR: no mmseqs at $MM; set MMSEQS_BIN" >&2; exit 1; }
[ -f "$DB.dbtype" ] || { echo "ERROR: no uniref30 at $DB; set UNIREF30_DB" >&2; exit 1; }
W=.scratch_acr/blast
echo "converting uniref30 to FASTA..."
"$MM" convert2fasta "$DB" "$W/uniref30.faa"
ls -la "$W/uniref30.faa"
echo "building BLAST database..."
"$W/ncbi-blast-2.17.0+/bin/makeblastdb" -in "$W/uniref30.faa" -dbtype prot -out "$W/db/uniref30" -title uniref30
echo BLASTDB_DONE
