#!/usr/bin/env bash
# Build the UniRef30 BLAST database AcrNET's PSSM block needs.
#
# No cluster required — this is plain mmseqs + makeblastdb. It is CPU-only but
# heavy: ~16 GB of output and a few hours. Run it once.
#
#   scripts/build_blastdb.sh
#
# Then set these in your Acr config:
#   acrnet_psiblast: ./.scratch_acr/blast/ncbi-blast-2.17.0+/bin/psiblast
#   acrnet_blast_db: ./.scratch_acr/blast/db/uniref30
#
# AcrNET runs WITHOUT this. A missing PSSM is a supported state -- the score is
# read against the weaker `no_pssm` calibration regime rather than being wrong
# (see docs/ACR_PIPELINE.md). Build it if you can; skip it if you cannot.
#
# Environment:
#   PROTO_HOME    required, to locate the mmseqs binary and UniRef30 index
#   MMSEQS_BIN    override the mmseqs executable
#   UNIREF30_DB   override the mmseqs UniRef30 database prefix
#   THREADS       makeblastdb/mmseqs threads (default: nproc)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

: "${PROTO_HOME:?set PROTO_HOME (see README Setup)}"
MM="${MMSEQS_BIN:-$PROTO_HOME/proto_tool_envs/colabfold_search_env/bin/mmseqs}"
DB="${UNIREF30_DB:-${PROTO_DATABASES_DIR:-$PROTO_HOME/proto_model_cache/databases}/uniref30_2302/uniref30_2302_db}"
THREADS="${THREADS:-$(nproc)}"
W=.scratch_acr/blast

[ -x "$MM" ] || { echo "ERROR: no mmseqs at $MM; set MMSEQS_BIN" >&2; exit 1; }
[ -f "$DB.dbtype" ] || { echo "ERROR: no UniRef30 mmseqs index at $DB; set UNIREF30_DB" >&2; exit 1; }

mkdir -p "$W/db"
if [ ! -x "$W/ncbi-blast-2.17.0+/bin/makeblastdb" ]; then
    echo "=== fetching BLAST+ ==="
    curl -fsSL -o "$W/blast.tar.gz" \
      "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.17.0/ncbi-blast-2.17.0+-x64-linux.tar.gz"
    tar -xzf "$W/blast.tar.gz" -C "$W"
    rm -f "$W/blast.tar.gz"
fi

if [ ! -s "$W/uniref30.faa" ]; then
    echo "=== converting UniRef30 to FASTA (this is the slow part) ==="
    "$MM" convert2fasta "$DB" "$W/uniref30.faa"
fi
ls -la "$W/uniref30.faa"

echo "=== building BLAST database ==="
"$W/ncbi-blast-2.17.0+/bin/makeblastdb" \
    -in "$W/uniref30.faa" -dbtype prot -out "$W/db/uniref30" -title uniref30

echo "=== verifying ==="
"$W/ncbi-blast-2.17.0+/bin/blastdbcmd" -db "$W/db/uniref30" -info
echo "BLASTDB_DONE"
