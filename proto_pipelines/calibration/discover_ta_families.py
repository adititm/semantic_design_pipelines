"""Scan known type II TA proteins against full Pfam to derive the family list.

A curated list of TA-sounding family names does not work on this data: all
four experimentally validated EvoAT antitoxins have ``DUF6290`` -- a domain of
unknown function -- as their top Pfam hit. So the families are discovered
rather than named, by scanning proteins already known to be type II TA.

Sources are kept separate in the output so the list can be rebuilt from
naturals alone. Including the designs makes the list partly self-confirming
for those designs, which matters if you want to use them as a held-out test.

Scans are batched: ``SequenceHit.query_idx`` indexes the input sequence, so
hit-to-protein mapping survives batching and the 1.6 GB Pfam load is paid
once per chunk rather than once per protein.

Usage:
    python -m proto_pipelines.calibration.discover_ta_families \
        --pfam /path/Pfam-A.hmm --out ta_family_discovery_full.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

from proto_tools import PyHmmscanConfig, PyHmmscanInput
from proto_tools.tools.gene_annotation.pyhmmer.hmmscan import run_pyhmmer_hmmscan

logger = logging.getLogger(__name__)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN, MAX_LEN = 50, 300


def read_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into ``{header: sequence}``."""
    records: dict[str, str] = {}
    name: str | None = None
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            name = line[1:].strip()
        elif name:
            records[name] = records.get(name, "") + line.strip()
    return records


def collect(toxins: Path, antitoxins: Path, designs: Path | None) -> dict[str, tuple[str, str]]:
    """Gather every query, tagged by source.

    Args:
        toxins: TADB ``type_II_T_exp.fas``.
        antitoxins: TADB ``type_II_AT_exp.fas``.
        designs: Optional FASTA of characterised designs to include.

    Returns:
        ``{query_id: (source, sequence)}``.
    """
    out: dict[str, tuple[str, str]] = {}
    for path, source in ((toxins, "natural_toxin"), (antitoxins, "natural_antitoxin")):
        for header, sequence in read_fasta(path).items():
            if set(sequence) <= STANDARD_AA and MIN_LEN <= len(sequence) <= MAX_LEN:
                out[f"{source}::{header.split()[0]}"] = (source, sequence)
    if designs is not None and designs.exists():
        for header, sequence in read_fasta(designs).items():
            if set(sequence) <= STANDARD_AA and MIN_LEN <= len(sequence) <= MAX_LEN:
                out[f"design::{header.split()[0]}"] = ("design", sequence)
    return out


def scan(queries: dict[str, tuple[str, str]], pfam: Path, evalue: float,
         chunk: int) -> dict[str, list[dict[str, object]]]:
    """Batch-scan every query against the profile database.

    Args:
        queries: Output of :func:`collect`.
        pfam: Full Pfam-A.hmm.
        evalue: Sequence-level E-value cap.
        chunk: Sequences per hmmscan call.

    Returns:
        ``{query_id: [{profile, evalue}, ...]}``, best hit first.
    """
    ids = list(queries)
    results: dict[str, list[dict[str, object]]] = {i: [] for i in ids}
    start = time.time()
    for offset in range(0, len(ids), chunk):
        batch = ids[offset : offset + chunk]
        output = run_pyhmmer_hmmscan(
            PyHmmscanInput(sequences=[queries[i][1] for i in batch], hmm_db=str(pfam)),
            PyHmmscanConfig(evalue_threshold=evalue, domain_evalue_threshold=evalue),
        )
        for hit in output.sequence_hits:
            results[batch[hit.query_idx]].append(
                {"profile": hit.target_name, "evalue": hit.evalue}
            )
        done = min(offset + chunk, len(ids))
        print(f"  {done}/{len(ids)} scanned ({time.time() - start:.0f}s)", flush=True)
    for hits in results.values():
        hits.sort(key=lambda h: h["evalue"])
    return results


def main() -> None:
    """Run the discovery scan and write per-query hits plus family counts."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pfam", required=True, type=Path)
    parser.add_argument("--toxins", type=Path,
                        default=Path("proto_pipelines/calibration/data/type_II_T_exp.fas"))
    parser.add_argument("--antitoxins", type=Path,
                        default=Path("proto_pipelines/calibration/data/type_II_AT_exp.fas"))
    parser.add_argument("--designs", type=Path,
                        default=Path("proto_pipelines/calibration/data/denovo_sequences.fasta"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--evalue", type=float, default=0.01)
    parser.add_argument("--chunk", type=int, default=100)
    args = parser.parse_args()

    queries = collect(args.toxins, args.antitoxins, args.designs)
    by_source = Counter(source for source, _ in queries.values())
    print(f"scanning {len(queries)} proteins against {args.pfam.name} at E<{args.evalue}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())), flush=True)

    hits = scan(queries, args.pfam, args.evalue, args.chunk)

    counts_all: Counter = Counter()
    counts_natural: Counter = Counter()
    for query_id, query_hits in hits.items():
        source = queries[query_id][0]
        for hit in query_hits:
            counts_all[hit["profile"]] += 1
            if source.startswith("natural"):
                counts_natural[hit["profile"]] += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "evalue": args.evalue,
        "n_queries": len(queries),
        "by_source": dict(by_source),
        "per_query": hits,
        "family_counts_all": counts_all.most_common(),
        "family_counts_natural_only": counts_natural.most_common(),
    }, indent=1))
    print(f"\n{len(counts_all)} distinct families (all sources); "
          f"{len(counts_natural)} from naturals alone")
    print("top 20 (natural-only prevalence):")
    for family, count in counts_natural.most_common(20):
        print(f"  {count:4}  {family}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
