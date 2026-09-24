"""Build a compact TA profile-HMM subset from families known TA proteins hit.

A hand-curated list of TA-sounding family names fails on this data: all four
experimentally validated EvoAT antitoxins have ``DUF6290`` -- a domain of
unknown function -- as their top Pfam hit, and none hits a canonical
antitoxin family. Conversely, matching the word "toxin" pulls in 292 Pfam
families that are overwhelmingly snake, scorpion and diphtheria exotoxins.

So the family list is derived from evidence: scan known type II TA proteins
against full Pfam (``ta_family_discovery.json``) and keep the families they
hit. A subset HMM also makes the filter usable in-pipeline -- scanning full
Pfam costs ~11 s per protein.

The E-value used to build the list is a real choice: at E<1e-5 the list
misses EvoAT1 and EvoAT2, the weakest-hitting validated antitoxins. E<1e-2
catches all nine knowns and is the default.

Usage:
    python -m proto_pipelines.calibration.build_ta_hmm_subset \
        --discovery ta_family_discovery.json --pfam /path/Pfam-A.hmm \
        --out ta_families.hmm --max-evalue 0.01
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def families_from_discovery(path: Path, max_evalue: float) -> tuple[set[str], dict[str, int]]:
    """Collect families hit below ``max_evalue`` by any known TA protein.

    Args:
        path: ``ta_family_discovery.json`` from the discovery scan.
        max_evalue: E-value ceiling for a hit to contribute a family.

    Returns:
        ``(families, counts)`` -- the family names and how many known
        proteins hit each.
    """
    data = json.loads(path.read_text())
    counts: dict[str, int] = {}
    for hits in data["per_query"].values():
        for hit in hits:
            if hit["evalue"] < max_evalue:
                counts[hit["profile"]] = counts.get(hit["profile"], 0) + 1
    return set(counts), counts


def subset_hmm(pfam: Path, families: set[str], out: Path) -> int:
    """Write a new HMM file containing only the named profiles.

    HMMER3 flat files are records separated by ``//``, each carrying a
    ``NAME`` line, so this is a streaming text filter -- no hmmfetch index
    needed and memory stays flat on a 1.6 GB input.

    Args:
        pfam: Full Pfam-A.hmm (HMMER3 format).
        families: Profile names to keep.
        out: Destination HMM file.

    Returns:
        Number of profiles written.

    Raises:
        FileNotFoundError: If ``pfam`` does not exist.
    """
    if not pfam.exists():
        raise FileNotFoundError(f"Pfam HMM not found: {pfam}")
    written = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with pfam.open() as handle, out.open("w") as sink:
        record: list[str] = []
        name: str | None = None
        for line in handle:
            record.append(line)
            if line.startswith("NAME "):
                name = line.split(None, 1)[1].strip()
            elif line.startswith("//"):
                if name in families:
                    sink.writelines(record)
                    written += 1
                record, name = [], None
    return written


def main() -> None:
    """Build the TA subset HMM from the discovery scan."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--pfam", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-evalue", type=float, default=0.01)
    args = parser.parse_args()

    families, counts = families_from_discovery(args.discovery, args.max_evalue)
    print(f"{len(families)} families hit at E<{args.max_evalue} by known TA proteins")
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:15]
    print("  most frequent:", ", ".join(f"{f}({c})" for f, c in top))

    written = subset_hmm(args.pfam, families, args.out)
    print(f"\nwrote {written}/{len(families)} profiles to {args.out}")
    missing = len(families) - written
    if missing:
        print(f"WARNING: {missing} family name(s) were not present in this Pfam build.")
    (args.out.with_suffix(".families.txt")).write_text(
        "\n".join(f"{f}\t{counts[f]}" for f, _ in sorted(counts.items())) + "\n")
    print(f"family list + hit counts: {args.out.with_suffix('.families.txt')}")


if __name__ == "__main__":
    main()
