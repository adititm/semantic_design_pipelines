"""Build an in-domain monomer calibration set from the paper's de novo designs.

Calibrating the monomer screen on *natural* proteins measures the wrong
population: the screen is applied to Evo output, and natural chains both fold
better and are likely represented in AlphaFold 3's training data. This builds
the matched in-domain alternative.

* **Positives** - Evo-designed chains from the paper's synthesis-order set,
  restricted to those with no BLAST match ("De Novo"), so they are genuinely
  novel rather than near-copies of natural proteins.
* **Negatives** - each positive with its residues permuted. Composition,
  length and amino-acid usage identical; structure destroyed.

**Selection bias, stated plainly:** these designs passed the paper's own
ESMFold pLDDT/pTM screen before being ordered, so they are enriched for
foldable sequences relative to raw Evo output. The separation measured here
is therefore an *upper bound* on what the same gate would achieve on
unfiltered generations. The experimentally validated chains (EvoRele1,
EvoAT1-4) are reported separately as gold-label reference points.

Usage:
    python -m proto_pipelines.calibration.build_denovo_monomer_set \
        --order-csv compiled_toxin_antitoxins_order_full_info.csv \
        --validated-fasta denovo_sequences.fasta --out denovo_monomer_chains.csv
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from proto_pipelines.calibration.build_tadb_set import STANDARD_AA, read_fasta

logger = logging.getLogger(__name__)

MIN_LENGTH = 50
MAX_LENGTH = 300
VALIDATED = ("EvoRele1", "EvoAT1", "EvoAT2", "EvoAT3", "EvoAT4")


def build(order_csv: Path, validated_fasta: Path | None, seed: int) -> pd.DataFrame:
    """Assemble de novo positives and their composition-matched shuffles.

    Args:
        order_csv: The paper's synthesis-order table, with ``Sequence`` and a
            ``De Novo/Blast`` column.
        validated_fasta: Optional FASTA holding the experimentally validated
            designs, flagged separately in the output.
        seed: RNG seed for the shuffles.

    Returns:
        One row per chain to fold, ``label`` 1 de novo / 0 shuffled, with a
        ``validated`` flag marking the gold-label chains.

    Raises:
        ValueError: If the order table lacks the expected columns or yields
            no usable sequence.
    """
    frame = pd.read_csv(order_csv)
    for column in ("Sequence", "De Novo/Blast"):
        if column not in frame.columns:
            raise ValueError(f"{order_csv} is missing the {column!r} column")

    # Load the validated sequences FIRST so order-set duplicates inherit the
    # gold label instead of silently deduping it away.
    validated_by_seq: dict[str, str] = {}
    if validated_fasta is not None and validated_fasta.exists():
        for header, sequence in read_fasta(validated_fasta).items():
            name = header.split()[0]
            if name in VALIDATED:
                validated_by_seq[sequence.upper()] = name

    seen: set[str] = set()
    positives: list[dict[str, Any]] = []
    for row in frame.itertuples():
        sequence = str(row.Sequence).strip().upper()
        novel = str(getattr(row, "_5", "")).strip().lower() == "de novo"
        if not novel or sequence in seen:
            continue
        if not set(sequence) <= STANDARD_AA:
            continue
        if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):
            continue
        seen.add(sequence)
        gold = validated_by_seq.get(sequence)
        positives.append({
            "chain_uid": gold or f"denovo_{len(positives):03d}",
            "sequence": sequence,
            "validated": gold is not None,
        })

    # Any validated design not already present in the order table.
    for sequence, name in validated_by_seq.items():
        if sequence in seen:
            continue
        seen.add(sequence)
        positives.append({"chain_uid": name, "sequence": sequence, "validated": True})

    if not positives:
        raise ValueError("No de novo sequence survived filtering")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for record in positives:
        rows.append({**record, "length": len(record["sequence"]), "role": "denovo",
                     "label": 1, "chain_class": "denovo"})
        residues = list(record["sequence"])
        rng.shuffle(residues)
        rows.append({"chain_uid": f"{record['chain_uid']}_shuf", "sequence": "".join(residues),
                     "validated": record["validated"], "length": len(record["sequence"]),
                     "role": "denovo", "label": 0, "chain_class": "shuffled"})

    out = pd.DataFrame(rows)
    logger.info("Built %d chains (%d de novo, %d validated)", len(out),
                len(positives), sum(p["validated"] for p in positives))
    return out


def main() -> None:
    """Build the in-domain monomer calibration set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--order-csv", required=True, type=Path)
    parser.add_argument("--validated-fasta", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = build(args.order_csv, args.validated_fasta, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {len(out)} chains to {args.out}")
    print(out.groupby(["chain_class", "validated"]).size().to_string())
    print(f"length range: {out.length.min()}-{out.length.max()}")


if __name__ == "__main__":
    main()
