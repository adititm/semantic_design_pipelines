"""Build a labelled single-chain set for calibrating the monomer screen.

The monomer screen asks one question: does this ORF look like a real folded
protein, or like noise that Prodigal happened to call? So the calibration
mirrors that question rather than borrowing the complex-level one.

* **Positives** - individual toxin and antitoxin chains from TADB 3.0's
  experimentally validated type II loci. Real, folded, characterised proteins.
* **Negatives** - the same chains with residues permuted. Composition, length
  and amino-acid usage identical; structure destroyed. A monomer score that
  cannot separate these is not reading structure.

Shuffles are the right negative here (unlike at complex level, where
cross-family and same-family mattered) because the screen is not asking
"which protein is this" but "is this protein-like at all".

Folding runs **single-sequence**, matching how ``af3_monomer_screen`` actually
runs in the pipelines. MSA-mode pLDDT is substantially higher and thresholds
do not transfer between the two modes.

Usage:
    python -m proto_pipelines.calibration.build_monomer_set \
        --toxins type_II_T_exp.fas --antitoxins type_II_AT_exp.fas \
        --out monomer_pairs.csv --n-per-class 25
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

# The monomer screen's own length window (t2ta/acr configs use 50-300).
MIN_LENGTH = 50
MAX_LENGTH = 300


def collect_chains(toxin_fasta: Path, antitoxin_fasta: Path) -> list[dict[str, Any]]:
    """Gather individual TA chains that fit the screen's length window.

    Args:
        toxin_fasta: TADB ``type_II_T_exp.fas``.
        antitoxin_fasta: TADB ``type_II_AT_exp.fas``.

    Returns:
        One record per chain with ``chain_id``, ``sequence``, ``length`` and
        ``role`` (toxin or antitoxin).

    Raises:
        ValueError: If no chain survives filtering.
    """
    chains: list[dict[str, Any]] = []
    for path, role in ((toxin_fasta, "toxin"), (antitoxin_fasta, "antitoxin")):
        for header, sequence in read_fasta(path).items():
            name = header.split()[0]
            if not set(sequence) <= STANDARD_AA:
                continue
            if not (MIN_LENGTH <= len(sequence) <= MAX_LENGTH):
                continue
            chains.append(
                {"chain_id": name, "sequence": sequence, "length": len(sequence), "role": role}
            )
    if not chains:
        raise ValueError("No chain passed the length/alphabet filter")
    logger.info("Collected %d chains within %d-%d aa", len(chains), MIN_LENGTH, MAX_LENGTH)
    return chains


def build(chains: list[dict[str, Any]], n_per_class: int, seed: int) -> pd.DataFrame:
    """Sample natural chains and emit each with a shuffled counterpart.

    Pairing each positive with a shuffle *of itself* makes the comparison
    matched: every negative has exactly the composition and length of a
    positive, so a separation cannot be a length or composition artifact.

    Args:
        chains: Output of :func:`collect_chains`.
        n_per_class: Natural chains to sample (the same number of shuffles
            is produced).
        seed: RNG seed.

    Returns:
        One row per chain to fold, with ``label`` 1 natural / 0 shuffled.
    """
    rng = random.Random(seed)
    chosen = rng.sample(chains, min(n_per_class, len(chains)))

    rows: list[dict[str, Any]] = []
    for record in chosen:
        rows.append(
            {
                "chain_uid": record["chain_id"],
                "sequence": record["sequence"],
                "length": record["length"],
                "role": record["role"],
                "label": 1,
                "chain_class": "natural",
            }
        )
        residues = list(record["sequence"])
        rng.shuffle(residues)
        rows.append(
            {
                "chain_uid": f"{record['chain_id']}_shuf",
                "sequence": "".join(residues),
                "length": record["length"],
                "role": record["role"],
                "label": 0,
                "chain_class": "shuffled",
            }
        )

    frame = pd.DataFrame(rows)
    logger.info("Built %d chains: %s", len(frame), frame["chain_class"].value_counts().to_dict())
    return frame


def main() -> None:
    """Build the monomer calibration set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toxins", required=True, type=Path)
    parser.add_argument("--antitoxins", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    chains = collect_chains(args.toxins, args.antitoxins)
    frame = build(chains, args.n_per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"Wrote {len(frame)} chains to {args.out}")
    print(frame.groupby(["chain_class", "role"]).size().to_string())


if __name__ == "__main__":
    main()
