"""Assemble a labelled anti-CRISPR calibration set with four negative classes.

Any Acr caller worth gating on must separate known Acrs from sequences that
merely *look* like them. The negative classes escalate deliberately:

``shuffled``
    Composition-matched shuffles of the positives. A caller that cannot beat
    this is reading amino-acid composition, not Acr-ness.
``random_orf``
    ORFs translated from random DNA. The junk floor.
``phage``
    Non-Acr proteins from bacteriophages (UniProt ``virus_host_id:2``).
    **This is the control that can fail informatively.** The Acr prompts
    generate phage-like sequence, so a caller that has really learned "phage
    protein" rather than "anti-CRISPR" scores well on the first two classes
    and collapses here.
``aca``
    Anti-CRISPR *associated* proteins -- the HTH regulators encoded beside
    Acrs. Not Acrs, but maximally co-occurring with them, so this isolates
    guilt-by-association leakage from genuine Acr recognition.

Positives are the 64 experimentally named Acrs shipped with AcrFinder,
spanning CRISPR types I, II, III, V and VI. The much larger AcrDatabase.faa
is deliberately not used: its ``pAcr`` entries are of unresolved provenance,
and unvalidated predictions cannot serve as ground truth.

Lengths are caliper-matched to the positives, because the Type II TA
calibration found an apparent signal that turned out to be length alone once
matched. Without matching, a caller can win by preferring short proteins.

Usage:
    python -m proto_pipelines.calibration.build_acr_set \
        --known-acr known-acr.faa --phage phage_nonacr.faa \
        --aca 401-aca.faa --out acr_calibration.csv
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

STOP = "*"
CODON_STOPS = {"TAA", "TAG", "TGA"}


def read_fasta(path: Path) -> dict[str, str]:
    """Read a FASTA into ``{id: sequence}``, keyed by the first header token.

    Args:
        path: FASTA file.

    Returns:
        Sequences with any trailing stop character removed.

    Raises:
        ValueError: If the file yields no records, which means the wrong file
            was passed rather than that the set is legitimately empty.
    """
    records: dict[str, str] = {}
    name: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                records[name] = "".join(chunks).strip(STOP)
            name = line[1:].split()[0]
            chunks = []
        elif line.strip():
            chunks.append(line.strip())
    if name is not None:
        records[name] = "".join(chunks).strip(STOP)
    if not records:
        raise ValueError(f"{path} contained no FASTA records")
    return records


def caliper_match(
    pool: dict[str, str], targets: list[int], caliper: int, rng: random.Random
) -> list[tuple[str, str]]:
    """Draw one pool sequence per target length, within ``caliper`` residues.

    Sampling without replacement, so a small pool cannot be represented by the
    same few sequences repeatedly.

    Args:
        pool: Candidate sequences.
        targets: Lengths to match, one draw each.
        caliper: Maximum absolute length difference tolerated.
        rng: Seeded RNG.

    Returns:
        ``(id, sequence)`` pairs, shorter than ``targets`` when the pool runs
        out of in-caliper candidates.
    """
    remaining = dict(pool)
    matched: list[tuple[str, str]] = []
    for target in targets:
        eligible = [k for k, v in remaining.items() if abs(len(v) - target) <= caliper]
        if not eligible:
            continue
        pick = rng.choice(eligible)
        matched.append((pick, remaining.pop(pick)))
    return matched


def random_orf_proteins(count: int, lengths: list[int], rng: random.Random) -> list[str]:
    """Translate random uniform-composition DNA into stop-free proteins.

    Built at the protein level from a uniform amino-acid alphabet rather than
    by calling ORFs with Prodigal: the point of this class is a composition
    floor, and going through Prodigal would bias toward ORF-like codon usage
    that the positives do not share.

    Args:
        count: Number of proteins to build.
        lengths: Target lengths, cycled if shorter than ``count``.
        rng: Seeded RNG.

    Returns:
        Protein strings over the 20 standard amino acids.
    """
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    return [
        "".join(rng.choice(alphabet) for _ in range(lengths[i % len(lengths)]))
        for i in range(count)
    ]


def build(
    known_acr: Path, phage: Path, aca: Path, caliper: int, seed: int
) -> pd.DataFrame:
    """Assemble the labelled set.

    Args:
        known_acr: FASTA of experimentally named Acrs (the positives).
        phage: FASTA of non-Acr bacteriophage proteins.
        aca: FASTA of anti-CRISPR-associated (Aca) proteins.
        caliper: Length-matching tolerance in residues.
        seed: RNG seed.

    Returns:
        One row per sequence with ``seq_id``, ``sequence``, ``length``,
        ``label`` (1 for Acr) and ``seq_class``.
    """
    rng = random.Random(seed)
    positives = read_fasta(known_acr)
    lengths = [len(s) for s in positives.values()]

    rows: list[dict[str, Any]] = [
        {"seq_id": k, "sequence": v, "length": len(v), "label": 1, "seq_class": "acr"}
        for k, v in positives.items()
    ]

    for name, sequence in positives.items():
        residues = list(sequence)
        rng.shuffle(residues)
        rows.append({
            "seq_id": f"{name}_shuffled", "sequence": "".join(residues),
            "length": len(residues), "label": 0, "seq_class": "shuffled",
        })

    for index, protein in enumerate(random_orf_proteins(len(positives), lengths, rng)):
        rows.append({
            "seq_id": f"random_{index:03d}", "sequence": protein,
            "length": len(protein), "label": 0, "seq_class": "random_orf",
        })

    for source, klass in ((phage, "phage"), (aca, "aca")):
        pool = read_fasta(source)
        matched = caliper_match(pool, lengths, caliper, rng)
        if len(matched) < len(lengths):
            logger.warning(
                "%s: only %d/%d length-matched within +/-%d residues",
                klass, len(matched), len(lengths), caliper,
            )
        for name, sequence in matched:
            rows.append({
                "seq_id": name, "sequence": sequence, "length": len(sequence),
                "label": 0, "seq_class": klass,
            })

    frame = pd.DataFrame(rows)
    # Any caller can be gamed by length, so the matching has to be checkable
    # from the output rather than taken on trust.
    logger.info("Median length by class:\n%s",
                frame.groupby("seq_class")["length"].agg(["count", "median"]).to_string())
    return frame


def main() -> None:
    """Parse arguments and write the labelled calibration set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--known-acr", required=True, type=Path)
    parser.add_argument("--phage", required=True, type=Path)
    parser.add_argument("--aca", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--caliper", type=int, default=15,
                        help="Length-matching tolerance in residues.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    frame = build(args.known_acr, args.phage, args.aca, args.caliper, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    logger.info("Wrote %d sequence(s) to %s", len(frame), args.out)


if __name__ == "__main__":
    main()
