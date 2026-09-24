"""Build a labelled toxin-antitoxin pair set for threshold calibration.

The paper's ESMFold pLDDT/pTM and pDockQ cutoffs do not transfer to
AlphaFold 3 (see ``proto_pipelines/README.md``). Choosing replacements needs
pairs whose labels are independent of any structure prediction, which rules
out reusing an AF3-filtered candidate list: calibrating a structure threshold
on a set selected by a structure criterion is circular.

TADB 3.0 supplies that independence. Its ``type_II_{T,AT}_exp`` files hold
experimentally validated type II TA loci where ``T{n}`` and ``AT{n}`` are the
two genes of the same locus -- a literature label, not a prediction.

Negatives are the part that decides whether the result means anything, so
three kinds are emitted at different difficulty, and each is scored
separately:

``cross_family``
    Antitoxin from a locus whose toxin belongs to a *different* toxin family.
    Unambiguously not partners, but an easy discrimination: good performance
    here alone would not justify a threshold.
``same_family``
    Antitoxin from a different locus whose toxin is in the *same* family.
    Composition, length and fold are matched, so this is the honest test.
    Caveat: TA antitoxins are known to cross-react within a family (the paper
    itself reports EvoAT2 neutralising three different toxins), so some of
    these "negatives" may be genuine binders. Performance against this set is
    therefore a **lower bound**, and it is labelled as such.
``shuffled``
    The cognate antitoxin with its residues permuted. Same composition, no
    structure. A sanity floor: a scorer that cannot separate this from the
    cognate pair is not reading structure at all.

Output is a CSV in the schema :mod:`proto_pipelines.calibration.run_folds`
consumes, so the calibration set is scored by the same AlphaFold 3 + pDockQ2
path as a design run. Extra ``label`` / ``negative_type`` columns pass into
``cofold_scores.csv`` for :mod:`proto_pipelines.calibration.analyze`.

Usage:
    python -m proto_pipelines.calibration.build_tadb_set \
        --toxins type_II_T_exp.fas --antitoxins type_II_AT_exp.fas \
        --out calibration_pairs.csv --n-per-class 20
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

TADB_BASE_URL = "https://bioinfo-mml.sjtu.edu.cn/TADB3/download"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# AlphaFold 3 cost scales steeply with length, and the cofold stage's defaults
# drop pairs outside this window; matching them keeps the calibration set and
# the design runs on the same footing.
MIN_TOTAL_RESIDUES = 100
MAX_TOTAL_RESIDUES = 1024
MIN_CHAIN_RESIDUES = 40

# Family assignment for negative construction. 30% identity over 50% coverage
# is the conventional homology floor; clusters approximate TA toxin families
# (RelE, MazF, VapC, HigB, ...).
FAMILY_MIN_SEQ_ID = 0.3
FAMILY_COVERAGE = 0.5


def read_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into ``{header: sequence}``.

    Args:
        path: FASTA file.

    Returns:
        Header line (without ``>``) mapped to its sequence.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no records are found.
    """
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    records: dict[str, str] = {}
    header: str | None = None
    chunks: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records[header] = "".join(chunks)
            header, chunks = line[1:].strip(), []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        records[header] = "".join(chunks)
    if not records:
        raise ValueError(f"FASTA contained no records: {path}")
    return records


def _locus_id(header: str, prefix: str) -> int | None:
    """Extract the numeric locus id from a TADB header (``T28`` -> ``28``)."""
    match = re.match(rf"^{prefix}(\d+)\b", header)
    return int(match.group(1)) if match else None


def _organism(header: str) -> str | None:
    """Extract the bracketed organism name from a TADB header."""
    match = re.search(r"\[([^\]]+)\]", header)
    return match.group(1) if match else None


def load_cognate_loci(toxin_fasta: Path, antitoxin_fasta: Path) -> pd.DataFrame:
    """Join TADB toxins and antitoxins into cognate loci.

    Args:
        toxin_fasta: ``type_II_T_exp.fas``.
        antitoxin_fasta: ``type_II_AT_exp.fas``.

    Returns:
        One row per locus that has both genes and passes the length and
        alphabet filters, with ``locus``, ``organism``, ``toxin_id``,
        ``toxin``, ``antitoxin_id``, ``antitoxin``, ``total_residues``.

    Raises:
        ValueError: If no locus survives filtering, or if the toxin and
            antitoxin of a locus disagree on organism -- that would mean
            ``T{n}``/``AT{n}`` is not a same-locus convention and every
            cognate label would be suspect.
    """
    toxins = read_fasta(toxin_fasta)
    antitoxins = read_fasta(antitoxin_fasta)
    by_toxin = {
        _locus_id(h, "T"): (h, s) for h, s in toxins.items() if _locus_id(h, "T") is not None
    }
    by_antitoxin = {
        _locus_id(h, "AT"): (h, s) for h, s in antitoxins.items() if _locus_id(h, "AT") is not None
    }

    rows: list[dict[str, Any]] = []
    mismatched = 0
    for locus in sorted(set(by_toxin) & set(by_antitoxin)):
        toxin_header, toxin_seq = by_toxin[locus]
        antitoxin_header, antitoxin_seq = by_antitoxin[locus]
        if _organism(toxin_header) != _organism(antitoxin_header):
            mismatched += 1
            continue
        if not set(toxin_seq) <= STANDARD_AA or not set(antitoxin_seq) <= STANDARD_AA:
            continue
        if len(toxin_seq) < MIN_CHAIN_RESIDUES or len(antitoxin_seq) < MIN_CHAIN_RESIDUES:
            continue
        total = len(toxin_seq) + len(antitoxin_seq)
        if not (MIN_TOTAL_RESIDUES <= total <= MAX_TOTAL_RESIDUES):
            continue
        rows.append(
            {
                "locus": locus,
                "organism": _organism(toxin_header),
                "toxin_id": f"T{locus}",
                "toxin": toxin_seq,
                "antitoxin_id": f"AT{locus}",
                "antitoxin": antitoxin_seq,
                "total_residues": total,
            }
        )

    if mismatched:
        raise ValueError(
            f"{mismatched} loci had different organisms for T/AT; the T{{n}}/AT{{n}} "
            "same-locus assumption does not hold and cognate labels cannot be trusted."
        )
    if not rows:
        raise ValueError("No locus survived filtering; check the input FASTA files.")
    logger.info("Loaded %d cognate loci after filtering", len(rows))
    return pd.DataFrame(rows)


def assign_families(loci: pd.DataFrame) -> dict[int, int]:
    """Cluster toxins by sequence identity to approximate TA toxin families.

    Args:
        loci: Output of :func:`load_cognate_loci`.

    Returns:
        ``{locus: family index}``. Every locus is assigned; singletons get
        their own family.

    Raises:
        RuntimeError: If MMseqs2 returns a clustering that does not cover
            every input sequence, which would silently drop loci.
    """
    from proto_tools import (
        Mmseqs2ClusteringConfig,
        Mmseqs2ClusteringInput,
        run_mmseqs2_clustering,
    )

    sequences = loci["toxin"].tolist()
    ids = [f"T{locus}" for locus in loci["locus"]]
    result = run_mmseqs2_clustering(
        Mmseqs2ClusteringInput(input_sequences=sequences, sequence_ids=ids),
        Mmseqs2ClusteringConfig(min_seq_id=FAMILY_MIN_SEQ_ID, coverage=FAMILY_COVERAGE),
    )

    # One result per input sequence, each naming the cluster it landed in.
    cluster_index: dict[str, int] = {}
    family_of: dict[str, int] = {}
    for row in result.results:
        index = cluster_index.setdefault(row.cluster_id, len(cluster_index))
        family_of[str(row.sequence_id)] = index

    missing = [i for i in ids if i not in family_of]
    if missing:
        raise RuntimeError(
            f"MMseqs2 clustering did not assign {len(missing)} of {len(ids)} toxins "
            f"(e.g. {missing[:3]}); refusing to build negatives from a partial clustering."
        )

    by_locus = {int(locus): family_of[f"T{locus}"] for locus in loci["locus"]}
    sizes = defaultdict(int)
    for family in by_locus.values():
        sizes[family] += 1
    multi = sum(1 for n in sizes.values() if n > 1)
    logger.info(
        "Toxins fall into %d families (%d with >1 member) at %.0f%% identity",
        len(sizes),
        multi,
        FAMILY_MIN_SEQ_ID * 100,
    )
    return by_locus


def build_pairs(
    loci: pd.DataFrame, families: dict[int, int], n_per_class: int, seed: int
) -> pd.DataFrame:
    """Assemble cognate positives and the three negative classes.

    Args:
        loci: Output of :func:`load_cognate_loci`.
        families: Output of :func:`assign_families`.
        n_per_class: Target pairs per class. Fewer are emitted for a class
            when the data cannot supply that many (reported, never padded).
        seed: RNG seed, so the set is reproducible.

    Returns:
        One row per pair in the calibration pair schema, plus ``label``
        (1 cognate / 0 non-cognate), ``pair_class`` and ``negative_type``.
    """
    rng = random.Random(seed)
    records = loci.to_dict("records")
    by_family: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_family[families[record["locus"]]].append(record)

    rows: list[dict[str, Any]] = []

    def emit(toxin_row: dict[str, Any], antitoxin_row: dict[str, Any], klass: str, label: int,
             antitoxin_seq: str | None = None, suffix: str = "") -> None:
        sequence = antitoxin_seq if antitoxin_seq is not None else antitoxin_row["antitoxin"]
        toxin_uid = toxin_row["toxin_id"]
        antitoxin_uid = antitoxin_row["antitoxin_id"] + suffix
        rows.append(
            {
                "root_id": f"{klass}_{toxin_uid}_{antitoxin_uid}",
                "prompt_id": klass,
                "protein_uid_1": toxin_uid,
                "sequence_1": toxin_row["toxin"],
                "protein_uid_2": antitoxin_uid,
                "sequence_2": sequence,
                "label": label,
                "pair_class": klass,
                "negative_type": "" if label == 1 else klass,
                "toxin_locus": toxin_row["locus"],
                "antitoxin_locus": antitoxin_row["locus"],
                "toxin_family": families[toxin_row["locus"]],
                "antitoxin_family": families[antitoxin_row["locus"]],
                "toxin_organism": toxin_row["organism"],
            }
        )

    # Positives: sample loci, preferring families with >1 member so the same
    # toxins can also appear in a same_family negative (paired design).
    multi_family = [r for r in records if len(by_family[families[r["locus"]]]) > 1]
    pool = multi_family if len(multi_family) >= n_per_class else records
    positives = rng.sample(pool, min(n_per_class, len(pool)))
    for record in positives:
        emit(record, record, "cognate", 1)

    # same_family: swap the antitoxin with another locus in the same family.
    same_family_made = 0
    for record in positives:
        siblings = [r for r in by_family[families[record["locus"]]] if r["locus"] != record["locus"]]
        if not siblings:
            continue
        emit(record, rng.choice(siblings), "same_family", 0)
        same_family_made += 1

    # cross_family: swap with a locus from a different toxin family.
    for record in positives:
        others = [r for r in records if families[r["locus"]] != families[record["locus"]]]
        if not others:
            continue
        emit(record, rng.choice(others), "cross_family", 0)

    # shuffled: the cognate antitoxin, residues permuted.
    for record in positives:
        residues = list(record["antitoxin"])
        rng.shuffle(residues)
        emit(record, record, "shuffled", 0, antitoxin_seq="".join(residues), suffix="_shuf")

    frame = pd.DataFrame(rows)
    counts = frame["pair_class"].value_counts().to_dict()
    logger.info("Built %d pairs: %s", len(frame), counts)
    if same_family_made < len(positives):
        print(
            f"WARNING: only {same_family_made}/{len(positives)} positives had a same-family "
            "sibling; the hardest negative class is under-populated."
        )
    return frame


def main() -> None:
    """Build the calibration pair set and write it as CSV."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toxins", required=True, type=Path, help="TADB type_II_T_exp.fas")
    parser.add_argument("--antitoxins", required=True, type=Path, help="TADB type_II_AT_exp.fas")
    parser.add_argument("--out", required=True, type=Path, help="Destination pairs CSV")
    parser.add_argument("--n-per-class", type=int, default=20, help="Pairs per class")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    args = parser.parse_args()

    loci = load_cognate_loci(args.toxins, args.antitoxins)
    families = assign_families(loci)
    pairs = build_pairs(loci, families, args.n_per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False)
    logger.info("Wrote %d pairs to %s", len(pairs), args.out)


if __name__ == "__main__":
    main()
