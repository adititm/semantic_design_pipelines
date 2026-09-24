"""Build a monomer set of real pipeline ORFs, labelled by whether QC passed.

The shuffle-based calibration answered "can AlphaFold 3 read sequence order in
a design" (no). It did NOT answer the operationally important question: does
an AlphaFold 3 monomer score catch the junk ORFs a real run actually produces?
Shuffled designs inherit a real protein's composition and score high, so they
are not representative of spurious ORFs.

This builds the representative set instead. It takes DNA that Evo actually
generated, calls every ORF with Prodigal, and labels each by whether it passes
the paper's protein QC (length, partial, repetitiveness, diversity,
underrepresented residues, segmasker complexity):

* **label 1** - ORFs that cleared QC: what the screen should keep.
* **label 0** - ORFs QC rejected: real junk from real generations.

If AlphaFold 3 separates these, a basal monomer gate is justified and its
operating point can be read off. If it does not, QC is already doing the work
and the fold adds cost without discrimination.

Usage:
    python -m proto_pipelines.calibration.build_junk_monomer_set \
        --generated outputs/smoke/t2ta_sample/generated_sequences.csv \
        --out junk_monomer_chains.csv --max-per-class 40
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

from proto_pipelines.qc import (
    MIN_UNIQUE_AMINO_ACIDS,
    _call_orfs,
    _low_complexity_fractions,
    _strip_stop,
    has_underrepresented_amino_acids,
    is_highly_repetitive,
)

logger = logging.getLogger(__name__)

MIN_LENGTH = 50
MAX_LENGTH = 300
SEGMASKER_MAX = 0.1


def classify(dna_sequences: list[str]) -> list[dict[str, Any]]:
    """Call ORFs and label each by whether it clears the paper's protein QC.

    Args:
        dna_sequences: Generated DNA to call ORFs on.

    Returns:
        One record per ORF with ``sequence``, ``length``, ``qc_pass`` and
        ``reject_reason`` (empty when it passed).
    """
    records: list[dict[str, Any]] = []
    for orfs in _call_orfs(dna_sequences):
        for orf in orfs:
            protein = _strip_stop(orf.amino_acid_sequence)
            if not (MIN_LENGTH <= len(protein) <= MAX_LENGTH):
                # Out-of-window ORFs are excluded entirely rather than called
                # junk: length is a separate, cheaper filter and folding them
                # would confound length with quality.
                continue
            reason = ""
            if is_highly_repetitive(protein):
                reason = "repetitive"
            elif len(set(protein)) < MIN_UNIQUE_AMINO_ACIDS:
                reason = "low_diversity"
            elif has_underrepresented_amino_acids(protein):
                reason = "underrepresented_aas"
            records.append(
                {
                    "sequence": protein,
                    "length": len(protein),
                    "qc_pass": reason == "",
                    "reject_reason": reason,
                }
            )

    # segmasker last, in one batch, for the ORFs still passing.
    survivors = [r for r in records if r["qc_pass"]]
    if survivors:
        fractions = _low_complexity_fractions([r["sequence"] for r in survivors])
        for record, fraction in zip(survivors, fractions, strict=True):
            record["low_complexity_fraction"] = fraction
            if fraction > SEGMASKER_MAX:
                record["qc_pass"] = False
                record["reject_reason"] = "low_complexity"
    return records


def main() -> None:
    """Build the QC-pass vs QC-reject monomer set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--generated", required=True, nargs="+", type=Path,
                        help="generated_sequences.csv file(s) with a 'dna' column")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-per-class", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dna: list[str] = []
    for path in args.generated:
        frame = pd.read_csv(path)
        column = "dna" if "dna" in frame.columns else frame.columns[-1]
        dna.extend(s for s in frame[column].dropna().astype(str) if set(s) <= set("ACGTN"))
    if not dna:
        raise ValueError("No DNA sequences found in the supplied files")
    logger.info("Calling ORFs on %d generated sequence(s)", len(dna))

    records = classify(dna)
    passed = [r for r in records if r["qc_pass"]]
    failed = [r for r in records if not r["qc_pass"]]
    logger.info("ORFs in window: %d passed QC, %d rejected", len(passed), len(failed))
    if not failed:
        print("WARNING: no QC-rejected ORFs found; cannot build a junk negative class.")
    rng = random.Random(args.seed)
    passed = rng.sample(passed, min(args.max_per_class, len(passed)))
    failed = rng.sample(failed, min(args.max_per_class, len(failed)))

    rows = []
    for tag, group, label in (("qcpass", passed, 1), ("qcfail", failed, 0)):
        for i, r in enumerate(group):
            rows.append({"chain_uid": f"{tag}_{i:03d}", "sequence": r["sequence"],
                         "length": r["length"], "role": r["reject_reason"] or "qc_pass",
                         "label": label, "chain_class": tag})
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} chains to {args.out}")
    print(out.groupby(["chain_class", "role"]).size().to_string())


if __name__ == "__main__":
    main()
