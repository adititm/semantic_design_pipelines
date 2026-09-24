"""Build the de novo TA arm from the paper's functional rescue outcomes.

This is the arm that matters for choosing a threshold to apply to Evo output,
because its labels are growth-rescue outcomes on designed sequences -- the
same domain the pipelines generate into -- rather than annotations on natural
complexes that AlphaFold 3 has probably seen in training.

It is also the arm most easily corrupted, so two rules are enforced rather
than documented:

* Rows marked ``UNKNOWN`` in the rescue matrix are skipped, never assumed.
  A guessed label is worse than a missing one: it does not reduce n, it
  biases the answer.
* The ``RelE`` identity must be resolved explicitly by the caller. The repo
  holds two different proteins that could be meant -- the natural homolog
  EvoRele1 was derived from (70.8% identity to it) and E. coli RelE P0C077
  (29.1%) -- and picking the wrong one silently mislabels several pairs.

Usage:
    python -m proto_pipelines.calibration.build_denovo_set \
        --sequences denovo_sequences.fasta --matrix rescue_matrix.csv \
        --rele-variant wtRelE_homolog --out denovo_pairs.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from proto_pipelines.calibration.build_tadb_set import read_fasta

logger = logging.getLogger(__name__)

# Matrix toxin name -> FASTA record name. "RelE" is deliberately absent: it is
# ambiguous in this repository and must be supplied via --rele-variant.
TOXIN_ALIASES = {
    "EvoRele1": "EvoRele1",
    "MazF": "MazF_ECOLI",
    "YoeB": "YoeB_ECOLI",
}
RELE_CHOICES = ("wtRelE_homolog", "RelE_ECOLI")


def _short_name(header: str) -> str:
    """Take the record name (first whitespace-delimited token) from a header."""
    return header.split()[0]


def build(
    sequences: dict[str, str], matrix: pd.DataFrame, rele_variant: str
) -> tuple[pd.DataFrame, list[str]]:
    """Turn resolved rescue outcomes into scoreable pairs.

    Args:
        sequences: Record name to sequence, from the de novo FASTA.
        matrix: Rescue matrix with ``toxin``, ``antitoxin``, ``outcome``.
        rele_variant: Which FASTA record the matrix's ``RelE`` refers to.

    Returns:
        ``(pairs, skipped)`` -- pairs in the calibration pair schema, with
        ``label`` 1 for rescue and 0 for no_rescue, and a list of human-readable
        reasons for every row that was not turned into a pair.

    Raises:
        ValueError: If ``rele_variant`` is not one of the known records, or a
            named toxin/antitoxin has no sequence.
    """
    if rele_variant not in RELE_CHOICES:
        raise ValueError(f"--rele-variant must be one of {RELE_CHOICES}, got {rele_variant!r}")
    aliases = {**TOXIN_ALIASES, "RelE": rele_variant}

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in matrix.to_dict("records"):
        toxin_name = str(record["toxin"]).strip()
        antitoxin_name = str(record["antitoxin"]).strip()
        outcome = str(record["outcome"]).strip()

        if outcome == "UNKNOWN":
            skipped.append(f"{toxin_name}+{antitoxin_name}: outcome UNKNOWN")
            continue
        if outcome not in ("rescue", "no_rescue"):
            raise ValueError(
                f"{toxin_name}+{antitoxin_name}: outcome must be rescue/no_rescue/UNKNOWN, "
                f"got {outcome!r}"
            )

        toxin_record = aliases.get(toxin_name, toxin_name)
        if toxin_record not in sequences:
            raise ValueError(f"No sequence for toxin {toxin_name!r} (looked for {toxin_record!r})")
        if antitoxin_name not in sequences:
            raise ValueError(f"No sequence for antitoxin {antitoxin_name!r}")

        label = 1 if outcome == "rescue" else 0
        rows.append(
            {
                "root_id": f"denovo_{toxin_name}_{antitoxin_name}",
                "prompt_id": "denovo",
                "protein_uid_1": toxin_record,
                "sequence_1": sequences[toxin_record],
                "protein_uid_2": antitoxin_name,
                "sequence_2": sequences[antitoxin_name],
                "label": label,
                "pair_class": "denovo_rescue" if label == 1 else "denovo_no_rescue",
                "negative_type": "" if label == 1 else "functional_no_rescue",
                "outcome": outcome,
                "toxin_name": toxin_name,
                "source": record.get("source", ""),
            }
        )

    return pd.DataFrame(rows), skipped


def main() -> None:
    """Build the de novo pair set from resolved rescue outcomes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--rele-variant",
        required=True,
        choices=RELE_CHOICES,
        help="Which protein the matrix's 'RelE' means; this repo holds two candidates.",
    )
    args = parser.parse_args()

    sequences = {_short_name(h): s for h, s in read_fasta(args.sequences).items()}
    matrix = pd.read_csv(args.matrix, comment="#")
    pairs, skipped = build(sequences, matrix, args.rele_variant)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out, index=False)

    n_pos = int((pairs["label"] == 1).sum()) if not pairs.empty else 0
    n_neg = int((pairs["label"] == 0).sum()) if not pairs.empty else 0
    print(f"\nBuilt {len(pairs)} pair(s): {n_pos} rescue (positive), {n_neg} no_rescue (negative)")
    print(f"Skipped {len(skipped)} unresolved row(s):")
    for reason in skipped:
        print(f"  - {reason}")
    if n_neg < 3:
        print(
            f"\nWARNING: only {n_neg} functional negative(s). That is a spot-check, not a "
            "calibration: no threshold should be fitted on this arm alone. Resolving the "
            "UNKNOWN rows, and adding the antitoxin candidates that failed to rescue "
            "EvoRele1, is what would make this arm decisive."
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
