"""Convert MMseqs2 profiles into PSI-BLAST-format PSSMs for POSSUM descriptors.

AcrNET consumes four POSSUM descriptors (``dpc_pssm``, ``pssm_composition``,
``pssm_ac``, ``rpssm``; 400+400+200+110 = 1110 dims) computed from a
PSI-BLAST PSSM. There is no BLAST in this environment, so the profile comes
from MMseqs2 (``search`` -> ``result2profile`` -> ``profile2pssm``) against
uniref30 instead.

**The column order differs and must be permuted.** MMseqs2 emits amino acids
alphabetically (``ACDEFGHIKLMNPQRSTVWY``); PSI-BLAST -- and therefore
``pssmpro``, which hardcodes it -- uses ``ARNDCQEGHILKMFPSTWYV``. Feeding
MMseqs columns straight in produces a scrambled matrix that still has the
right shape, so nothing errors and every descriptor is quietly wrong.

This remains a substitution, not the real thing: an MMseqs2 profile is built
by a different search with different pseudocounts than a PSI-BLAST PSSM. Its
effect on AcrNET is measured against the model's own shipped test set rather
than assumed.

Usage:
    python -m proto_pipelines.calibration.mmseqs_pssm \
        --pssm-tsv pssm.tsv --ids ids.txt --out-dir pssm_files/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: MMseqs2 profile2pssm column order.
MMSEQS_ORDER = "ACDEFGHIKLMNPQRSTVWY"
#: PSI-BLAST column order, which pssmpro assumes.
PSIBLAST_ORDER = "ARNDCQEGHILKMFPSTWYV"


def permutation() -> list[int]:
    """Index map from MMseqs2 column order to PSI-BLAST column order.

    Returns:
        ``perm`` such that ``[row[i] for i in perm]`` reorders one MMseqs row
        into PSI-BLAST order.

    Raises:
        ValueError: If the two orders are not permutations of one another,
            which would mean one of the constants is wrong.
    """
    if sorted(MMSEQS_ORDER) != sorted(PSIBLAST_ORDER):
        raise ValueError("column-order constants are not permutations of each other")
    return [MMSEQS_ORDER.index(a) for a in PSIBLAST_ORDER]


def parse(pssm_tsv: Path) -> list[tuple[str, list[str], list[list[int]]]]:
    """Split an MMseqs2 ``profile2pssm`` file into per-sequence matrices.

    Args:
        pssm_tsv: Output of ``mmseqs profile2pssm``.

    Returns:
        One ``(header, residues, rows)`` per query, ``rows`` in MMseqs column
        order.
    """
    blocks: list[tuple[str, list[str], list[list[int]]]] = []
    header: str | None = None
    residues: list[str] = []
    rows: list[list[int]] = []
    for line in pssm_tsv.read_text().splitlines():
        if line.startswith("Query profile"):
            if header is not None:
                blocks.append((header, residues, rows))
            header, residues, rows = line, [], []
        elif line.startswith("Pos"):
            continue
        elif line.strip():
            parts = line.split("\t")
            residues.append(parts[1])
            rows.append([int(x) for x in parts[2:22]])
    if header is not None:
        blocks.append((header, residues, rows))
    return blocks


def write_psiblast(path: Path, residues: list[str], rows: list[list[int]]) -> None:
    """Write one PSSM in the ASCII layout ``pssmpro.read_pssm_matrix`` parses.

    That reader skips three lines, then takes fields ``1:42`` of each row, so
    the file needs three leading lines and 41 fields per row (residue, 20
    scores, 20 placeholder percentages).

    Args:
        path: Destination ``.pssm`` file.
        residues: Consensus residue per position.
        rows: Score rows, already in PSI-BLAST column order.
    """
    perm = permutation()
    with path.open("w") as fh:
        fh.write("\n")
        fh.write("Last position-specific scoring matrix computed\n")
        fh.write("           " + " ".join(PSIBLAST_ORDER) + "  "
                 + " ".join(PSIBLAST_ORDER) + "\n")
        for index, (residue, row) in enumerate(zip(residues, rows, strict=True), start=1):
            ordered = [row[i] for i in perm]
            scores = " ".join(f"{v:4d}" for v in ordered)
            # pssmpro only reads the score block; the percentage columns must
            # exist for the field count but their values are never used.
            pct = " ".join(f"{0:4d}" for _ in ordered)
            fh.write(f"{index:>5} {residue} {scores} {pct}\n")


def main() -> None:
    """Convert every profile in an MMseqs2 PSSM file to a PSI-BLAST file."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pssm-tsv", required=True, type=Path)
    parser.add_argument("--ids", required=True, type=Path,
                        help="One query id per line, in the order MMseqs emitted them.")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    blocks = parse(args.pssm_tsv)
    ids = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
    if len(blocks) != len(ids):
        raise ValueError(
            f"{len(blocks)} profiles but {len(ids)} ids; the id list must match "
            "the order profile2pssm emitted, or PSSMs attach to the wrong sequence"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for (header, residues, rows), name in zip(blocks, ids, strict=True):
        write_psiblast(args.out_dir / f"{name}.pssm", residues, rows)
    logger.info("Wrote %d PSSM file(s) to %s", len(blocks), args.out_dir)


if __name__ == "__main__":
    main()
