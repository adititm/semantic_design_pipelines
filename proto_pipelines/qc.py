"""ORF calling and protein QC, matching the published filtering criteria.

proto-language ships ``overall-protein-quality``, which covers the same five
checks, but its sub-constraints are parameterised differently from the paper's
(diversity is ``unique_AAs / 20``, repetitiveness defaults to 0.1, and the
balanced-AA check uses relative frequencies rather than a raw count of 2).
Swapping it in would quietly change which sequences survive, so the predicates
below reproduce ``filter_protein_fasta`` exactly. Prodigal and segmasker still
run through proto-tools, so the tool environments stay proto-managed.

Unlike the published workflow, which wrote one FASTA per stage, this constraint keeps the
surviving proteins on the proposal's metadata under ``qc_proteins`` so the
AlphaFold 3 screens downstream can read them without a file round-trip.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput
from proto_language.utils.base import BaseConfig, ConfigField
from proto_tools import (
    ProdigalConfig,
    ProdigalInput,
    SegmaskerConfig,
    SegmaskerInput,
    run_prodigal_prediction,
    run_segmasker,
)

# segmasker settings hard-coded in the published workflow `is_segmasked_greater_than_threshold`.
SEGMASKER_WINDOW = 15
SEGMASKER_LOCUT = 1.8
SEGMASKER_HICUT = 3.4

# Fixed parameters of the published workflow repetitiveness / diversity predicates.
REPEAT_MIN_K = 3
REPEAT_K_SPAN = 7
REPEAT_FRACTION = 0.3
MIN_UNIQUE_AMINO_ACIDS = 12
UNDERREPRESENTED_COUNT = 2
UNDERREPRESENTED_FRACTION = 0.3


class ProteinQCConfig(BaseConfig):
    """Configuration for :func:`prodigal_protein_qc_constraint`.

    Attributes:
        min_length: Minimum protein length in amino acids (inclusive).
        max_length: Maximum protein length in amino acids (inclusive).
        filter_partial: Drop ORFs Prodigal flagged as running off a sequence
            edge (the published workflow's ``partial=00`` requirement).
        segmasker_threshold: Maximum tolerated fraction of low-complexity
            residues.
        min_surviving_proteins: Number of proteins a generation must yield for
            the proposal to pass. ``1`` for single-protein screens (Acr),
            ``2`` for pipelines that need an intra-generation pair (Type II TA).
        require_prompt_dna: When set, keep only ORFs whose nucleotide sequence
            contains this DNA string. This is the completion pipelines'
            ``filter_orfs_by_prompt`` step: it keeps the ORF that continues the
            truncated gene and drops ORFs Prodigal found elsewhere in the
            generation. Matching is case-insensitive and forward-strand only,
            as in the published workflow.
    """

    min_length: int = ConfigField(
        default=50, ge=1, title="Minimum Protein Length", description="Minimum protein length in aa."
    )
    max_length: int = ConfigField(
        default=300, ge=1, title="Maximum Protein Length", description="Maximum protein length in aa."
    )
    filter_partial: bool = ConfigField(
        default=True, title="Drop Partial ORFs", description="Require Prodigal partial=00."
    )
    segmasker_threshold: float = ConfigField(
        default=0.1,
        ge=0.0,
        le=1.0,
        title="Low-Complexity Fraction",
        description="Maximum tolerated fraction of segmasker-masked residues.",
    )
    min_surviving_proteins: int = ConfigField(
        default=1,
        ge=1,
        title="Minimum Surviving Proteins",
        description="Proteins a generation must yield to pass this filter.",
    )
    require_prompt_dna: str | None = ConfigField(
        default=None,
        title="Require Prompt DNA in ORF",
        description="Keep only ORFs whose nucleotide sequence contains this prompt DNA.",
    )


def is_highly_repetitive(sequence: str) -> bool:
    """Return True when repeated k-mers cover more than 30% of the sequence.

    Mirrors ``filter_protein_fasta.is_highly_repetitive``: for each k in
    3..9, the most common k-mer's count times k is compared against
    ``0.3 * len(sequence)``.
    """
    length = len(sequence)
    if length < REPEAT_MIN_K:
        return False
    residues = np.array(list(sequence))
    for k in range(REPEAT_MIN_K, REPEAT_MIN_K + REPEAT_K_SPAN):
        if k > length:
            break
        kmers = ["".join(window) for window in np.lib.stride_tricks.sliding_window_view(residues, k)]
        if kmers and max(Counter(kmers).values()) * k > length * REPEAT_FRACTION:
            return True
    return False


def has_underrepresented_amino_acids(sequence: str) -> bool:
    """Return True when the rarest 30% of observed residues all occur < 2 times.

    Mirrors ``filter_protein_fasta.is_underrepresented_amino_acids``.
    """
    counts = Counter(sequence)
    total_unique = len(counts)
    if total_unique == 0:
        return True
    sorted_counts = sorted(counts.values(), reverse=True)
    num_bottom = max(1, int(UNDERREPRESENTED_FRACTION * total_unique))
    return all(count < UNDERREPRESENTED_COUNT for count in sorted_counts[-num_bottom:])


def _strip_stop(sequence: str) -> str:
    """Drop Prodigal's trailing stop-codon asterisk, if present."""
    return sequence[:-1] if sequence.endswith("*") else sequence


def _call_orfs(dna_sequences: list[str]) -> list[list[Any]]:
    """Run Prodigal in meta mode over a batch of generated DNA sequences."""
    result = run_prodigal_prediction(
        inputs=ProdigalInput(input_sequences=dna_sequences),
        config=ProdigalConfig(meta_mode=True),
    )
    return list(result.predicted_orfs)


def _low_complexity_fractions(proteins: list[str]) -> list[float]:
    """Return the segmasker-masked residue fraction for each protein."""
    if not proteins:
        return []
    result = run_segmasker(
        inputs=SegmaskerInput(sequences=proteins),
        config=SegmaskerConfig(
            window=SEGMASKER_WINDOW, locut=SEGMASKER_LOCUT, hicut=SEGMASKER_HICUT
        ),
    )
    if len(result.results) != len(proteins):
        raise RuntimeError(
            f"segmasker returned {len(result.results)} rows for {len(proteins)} proteins"
        )
    return [row.low_complexity_fraction for row in result.results]


@constraint(
    key="prodigal-protein-qc",
    label="Prodigal Protein QC",
    config=ProteinQCConfig,
    description="Call ORFs with Prodigal and keep generations whose proteins pass the published QC predicates.",
    tools_called=["prodigal-prediction", "segmasker-score"],
    category="protein_quality",
    supported_sequence_types=["dna"],
)
def prodigal_protein_qc_constraint(
    input_sequences: list[tuple[Any, ...]],
    config: ProteinQCConfig,
) -> list[ConstraintOutput]:
    """Call ORFs on generated DNA and keep proteins that pass the paper's QC.

    Args:
        input_sequences: One tuple per proposal, each holding a single DNA
            ``Sequence``.
        config: Length, partial-ORF, complexity, and survivor-count settings.

    Returns:
        One ``ConstraintOutput`` per proposal. Score is ``0.0`` when at least
        ``min_surviving_proteins`` proteins survive and ``1.0`` otherwise.
        Metadata carries ``qc_proteins`` (the survivors, each with
        ``protein_id``, ``sequence``, ``length``, ``strand``,
        ``nucleotide_sequence``, ``low_complexity_fraction``),
        ``qc_orfs`` (every called ORF with ``passed_qc`` and ``rejected_by``),
        ``qc_protein_count``, ``orf_count``, and ``qc_rejections`` (a count per
        failing predicate -- ``missing_prompt``, ``length``, ``partial``,
        ``repetitive``, ``low_diversity``, ``underrepresented_aas``,
        ``low_complexity`` -- so a run that filters everything is diagnosable).
    """
    dna_sequences = [str(seq_tuple[0].sequence) for seq_tuple in input_sequences]
    all_orfs = _call_orfs(dna_sequences)
    if len(all_orfs) != len(dna_sequences):
        raise RuntimeError(
            f"Prodigal returned {len(all_orfs)} ORF lists for {len(dna_sequences)} sequences"
        )

    outputs: list[ConstraintOutput] = []
    for orfs in all_orfs:
        rejections = Counter()
        candidates: list[dict[str, Any]] = []
        # One row per ORF Prodigal called, carrying the predicate that
        # rejected it. Counts alone cannot tell you *which* ORF was dropped or
        # what it looked like, which is the first thing you need when a
        # generation yields nothing.
        verdicts: list[dict[str, Any]] = []

        def _reject(orf: Any, protein: str, reason: str) -> None:
            rejections[reason] += 1
            verdicts.append({
                "protein_id": orf.orf_id, "length": len(protein), "strand": orf.strand,
                "passed_qc": False, "rejected_by": reason, "sequence": protein,
            })

        prompt_dna = (config.require_prompt_dna or "").upper()
        for orf in orfs:
            if prompt_dna and prompt_dna not in orf.nucleotide_sequence.upper():
                _reject(orf, _strip_stop(orf.amino_acid_sequence), "missing_prompt")
                continue
            protein = _strip_stop(orf.amino_acid_sequence)
            if not (config.min_length <= len(protein) <= config.max_length):
                _reject(orf, protein, "length")
                continue
            if config.filter_partial:
                partial_begin = orf.metrics.get("partial_begin", 0)
                partial_end = orf.metrics.get("partial_end", 0)
                if partial_begin or partial_end:
                    _reject(orf, protein, "partial")
                    continue
            if is_highly_repetitive(protein):
                _reject(orf, protein, "repetitive")
                continue
            if len(set(protein)) < MIN_UNIQUE_AMINO_ACIDS:
                _reject(orf, protein, "low_diversity")
                continue
            if has_underrepresented_amino_acids(protein):
                _reject(orf, protein, "underrepresented_aas")
                continue
            candidates.append(
                {
                    "protein_id": orf.orf_id,
                    "sequence": protein,
                    "length": len(protein),
                    "strand": orf.strand,
                    "nucleotide_sequence": orf.nucleotide_sequence,
                }
            )

        # segmasker is the only subprocess check, so it runs last and in one batch.
        fractions = _low_complexity_fractions([entry["sequence"] for entry in candidates])
        survivors: list[dict[str, Any]] = []
        for entry, fraction in zip(candidates, fractions, strict=True):
            if fraction > config.segmasker_threshold:
                rejections["low_complexity"] += 1
                verdicts.append({
                    "protein_id": entry["protein_id"], "length": entry["length"],
                    "strand": entry["strand"], "passed_qc": False,
                    "rejected_by": "low_complexity", "sequence": entry["sequence"],
                })
                continue
            entry["low_complexity_fraction"] = fraction
            survivors.append(entry)
            verdicts.append({
                "protein_id": entry["protein_id"], "length": entry["length"],
                "strand": entry["strand"], "passed_qc": True,
                "rejected_by": None, "sequence": entry["sequence"],
            })

        passed = len(survivors) >= config.min_surviving_proteins
        outputs.append(
            ConstraintOutput(
                score=0.0 if passed else 1.0,
                metadata={
                    "qc_proteins": survivors or None,
                    "qc_orfs": verdicts or None,
                    "qc_protein_count": len(survivors),
                    "orf_count": len(orfs),
                    "qc_rejections": dict(rejections) or None,
                },
            )
        )

    return outputs
