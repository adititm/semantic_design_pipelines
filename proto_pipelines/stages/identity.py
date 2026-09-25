"""MAFFT identity metrics for the gene- and operon-completion evaluations.

The paper reports three different identity definitions and they are not
interchangeable, so all three are reproduced here exactly as
``pipelines/gene_completion.py`` computes them:

``pairwise_identity``
    matches / columns where **neither** sequence is gapped
    (``gene_completion.align_pair``). Used for reference matching.
``alignment_identity``
    matches / total alignment columns, gaps included
    (``gene_completion.calculate_sequence_identity_w_prompt``). Lower than
    ``pairwise_identity`` whenever the two sequences differ in length.
``non_prompt_identity``
    ``pairwise_identity`` restricted to alignment columns past the translated
    prompt region on both sequences
    (``gene_completion.calculate_non_prompt_sequence_identity``). This is the
    figure that speaks to what the model generated rather than what it copied.

Alignments run through proto-tools' MAFFT tool rather than a ``mafft``
subprocess, so no ``mafft_path`` is needed in the configs.
"""

from __future__ import annotations

from dataclasses import dataclass

from proto_tools import MafftConfig, MafftInput, run_mafft_align

# `mafft <file>` with no flags, which is what the published workflow invoked.
DEFAULT_ALIGN_METHOD = "auto"


@dataclass(frozen=True)
class ReferenceMatch:
    """Best-scoring reference for one query protein.

    Attributes:
        reference_id: FASTA ID of the closest reference.
        identity: Percent identity (0-100) under ``pairwise_identity``.
    """

    reference_id: str
    identity: float


def align_pair(query: str, reference: str, threads: int = 1) -> tuple[str, str]:
    """Align two sequences with MAFFT and return the aligned pair.

    Args:
        query: First sequence.
        reference: Second sequence.
        threads: MAFFT CPU threads.

    Returns:
        The two aligned sequences, gaps included.

    Raises:
        RuntimeError: If MAFFT returns anything other than two aligned rows.
    """
    result = run_mafft_align(
        MafftInput(sequences=[query, reference], sequence_ids=["query", "reference"]),
        MafftConfig(align_method=DEFAULT_ALIGN_METHOD, threads=threads),
    )
    aligned = result.msa.aligned_sequences
    if len(aligned) != 2:
        raise RuntimeError(f"MAFFT returned {len(aligned)} aligned rows; expected 2")
    return aligned[0], aligned[1]


def pairwise_identity(query: str, reference: str, threads: int = 1) -> float:
    """Percent identity over columns where neither sequence is gapped.

    Args:
        query: Query protein sequence.
        reference: Reference protein sequence.
        threads: MAFFT CPU threads.

    Returns:
        Percent identity in ``[0, 100]``; ``0.0`` when either input is empty
        or the alignment shares no ungapped column.
    """
    if not query or not reference:
        return 0.0
    aligned_query, aligned_reference = align_pair(query, reference, threads)
    matches = 0
    columns = 0
    for a, b in zip(aligned_query, aligned_reference, strict=True):
        if a == "-" or b == "-":
            continue
        columns += 1
        if a == b:
            matches += 1
    return (matches / columns) * 100.0 if columns else 0.0


def alignment_identity(query: str, reference: str, threads: int = 1) -> float:
    """Percent identity over every alignment column, gaps included.

    Args:
        query: Query protein sequence.
        reference: Reference protein sequence.
        threads: MAFFT CPU threads.

    Returns:
        Percent identity in ``[0, 100]``; ``0.0`` when either input is empty.
    """
    if not query or not reference:
        return 0.0
    aligned_query, aligned_reference = align_pair(query, reference, threads)
    matches = sum(a == b for a, b in zip(aligned_query, aligned_reference, strict=True))
    return (matches / len(aligned_query)) * 100.0 if aligned_query else 0.0


def non_prompt_identity(
    query: str, reference: str, prompt_amino_acids: str, threads: int = 1
) -> float:
    """Percent identity past the prompt region on both sequences.

    Args:
        query: Generated protein sequence, prompt region included.
        reference: Reference protein sequence.
        prompt_amino_acids: Translation of the DNA prompt, used only for its
            length.
        threads: MAFFT CPU threads.

    Returns:
        Percent identity in ``[0, 100]`` over ungapped columns after both
        sequences have consumed ``len(prompt_amino_acids)`` residues; ``0.0``
        when no such column exists.
    """
    prompt_length = len(prompt_amino_acids)
    if not query or not reference or prompt_length == 0:
        return 0.0

    aligned_query, aligned_reference = align_pair(query, reference, threads)
    consumed_query = 0
    consumed_reference = 0
    matches = 0
    columns = 0
    for a, b in zip(aligned_query, aligned_reference, strict=True):
        if a != "-":
            consumed_query += 1
        if b != "-":
            consumed_reference += 1
        if consumed_query <= prompt_length or consumed_reference <= prompt_length:
            continue
        if a == "-" or b == "-":
            continue
        columns += 1
        if a == b:
            matches += 1
    return (matches / columns) * 100.0 if columns else 0.0


def translate_prompt(prompt_dna: str) -> str:
    """Translate a DNA prompt in frame 1, trimming to a codon boundary.

    Mirrors the prompt translation in
    ``gene_completion.calculate_non_prompt_sequence_identity``.

    Args:
        prompt_dna: DNA prompt sequence.

    Returns:
        The translated prompt, including any stop-codon ``*`` characters.
    """
    from Bio.Seq import Seq

    remainder = len(prompt_dna) % 3
    trimmed = prompt_dna[:-remainder] if remainder else prompt_dna
    return str(Seq(trimmed).translate())


def best_reference_match(
    query: str, references: dict[str, str], threads: int = 1
) -> ReferenceMatch | None:
    """Return the reference with the highest ``pairwise_identity`` to ``query``.

    Args:
        query: Query protein sequence.
        references: Reference ID to sequence.
        threads: MAFFT CPU threads.

    Returns:
        The best match, or ``None`` when ``references`` is empty or every
        alignment scores 0.
    """
    best: ReferenceMatch | None = None
    for reference_id, reference_sequence in references.items():
        identity = pairwise_identity(query, reference_sequence, threads)
        if best is None or identity > best.identity:
            best = ReferenceMatch(reference_id=reference_id, identity=identity)
    if best is not None and best.identity <= 0.0:
        return None
    return best
