"""Intra-generation cofolding as a proto constraint.

Cofolding is expressed
as a ``Constraint`` it joins the same ordered filter chain as QC, the profile
HMM and the monomer screen, which buys three things: the optimizer's
short-circuiting means only generations that already produced two confident
proteins are ever cofolded; pairs and their interface scores live on the
proposal's metadata rather than in an intermediate file; and the whole design
runs as one ``Program``.

Pairing is intra-generation, the paper's rule: two proteins pair when Prodigal
called them on the same Evo generation. That is co-occurrence, not evidence of
interaction. Optionally each protein is also paired against fixed target
sequences, which asks the different question "does any generated protein bind
*this* one?".

The gates (pDockQ2, ipTM, pLDDT) filter for a plausible predicted complex.
They do not identify functional pairs -- see the calibration results in
``proto_pipelines/README.md``.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import pandas as pd
from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput
from proto_language.utils.base import BaseConfig, ConfigField

from proto_pipelines.af3 import AlphaFold3RunConfig, job_name_for, score_complex
from proto_pipelines.identity import pairwise_identity

logger = logging.getLogger(__name__)

MIN_TOTAL_RESIDUES = 100
MAX_TOTAL_RESIDUES = 1024


class TACofoldConfig(BaseConfig):
    """Configuration for :func:`ta_cofold_constraint`.

    Attributes:
        source_constraint_label: Constraint whose metadata supplies the
            proteins to pair. Defaults to the AlphaFold 3 monomer screen;
            point it at the QC constraint when folding is disabled.
        source_metadata_key: Key within that metadata holding the protein
            list.
        max_pair_identity: Drop a pair whose members exceed this percent
            identity (MAFFT, ungapped-column definition). A TA pair is two
            different proteins; near-identical members are usually one gene
            called twice.
        pdockq2_threshold: pDockQ2 floor. 0.23 is the DockQ "Acceptable"
            class boundary.
        min_iptm: AlphaFold 3 ipTM floor.
        min_avg_plddt: AlphaFold 3 pLDDT floor; a non-binding guard at the
            calibrated value.
        pdockq2_distance_cutoff: CA-CA cutoff for pDockQ2's interface.
        min_passing_pairs: Pairs that must clear every gate for the
            generation to pass.
        max_pairs_per_proposal: Cap on pairs cofolded per generation,
            shortest first. Pair count grows quadratically in proteins, and
            each fold is minutes. ``0`` disables the cap.
        target_sequences: Optional ``{name: sequence}`` folded against every
            candidate protein in addition to intra-generation pairs.
        alphafold3: AlphaFold 3 execution settings; MSA on by default because
            this is the scoring step.
    """

    source_constraint_label: str = ConfigField(
        default="af3_monomer_screen",
        title="Source Constraint",
        description="Supplies the proteins.",
    )
    source_metadata_key: str = ConfigField(
        default="af3_proteins",
        title="Source Metadata Key",
        description="Protein list key.",
    )
    mafft_threads: int = ConfigField(
        default=1,
        ge=1,
        title="MAFFT Threads",
        description="CPU threads for the pairwise identity alignments.",
    )
    max_pair_identity: float = ConfigField(
        default=70.0,
        ge=0.0,
        le=100.0,
        title="Max Pair Identity",
        description="Drop pairs whose members exceed this percent identity (MAFFT).",
    )
    pdockq2_threshold: float = ConfigField(
        default=0.23,
        title="pDockQ2 Threshold",
        description="DockQ 'Acceptable' class boundary.",
    )
    min_iptm: float = ConfigField(
        default=0.55, title="Minimum ipTM", description="AF3 ipTM floor."
    )
    min_avg_plddt: float = ConfigField(
        default=80.0,
        title="Minimum pLDDT",
        description="AF3 pLDDT floor (non-binding guard).",
    )
    pdockq2_distance_cutoff: float = ConfigField(
        default=8.0,
        gt=0.0,
        title="pDockQ2 Cutoff",
        description="CA-CA interface cutoff, angstroms.",
    )
    min_passing_pairs: int = ConfigField(
        default=1,
        ge=1,
        title="Minimum Passing Pairs",
        description="Pairs that must clear the gates.",
    )
    max_pairs_per_proposal: int = ConfigField(
        default=6,
        ge=0,
        title="Max Pairs Cofolded",
        description="Per-generation cap; 0 = no cap.",
    )
    target_sequences: dict[str, str] = ConfigField(
        default={},
        title="Target Sequences",
        description="Fixed targets to fold every protein against.",
    )
    alphafold3: AlphaFold3RunConfig = ConfigField(
        default_factory=lambda: AlphaFold3RunConfig(
            use_msa=True, pair_heterocomplex_msas=True
        ),
        title="AlphaFold 3 Settings",
        description="MSA on: this is the scoring step.",
    )


def _candidate_pairs(
    proteins: list[dict[str, Any]], config: TACofoldConfig
) -> tuple[list[dict[str, Any]], int, int]:
    """Enumerate intra-generation pairs plus any fixed-target pairs.

    Args:
        proteins: Surviving proteins for one generation.
        config: Pairing settings, including the MAFFT thread count.

    Returns:
        ``(pairs, dropped_similar, dropped_size)`` -- pairs inside the length
        window, how many were dropped for exceeding ``max_pair_identity``,
        and how many fell outside ``[MIN_TOTAL_RESIDUES, MAX_TOTAL_RESIDUES]``.
        Both drop counts are returned because a generation that produces no
        pairs is otherwise indistinguishable from one that produced none for
        a reason worth knowing.
    """
    pairs: list[dict[str, Any]] = []
    dropped = 0
    for first, second in itertools.combinations(proteins, 2):
        identity = pairwise_identity(
            first["sequence"], second["sequence"], config.mafft_threads
        )
        if identity > config.max_pair_identity:
            dropped += 1
            continue
        pairs.append(
            {
                "uid_1": first["protein_id"],
                "seq_1": first["sequence"],
                "uid_2": second["protein_id"],
                "seq_2": second["sequence"],
                "pair_identity": identity,
                "pair_source": "intra_generation",
            }
        )
    for name, target in config.target_sequences.items():
        for protein in proteins:
            pairs.append(
                {
                    "uid_1": name,
                    "seq_1": target,
                    "uid_2": protein["protein_id"],
                    "seq_2": protein["sequence"],
                    "pair_identity": None,
                    "pair_source": "set_target",
                }
            )

    sized = [
        p
        for p in pairs
        if MIN_TOTAL_RESIDUES <= len(p["seq_1"]) + len(p["seq_2"]) <= MAX_TOTAL_RESIDUES
    ]
    dropped_size = len(pairs) - len(sized)
    sized.sort(key=lambda p: len(p["seq_1"]) + len(p["seq_2"]))
    if config.max_pairs_per_proposal:
        sized = sized[: config.max_pairs_per_proposal]
    return sized, dropped, dropped_size


@constraint(
    key="ta-cofold",
    label="TA Pair Cofold",
    config=TACofoldConfig,
    description="Cofold intra-generation protein pairs with AlphaFold 3 and gate on pDockQ2/ipTM/pLDDT.",
    tools_called=["alphafold3-prediction", "pdockq2-score", "mafft-align"],
    category="protein_structure",
    supported_sequence_types=["dna"],
    uses_gpu=True,
)
def ta_cofold_constraint(
    input_sequences: list[tuple[Any, ...]],
    config: TACofoldConfig,
) -> list[ConstraintOutput]:
    """Cofold each generation's protein pairs and gate on interface confidence.

    Args:
        input_sequences: One tuple per proposal holding a DNA ``Sequence``
            carrying upstream protein metadata.
        config: Pairing, gating and AlphaFold 3 settings.

    Returns:
        One ``ConstraintOutput`` per proposal. Score ``0.0`` when at least
        ``min_passing_pairs`` clear every gate. Metadata carries
        ``cofold_pairs`` -- every pair scored, each with ``pdockq2``,
        ``iptm``, ``ptm``, ``avg_plddt``, the reported-only ``pdockq_v1`` and
        a ``passed_gates`` flag -- plus ``cofold_pair_count``,
        ``cofold_passing_count`` and ``cofold_dropped_similar``.

    Raises:
        KeyError: If the source constraint produced no metadata, which means
            it was not declared before this one.
    """
    outputs: list[ConstraintOutput] = []
    for sequence_tuple in input_sequences:
        sequence = sequence_tuple[0]
        entry = sequence._constraints_metadata.get(config.source_constraint_label)
        if entry is None:
            raise KeyError(
                f"ta_cofold: no metadata from {config.source_constraint_label!r}; "
                "declare that constraint before this one."
            )
        proteins = entry.get("data", {}).get(config.source_metadata_key) or []

        pairs, dropped, dropped_size = _candidate_pairs(proteins, config)
        scored: list[dict[str, Any]] = []
        passing = 0
        for pair in pairs:
            job = job_name_for("cofold", [pair["seq_1"], pair["seq_2"]])
            metrics = score_complex(
                chain_a=pair["seq_1"],
                chain_b=pair["seq_2"],
                job_name=job,
                run_config=config.alphafold3,
                pdockq2_distance_cutoff=config.pdockq2_distance_cutoff,
            )
            ok = (
                (metrics.get("pdockq2") or 0.0) >= config.pdockq2_threshold
                and (metrics.get("iptm") or 0.0) >= config.min_iptm
                and (metrics.get("avg_plddt") or 0.0) >= config.min_avg_plddt
            )
            passing += ok
            scored.append({**pair, **metrics, "passed_gates": ok})

        if dropped:
            logger.info(
                "ta_cofold: dropped %d pair(s) above %.0f%% identity",
                dropped,
                config.max_pair_identity,
            )
        if dropped_size:
            logger.info(
                "ta_cofold: dropped %d pair(s) outside %d-%d total residues",
                dropped_size,
                MIN_TOTAL_RESIDUES,
                MAX_TOTAL_RESIDUES,
            )
        if proteins and not pairs:
            print(
                f"WARNING: ta_cofold had {len(proteins)} protein(s) but produced no "
                f"pairs to score ({dropped} too similar, {dropped_size} outside "
                f"{MIN_TOTAL_RESIDUES}-{MAX_TOTAL_RESIDUES} residues)."
            )
        outputs.append(
            ConstraintOutput(
                score=0.0 if passing >= config.min_passing_pairs else 1.0,
                metadata={
                    "cofold_pairs": scored or None,
                    "cofold_pair_count": len(scored),
                    "cofold_passing_count": passing,
                    "cofold_dropped_similar": dropped,
                    "cofold_dropped_size": dropped_size,
                },
            )
        )
    return outputs


def apply_novelty_filter(
    scored: pd.DataFrame, mmseqs_db: str, max_identity: float
) -> pd.DataFrame:
    """Flag pairs whose members are too similar to known proteins.

    A structurally confident complex built from near-copies of natural
    proteins is not a novel design. Both chains are searched, and the pair's
    novelty is governed by the *less* novel member.

    Args:
        scored: Scored pairs with ``sequence_1`` / ``sequence_2``.
        mmseqs_db: FASTA or MMseqs2 database of known proteins.
        max_identity: Maximum tolerated percent identity to the top hit.

    Returns:
        ``scored`` with ``top_identity_1``, ``top_identity_2``,
        ``max_top_identity`` and a boolean ``is_novel`` column added.
    """
    from proto_tools import (
        Mmseqs2SearchProteinsConfig,
        Mmseqs2SearchProteinsInput,
        run_mmseqs2_search_proteins,
    )

    unique = sorted({*scored["sequence_1"], *scored["sequence_2"]})
    logger.info("Novelty search: %d unique chain(s) vs %s", len(unique), mmseqs_db)
    result = run_mmseqs2_search_proteins(
        Mmseqs2SearchProteinsInput(query_sequences=unique),
        Mmseqs2SearchProteinsConfig(target_db=mmseqs_db),
    )
    # No hit means nothing similar was found, i.e. maximally novel.
    best: dict[str, float] = {}
    for sequence, hits in zip(unique, result.results, strict=True):
        best[sequence] = max((h.pident for h in hits), default=0.0)

    out = scored.copy()
    out["top_identity_1"] = out["sequence_1"].map(best)
    out["top_identity_2"] = out["sequence_2"].map(best)
    out["max_top_identity"] = out[["top_identity_1", "top_identity_2"]].max(axis=1)
    out["is_novel"] = out["max_top_identity"] < max_identity
    logger.info(
        "Novelty: %d/%d pair(s) below %.0f%% identity to the reference set",
        int(out["is_novel"].sum()),
        len(out),
        max_identity,
    )
    return out
