"""Optional profile-HMM filter over QC-surviving proteins.

Sits between ``prodigal_protein_qc`` and the AlphaFold 3 screen: it keeps
generations that produced at least one protein resembling a known toxin or
antitoxin family, so expensive folding is spent on plausible candidates.

Deliberately permissive by default. A high E-value cutoff is the point: the
paper's value is finding *divergent* sequences, and a strict cutoff would
select for near-copies of the training families and defeat that. Treat this
as "does anything here look TA-like at all", not as an annotation.

proto's own ``protein-profile-hmm`` constraint is not used because it scores
the single longest canonical ORF of a DNA segment. A Type II TA generation is
expected to yield *two* proteins from one locus, and either could carry the
hit, so every QC survivor is searched instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput
from proto_language.utils.base import BaseConfig, ConfigField
from proto_tools import PyHmmscanConfig, PyHmmscanInput
from proto_tools.tools.gene_annotation.pyhmmer.hmmscan import run_pyhmmer_hmmscan

logger = logging.getLogger(__name__)


class ProfileHMMFilterConfig(BaseConfig):
    """Configuration for :func:`profile_hmm_filter_constraint`.

    Attributes:
        hmm_path: HMMER3 profile database to scan against (e.g. Pfam TA
            families). Required.
        qc_constraint_label: Upstream constraint supplying ``qc_proteins``.
        evalue_threshold: Sequence-level E-value cap. Permissive by design;
            tighten only if you are willing to select against divergence.
        required_profiles: Optional profile names that must be among the hits.
            Empty means any hit counts.
        min_matching_proteins: Proteins in the generation that must carry a
            hit for the proposal to pass.
        annotate_only: Record hits on the metadata but never reject. Use this
            to measure what the filter *would* do before enabling it.
    """

    hmm_path: str = ConfigField(
        title="Profile HMM Path", description="HMMER3 profile database to scan against."
    )
    qc_constraint_label: str = ConfigField(
        default="protein_qc",
        title="QC Constraint Label",
        description="Upstream constraint supplying qc_proteins metadata.",
    )
    evalue_threshold: float = ConfigField(
        default=1.0,
        gt=0.0,
        title="E-value Threshold",
        description="Sequence-level E-value cap; permissive by default to retain divergent hits.",
    )
    required_profiles: list[str] = ConfigField(
        default=[],
        title="Required Profiles",
        description="Profile names that must be hit; empty means any hit counts.",
    )
    min_matching_proteins: int = ConfigField(
        default=1, ge=1, title="Minimum Matching Proteins", description="Proteins needing a hit."
    )
    annotate_only: bool = ConfigField(
        default=False,
        title="Annotate Only",
        description="Record hits without rejecting, to measure the filter before enabling it.",
    )


def _scan(sequence: str, hmm_path: str, evalue: float) -> list[dict[str, Any]]:
    """Scan one protein against the profile database.

    One call per sequence: ``SequenceHit`` carries ``query_idx`` but no
    target index, so batching would leave hit-to-protein mapping ambiguous.
    pyhmmer runs in-process and is fast enough that this is not the
    bottleneck next to AlphaFold 3.

    Args:
        sequence: Protein sequence to scan.
        hmm_path: HMMER3 profile database.
        evalue: Sequence-level E-value cap.

    Returns:
        Hits as ``{profile, evalue, score}``, best (lowest E-value) first.
    """
    result = run_pyhmmer_hmmscan(
        PyHmmscanInput(sequences=[sequence], hmm_db=hmm_path),
        PyHmmscanConfig(evalue_threshold=evalue, domain_evalue_threshold=evalue),
    )
    # For hmmscan the HMM profile is the TARGET and the input sequence is the
    # query, so the profile name lives in target_name; query_name is just the
    # sequence's index. Reading query_name yields "0" and silently breaks
    # required_profiles matching.
    hits = [
        {"profile": hit.target_name, "evalue": hit.evalue, "score": hit.score}
        for hit in result.sequence_hits
    ]
    return sorted(hits, key=lambda h: h["evalue"])


@constraint(
    key="ta-profile-hmm-filter",
    label="TA Profile HMM Filter",
    config=ProfileHMMFilterConfig,
    description="Scan every QC-surviving protein against a profile-HMM set; gate the generation, not the protein.",
    tools_called=["pyhmmer-hmmscan"],
    category="protein_quality",
    supported_sequence_types=["dna"],
)
def profile_hmm_filter_constraint(
    input_sequences: list[tuple[Any, ...]],
    config: ProfileHMMFilterConfig,
) -> list[ConstraintOutput]:
    """Keep generations whose QC-surviving proteins hit a TA profile.

    Args:
        input_sequences: One tuple per proposal holding a DNA ``Sequence``
            that already carries QC metadata.
        config: Profile database, E-value cap and pass criteria.

    Returns:
        One ``ConstraintOutput`` per proposal. Score ``0.0`` when at least
        ``min_matching_proteins`` carry a qualifying hit (or always, under
        ``annotate_only``). Metadata carries ``hmm_hits`` (per protein, best
        hit first), ``hmm_matching_proteins`` and ``hmm_profiles_found``.

    Raises:
        FileNotFoundError: If ``hmm_path`` does not exist.
    """
    hmm_path = Path(config.hmm_path)
    if not hmm_path.exists():
        raise FileNotFoundError(f"Profile HMM database not found: {hmm_path}")
    required = set(config.required_profiles)

    outputs: list[ConstraintOutput] = []
    for sequence_tuple in input_sequences:
        sequence = sequence_tuple[0]
        qc_entry = sequence._constraints_metadata.get(config.qc_constraint_label)
        if qc_entry is None:
            raise KeyError(
                f"profile_hmm_filter: no metadata from {config.qc_constraint_label!r}; "
                "declare the QC constraint before this filter."
            )
        proteins: list[dict[str, Any]] = qc_entry.get("data", {}).get("qc_proteins") or []

        per_protein: list[dict[str, Any]] = []
        matching = 0
        profiles_found: set[str] = set()
        for protein in proteins:
            hits = _scan(protein["sequence"], str(hmm_path), config.evalue_threshold)
            qualifying = [h for h in hits if not required or h["profile"] in required]
            if qualifying:
                matching += 1
                profiles_found.update(h["profile"] for h in qualifying)
            per_protein.append(
                {
                    "protein_id": protein["protein_id"],
                    "hits": hits or None,
                    "best_profile": hits[0]["profile"] if hits else None,
                    "best_evalue": hits[0]["evalue"] if hits else None,
                    "qualifies": bool(qualifying),
                }
            )

        passed = config.annotate_only or matching >= config.min_matching_proteins
        outputs.append(
            ConstraintOutput(
                score=0.0 if passed else 1.0,
                metadata={
                    "hmm_hits": per_protein or None,
                    "hmm_matching_proteins": matching,
                    "hmm_profiles_found": sorted(profiles_found) or None,
                    "hmm_annotate_only": config.annotate_only,
                },
            )
        )
    return outputs
