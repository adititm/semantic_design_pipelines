"""Output tables written by every pipeline.

``generated_sequences.csv``
    One row per generated sequence, accepted or not, with the filter that
    rejected it and per-stage summary counts. Records excluded at
    intermediate stages are retained rather than dropped.

``filter_summary.csv``
    Sequences evaluated and passed per filter, in filter order. A run whose
    yield is zero is diagnosable from this table alone.

``orfs.csv`` / ``qc_proteins.csv`` / ``hmm_proteins.csv``
    One table per filtering stage: every called ORF with the predicate that
    rejected it, the QC survivors, and the per-protein HMM verdict.

``hmm_hits.csv``
    Every profile-HMM hit, per protein, with its family and E-value.

``af3_fold_scores.csv``
    Every AlphaFold 3 prediction, passing or not. This is the table to set
    ``af3_plddt_threshold`` / ``af3_ptm_threshold`` from for your own
    generations.

``raw_metadata.json``
    Everything every filter recorded, for every sequence. The CSVs above are
    curated views and can lag the filters; this is the complete record.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from proto_pipelines.runner import ProposalRecord


def write_proposal_table(
    records: list[ProposalRecord],
    path: Path,
    extra_columns: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Write one row per proposal, accepted and rejected alike.

    Args:
        records: Proposals from :func:`proto_pipelines.runner.run_prompts`.
        path: Destination CSV.
        extra_columns: Constant columns stamped onto every row (for example
            the model checkpoint and sampling temperature), so a results
            directory is self-describing.

    Returns:
        The DataFrame that was written.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        qc = record.data("protein_qc")
        af3 = record.data("af3_monomer_screen")
        hmm = record.data("profile_hmm")
        cof = record.data("ta_cofold")
        rows.append(
            {
                "prompt_id": record.prompt_id,
                "proposal_index": record.proposal_index,
                "outcome": record.outcome,
                "accepted": record.accepted,
                "energy": record.energy,
                "evo_score": record.evo_score,
                "dna_length": len(record.dna),
                "orf_count": qc.get("orf_count"),
                "qc_protein_count": qc.get("qc_protein_count"),
                "qc_rejections": qc.get("qc_rejections"),
                "hmm_matching_proteins": hmm.get("hmm_matching_proteins"),
                "hmm_profiles_found": hmm.get("hmm_profiles_found"),
                "af3_folded_count": af3.get("af3_folded_count"),
                "af3_protein_count": af3.get("af3_protein_count"),
                "cofold_pair_count": cof.get("cofold_pair_count"),
                "cofold_passing_count": cof.get("cofold_passing_count"),
                "cofold_dropped_similar": cof.get("cofold_dropped_similar"),
                "dna": record.dna,
                **(extra_columns or {}),
            }
        )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def write_filter_summary(
    records: list[ProposalRecord], filter_labels: list[str], path: Path
) -> pd.DataFrame:
    """Write per-filter evaluated/passed counts in filter order.

    A proposal counts as *evaluated* by a filter when that filter wrote
    metadata for it; the optimizer only evaluates a filter on proposals that
    cleared every earlier one, so the evaluated counts fall monotonically down
    the chain and show where candidates were lost.

    A proposal can also be dropped by something that is not a filter -- the
    optimizer evicts proposals that pass everything but fall outside the top
    ``num_results`` by energy, recording ``did_not_enter_top_k``. Those get
    their own row so the per-row rejections always sum to ``TOTAL``.

    Args:
        records: Proposals from the run.
        filter_labels: Filter labels in the order the optimizer applied them.
        path: Destination CSV.

    Returns:
        The DataFrame that was written.
    """
    rejected_by = Counter(record.outcome for record in records if not record.accepted)
    evaluated = Counter()
    for record in records:
        for label in filter_labels:
            if label in record.constraint_data:
                evaluated[label] += 1

    rows = []
    for label in filter_labels:
        n_evaluated = evaluated[label]
        n_rejected = rejected_by.get(label, 0)
        rows.append(
            {
                "filter": label,
                "evaluated": n_evaluated,
                "rejected": n_rejected,
                "passed": n_evaluated - n_rejected,
            }
        )

    for outcome in sorted(set(rejected_by) - set(filter_labels)):
        rows.append(
            {
                "filter": f"(non-filter) {outcome}",
                "evaluated": rejected_by[outcome],
                "rejected": rejected_by[outcome],
                "passed": 0,
            }
        )

    rows.append(
        {
            "filter": "TOTAL",
            "evaluated": len(records),
            "rejected": sum(1 for record in records if not record.accepted),
            "passed": sum(1 for record in records if record.accepted),
        }
    )
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def write_fasta(entries: Iterable[tuple[str, str]], path: Path) -> int:
    """Write ``(header, sequence)`` pairs as a FASTA file.

    Args:
        entries: Header text (without ``>``) and sequence, in output order.
        path: Destination FASTA.

    Returns:
        Number of records written. Zero produces an empty file and a warning,
        rather than no file at all, so a downstream step fails loudly on empty
        input instead of on a missing path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for header, sequence in entries:
            handle.write(f">{header}\n{sequence}\n")
            count += 1
    if count == 0:
        print(f"WARNING: no records written to {path}; the file is empty.")
    return count


def folded_proteins(records: list[ProposalRecord]) -> list[dict[str, Any]]:
    """Flatten every AlphaFold 3 prediction in the run, passing or not.

    The README tells users to fold first and set the pLDDT/pTM cutoffs from
    what they observe, because the paper's ESMFold numbers do not transfer.
    That is only possible if the failures are kept: this returns every folded
    protein from every proposal -- including proposals a later filter rejected
    -- each carrying ``passed_af3_screen``.

    Args:
        records: Proposals from the run.

    Returns:
        One row per AlphaFold 3 prediction, with ``root_id``, ``protein_uid``,
        the confidences, and the pass/fail flag.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        root_id = f"{record.prompt_id}_{record.proposal_index}"
        screen = record.data("af3_monomer_screen")
        for protein in screen.get("af3_folded_proteins") or []:
            rows.append(
                {
                    "root_id": root_id,
                    "prompt_id": record.prompt_id,
                    "proposal_index": record.proposal_index,
                    "proposal_outcome": record.outcome,
                    "protein_uid": f"{root_id}_{protein['protein_id']}",
                    **protein,
                }
            )
    return rows


def write_fold_scores(records: list[ProposalRecord], path: Path) -> pd.DataFrame:
    """Write every AlphaFold 3 confidence in the run to a CSV.

    Args:
        records: Proposals from the run.
        path: Destination CSV.

    Returns:
        The DataFrame that was written, sorted by pLDDT descending. Empty (but
        still written) when no structure prediction ran.
    """
    frame = pd.DataFrame(folded_proteins(records))
    if not frame.empty and "avg_plddt" in frame.columns:
        frame = frame.sort_values("avg_plddt", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def cofold_pairs(records: list[ProposalRecord]) -> pd.DataFrame:
    """Flatten every cofolded pair out of the proposal metadata.

    Pairs now live on the proposal rather than in an intermediate CSV, so
    this is the reporting path for the cofold stage. Pairs from rejected
    proposals are included too: a pair that was scored but failed the gates
    is exactly what you need to see when nothing survives.

    Args:
        records: Proposals from the run.

    Returns:
        One row per scored pair with its interface metrics and
        ``passed_gates``, sorted by pDockQ2 descending. Empty when no
        cofolding ran.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        root_id = f"{record.prompt_id}_{record.proposal_index}"
        for pair in record.data("ta_cofold").get("cofold_pairs") or []:
            rows.append({
                "root_id": root_id,
                "prompt_id": record.prompt_id,
                "proposal_outcome": record.outcome,
                **pair,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty and "pdockq2" in frame.columns:
        frame = frame.sort_values("pdockq2", ascending=False)
    return frame



def write_stage_tables(records: list[ProposalRecord], output_dir: Path) -> dict[str, int]:
    """Write one CSV per filtering stage, so each step is inspectable on its own.

    The end-of-run tables describe survivors and folds; they do not show what
    a stage *removed*. These do:

    ``orfs.csv``
        Every ORF Prodigal called, with ``passed_qc`` and the predicate in
        ``rejected_by``. This is the only place a QC-rejected ORF's sequence
        survives.
    ``qc_proteins.csv``
        The QC survivors handed to the next stage.
    ``hmm_proteins.csv``
        Per protein: whether it qualified, and its best profile and E-value.

    Every table covers rejected proposals as well as accepted ones, since a
    run that yields nothing is exactly when they are needed. A table whose
    stage did not run is written empty rather than skipped, so a missing file
    always means a bug and never "that stage was off".

    Args:
        records: Proposals from the run.
        output_dir: Directory to write into.

    Returns:
        ``{filename: row count}`` for what was written.
    """
    orfs: list[dict[str, Any]] = []
    qc_proteins: list[dict[str, Any]] = []
    hmm_proteins: list[dict[str, Any]] = []

    for record in records:
        root_id = f"{record.prompt_id}_{record.proposal_index}"
        common = {"root_id": root_id, "prompt_id": record.prompt_id,
                  "proposal_outcome": record.outcome}
        qc = record.data("protein_qc")
        for orf in qc.get("qc_orfs") or []:
            orfs.append({**common, **orf})
        for protein in qc.get("qc_proteins") or []:
            qc_proteins.append({**common, **protein})
        for protein in record.data("profile_hmm").get("hmm_hits") or []:
            hmm_proteins.append({
                **common,
                "protein_id": protein["protein_id"],
                "qualifies": protein.get("qualifies"),
                "best_profile": protein.get("best_profile"),
                "best_evalue": protein.get("best_evalue"),
                "n_hits": len(protein.get("hits") or []),
            })

    written: dict[str, int] = {}
    for name, rows in (("orfs.csv", orfs), ("qc_proteins.csv", qc_proteins),
                       ("hmm_proteins.csv", hmm_proteins)):
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        written[name] = len(rows)
    return written


def write_hmm_hits(records: list[ProposalRecord], path: Path) -> pd.DataFrame:
    """Write every profile-HMM hit, per protein.

    Which TA family a candidate resembles, and how strongly, is the whole
    point of running the filter -- it is what lets you tell a confident
    ParE_toxin match from a marginal HTH hit. Without this table the filter
    is a silent pass/fail.

    Args:
        records: Proposals from the run.
        path: Destination CSV.

    Returns:
        One row per (protein, hit), best E-value first. Empty (still written)
        when the HMM filter did not run.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        root_id = f"{record.prompt_id}_{record.proposal_index}"
        for protein in record.data("profile_hmm").get("hmm_hits") or []:
            for hit in protein.get("hits") or []:
                rows.append({
                    "root_id": root_id,
                    "proposal_outcome": record.outcome,
                    "protein_id": protein["protein_id"],
                    "qualifies": protein.get("qualifies"),
                    **hit,
                })
    frame = pd.DataFrame(rows)
    if not frame.empty and "evalue" in frame.columns:
        frame = frame.sort_values("evalue")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame



def write_acr_evidence(records: list[ProposalRecord], path: Path) -> pd.DataFrame:
    """Write the per-protein anti-CRISPR evidence table.

    One row per scored protein with every caller's score side by side, so a
    candidate that passed on one weak signal is distinguishable from one that
    several callers agree on. Rejected proposals are included.

    Args:
        records: Proposals from the run.
        path: Destination CSV.

    Returns:
        The DataFrame written, best ``acr_locus_score`` first. Empty (still
        written) when the Acr callers did not run.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        root_id = f"{record.prompt_id}_{record.proposal_index}"
        evidence = record.data("acr_evidence")
        for protein in evidence.get("acr_evidence") or []:
            rows.append({
                "root_id": root_id, "prompt_id": record.prompt_id,
                "proposal_outcome": record.outcome,
                # Locus-level context on every row, so a candidate can be read
                # without joining back to the proposal.
                "locus_has_acr_and_aca_like": evidence.get("locus_has_acr_and_aca_like"),
                **protein,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty and "acr_locus_score" in frame.columns:
        frame = frame.sort_values("acr_locus_score", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def write_raw_metadata(records: list[ProposalRecord], path: Path) -> int:
    """Dump every constraint's full metadata for every proposal, as JSON.

    The CSV tables are curated views and will always lag the constraints --
    the HMM filter shipped once with its hits computed and then discarded.
    This is the catch-all: whatever a constraint recorded is on disk, for
    rejected proposals as well as accepted ones.

    Args:
        records: Proposals from the run.
        path: Destination JSON.

    Returns:
        Number of proposals written.
    """
    payload = [
        {
            "prompt_id": r.prompt_id,
            "proposal_index": r.proposal_index,
            "outcome": r.outcome,
            "accepted": r.accepted,
            "energy": r.energy,
            "evo_score": r.evo_score,
            "dna": r.dna,
            "constraints": r.constraint_data,
        }
        for r in records
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=str))
    return len(payload)


def accepted_proteins(records: list[ProposalRecord]) -> list[dict[str, Any]]:
    """Flatten the surviving proteins of accepted proposals into rows.

    The surviving set comes from the last stage that filtered proteins: the
    AlphaFold 3 screen when a pipeline folds (Type II TA, Acr), and the QC
    constraint when it does not (the completion pipelines, which never used
    ESMFold and so have no structure stage). Reading only the AF3 metadata
    would silently return nothing for the latter.

    Args:
        records: Proposals from the run.

    Returns:
        One row per surviving protein, carrying its ``root_id`` (the proposal
        it came from, this implementation's equivalent of the published workflow's ``Root_ID``),
        ``protein_uid``, sequence, QC fields, and -- when the pipeline folded
        -- the AlphaFold 3 confidences.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        if not record.accepted:
            continue
        root_id = f"{record.prompt_id}_{record.proposal_index}"
        survivors = record.data("af3_monomer_screen").get("af3_proteins")
        if survivors is None:
            survivors = record.data("protein_qc").get("qc_proteins")
        for protein in survivors or []:
            rows.append(
                {
                    "root_id": root_id,
                    "prompt_id": record.prompt_id,
                    "proposal_index": record.proposal_index,
                    "protein_uid": f"{root_id}_{protein['protein_id']}",
                    **protein,
                }
            )
    return rows
