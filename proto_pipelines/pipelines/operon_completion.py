"""Operon-completion evaluation: does Evo produce the downstream genes.

Runs

    generate -> Prodigal + QC -> MAFFT identity against the expected operon gene

Unlike gene completion, the prompt is **not** prepended: the published workflow used
``make_fasta`` rather than ``make_gene_completion_fasta``, so ORFs are called
on the continuation alone. That is the right choice for the task -- the prompt
holds the upstream gene and the question is whether Evo produces the
downstream ones -- and it is preserved here.

Scoring follows ``process_operon_sequences``: for each generation, the protein
with the highest ungapped-column identity to the reference named by the
prompt's ``Expected_Response`` is kept, and identities are summarised per
(prompt, expected response). Like gene completion, this pipeline never used
ESMFold, so no structure prediction is performed.

Usage:
    python -m proto_pipelines.pipelines.operon_completion --config configs/operon_completion.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from proto_language.core import Constraint, Segment

from proto_pipelines.config import generation_settings, load_yaml, resolve_output_dir
from proto_pipelines.identity import best_reference_match, pairwise_identity
from proto_pipelines.prompts import Prompt, read_prompts
from proto_pipelines.qc import ProteinQCConfig, prodigal_protein_qc_constraint
from proto_pipelines.reporting import (
    write_stage_tables,
    accepted_proteins,
    write_fasta,
    write_filter_summary,
    write_hmm_hits,
    write_proposal_table,
    write_raw_metadata,
)
from proto_pipelines.runner import ProposalRecord, run_prompts

logger = logging.getLogger(__name__)

ALLOWED_KEYS = {
    "input_prompts",
    "reference_seqs",
    "output_dir",
    "output_root",
    "model_name",
    "model_local_path",
    "generator",
    "n_tokens",
    "temperature",
    "top_k",
    "batch_size",
    "n_sample_per_prompt",
    "seed",
    "device",
    "verbose",
    "filter_min_length",
    "filter_max_length",
    "filter_partial_bool",
    "segmasker_threshold",
    "run_identity_analysis",
    "seq_identity_match_threshold",
    "expected_response_column",
    "mafft_threads",
}

QC_LABEL = "protein_qc"


def read_reference_fasta(path: Path) -> dict[str, str]:
    """Load reference operon proteins keyed by record ID.

    Args:
        path: Reference protein FASTA. Record IDs must match the prompt CSV's
            ``Expected_Response`` values.

    Returns:
        Record ID to sequence.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the FASTA contains no records.
    """
    from Bio import SeqIO

    if not Path(path).exists():
        raise FileNotFoundError(f"Reference FASTA not found: {path}")
    references = {
        record.id: str(record.seq).replace("*", "")
        for record in SeqIO.parse(str(path), "fasta")
    }
    if not references:
        raise ValueError(f"Reference FASTA contained no records: {path}")
    return references


def build_constraint_chain(data: dict[str, Any]) -> Any:
    """Return a callback that builds this pipeline's ordered filter chain.

    Args:
        data: Parsed config mapping.

    Returns:
        A callable taking the run's ``Segment`` and ``Prompt``. This
        pipeline's filters do not depend on the prompt.
    """
    qc_config = ProteinQCConfig(
        min_length=int(data.get("filter_min_length", 50)),
        max_length=int(data.get("filter_max_length", 1200)),
        filter_partial=bool(data.get("filter_partial_bool", False)),
        segmasker_threshold=float(data.get("segmasker_threshold", 0.1)),
        min_surviving_proteins=1,
    )

    def build(segment: Segment, prompt: Prompt) -> list[Constraint]:  # noqa: ARG001
        return [
            Constraint(
                inputs=[segment],
                function=prodigal_protein_qc_constraint,
                function_config=qc_config,
                label=QC_LABEL,
                threshold=0.5,
            )
        ]

    return build


def score_identities(
    records: list[ProposalRecord],
    prompts: list[Prompt],
    references: dict[str, str],
    identity_threshold: float,
    expected_response_column: str,
    threads: int,
) -> pd.DataFrame:
    """Score each generation's best protein against its expected operon gene.

    Args:
        records: Proposals from the run.
        prompts: The prompts, used to recover ``Expected_Response``.
        references: Reference ID to sequence.
        identity_threshold: Minimum best-match identity across all references
            for a protein to be considered at all
            (``align_and_save_closest_match``).
        expected_response_column: Prompt CSV column naming the expected gene.
        threads: MAFFT CPU threads.

    Returns:
        One row per generation that produced a scoreable protein. Empty with
        the correct columns otherwise.
    """
    columns = [
        "root_id",
        "prompt_id",
        "expected_response",
        "protein_uid",
        "sequence_identity",
        "best_reference_id",
        "best_reference_identity",
        "sequence",
    ]
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}

    by_root: dict[str, list[dict[str, Any]]] = {}
    for protein in accepted_proteins(records):
        by_root.setdefault(protein["root_id"], []).append(protein)

    rows: list[dict[str, Any]] = []
    for root_id, group in by_root.items():
        prompt = prompts_by_id[group[0]["prompt_id"]]
        expected = prompt.get(expected_response_column)
        reference = references.get(expected)
        if reference is None:
            print(
                f"WARNING: prompt {prompt.prompt_id}: no reference record named "
                f"{expected!r}; this generation is not scored."
            )
            continue

        best_row: dict[str, Any] | None = None
        for protein in group:
            match = best_reference_match(protein["sequence"], references, threads)
            if match is None or match.identity < identity_threshold:
                continue
            identity = pairwise_identity(protein["sequence"], reference, threads)
            if best_row is None or identity > best_row["sequence_identity"]:
                best_row = {
                    "root_id": root_id,
                    "prompt_id": prompt.prompt_id,
                    "expected_response": expected,
                    "protein_uid": protein["protein_uid"],
                    "sequence_identity": identity,
                    "best_reference_id": match.reference_id,
                    "best_reference_identity": match.identity,
                    "sequence": protein["sequence"],
                }
        if best_row is not None:
            rows.append(best_row)

    return pd.DataFrame(rows, columns=columns)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate identity per prompt and expected operon gene.

    Args:
        results: Output of :func:`score_identities`.

    Returns:
        Mean, standard deviation, and count of ``sequence_identity`` per
        ``(prompt_id, expected_response)`` group.
    """
    if results.empty:
        return pd.DataFrame(
            columns=["prompt_id", "expected_response", "avg_identity", "std_identity", "count"]
        )
    grouped = results.groupby(["prompt_id", "expected_response"], dropna=False).agg(
        avg_identity=("sequence_identity", "mean"),
        std_identity=("sequence_identity", "std"),
        count=("sequence_identity", "size"),
    )
    return grouped.reset_index().round(2)


def run_pipeline(config_path: Path, output_root: str | None = None) -> None:
    """Generate operon completions and score them against the references.

    Args:
        config_path: Path to the pipeline YAML config.
        output_root: Overrides where the run writes; see
            :func:`proto_pipelines.config.resolve_output_dir` for precedence.

    Returns:
        None. Writes ``generated_sequences.csv``, ``filter_summary.csv``,
        ``proteins.fasta``, ``identity_results.csv`` and
        ``summary_statistics.csv`` into the configured ``output_dir``.
    """
    data = load_yaml(config_path, allowed_keys=ALLOWED_KEYS)
    output_dir = resolve_output_dir(data, cli_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(data["input_prompts"])
    settings = generation_settings(data, prepend_prompt=False)
    logger.info(
        "Operon completion: %d prompt(s) x %d sample(s) with %s",
        len(prompts),
        settings.n_sample_per_prompt,
        settings.model_checkpoint,
    )

    records = run_prompts(
        prompts,
        settings,
        build_constraint_chain(data),
        checkpoint_path=output_dir / "checkpoint.jsonl",
    )

    write_proposal_table(
        records,
        output_dir / "generated_sequences.csv",
        extra_columns={
            "model": settings.model_checkpoint,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
        },
    )
    summary = write_filter_summary(records, [QC_LABEL], output_dir / "filter_summary.csv")
    write_hmm_hits(records, output_dir / "hmm_hits.csv")
    write_raw_metadata(records, output_dir / "raw_metadata.json")
    stages = write_stage_tables(records, output_dir)
    logger.info("Stage tables: %s", ", ".join(f"{k}={v}" for k, v in stages.items()))
    logger.info("Filter summary:\n%s", summary.to_string(index=False))

    proteins = accepted_proteins(records)
    write_fasta(
        ((protein["protein_uid"], protein["sequence"]) for protein in proteins),
        output_dir / "proteins.fasta",
    )

    if not data.get("run_identity_analysis", True):
        logger.info("run_identity_analysis is false; stopping after generation.")
        return

    references = read_reference_fasta(Path(data["reference_seqs"]))
    results = score_identities(
        records,
        prompts,
        references,
        identity_threshold=float(data.get("seq_identity_match_threshold", 30.0)),
        expected_response_column=data.get("expected_response_column", "Expected_Response"),
        threads=int(data.get("mafft_threads", 1)),
    )
    results.to_csv(output_dir / "identity_results.csv", index=False)
    summarize(results).to_csv(output_dir / "summary_statistics.csv", index=False)

    logger.info("Scored %d generation(s); outputs in %s", len(results), output_dir)
    if results.empty:
        print(
            "WARNING: no generation cleared the identity threshold; "
            "identity_results.csv is empty."
        )


def main() -> None:
    """Parse ``--config`` and run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to the pipeline YAML config.")
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Directory to write this run under; output_dir from the config is "
            "appended to it. Defaults to $PROTO_PIPELINES_OUTPUT_ROOT, then "
            "output_root in the config, then <repo>/outputs."
        ),
    )
    args = parser.parse_args()
    run_pipeline(Path(args.config), output_root=args.output_root)


if __name__ == "__main__":
    main()
