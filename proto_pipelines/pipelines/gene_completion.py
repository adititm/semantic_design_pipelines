"""Gene-completion evaluation: how closely Evo completes a truncated gene.

Runs

    Evo 1.5 (prompt prepended) -> Prodigal + prompt-spanning ORF filter + QC
    -> MAFFT identity against the reference protein

This is an evaluation, not a design run: it measures how closely Evo's
completion of a truncated gene matches the real protein. It never used
ESMFold, so there is nothing for AlphaFold 3 to replace and no structure
prediction is performed.

Three identity definitions are reported, all reproduced from the published workflow and
all different (see ``proto_pipelines.identity``):

* ``best_reference_identity`` -- ungapped-column identity to the closest
  sequence in the reference FASTA, used with ``seq_identity_match_threshold``
  to decide whether a generation is scored at all
  (``align_and_save_closest_match``).
* ``full_sequence_identity`` -- ungapped-column identity to the reference
  named by the prompt's ``Protein_Label``.
* ``non_prompt_sequence_identity`` -- the same, restricted to the region past
  the prompt. This is the figure that reflects what the model generated rather
  than what it was given, and it is the one to read.

Usage:
    python -m proto_pipelines.pipelines.gene_completion --config configs/gene_completion.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from proto_language.core import Constraint, Segment

from proto_pipelines.config import generation_settings, load_yaml, resolve_output_dir
from proto_pipelines.identity import (
    best_reference_match,
    non_prompt_identity,
    pairwise_identity,
    translate_prompt,
)
from proto_pipelines.prompts import Prompt, read_prompts
from proto_pipelines.qc import ProteinQCConfig, prodigal_protein_qc_constraint
from proto_pipelines.reporting import (
    accepted_proteins,
    write_fasta,
    write_filter_summary,
    write_hmm_hits,
    write_proposal_table,
    write_raw_metadata,
    write_stage_tables,
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
    "protein_label_column",
    "mafft_threads",
}

QC_LABEL = "protein_qc"


def read_reference_fasta(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load reference proteins and build the label lookup used by the published workflow.

    Args:
        path: Reference protein FASTA.

    Returns:
        ``(by_id, by_label)``. ``by_id`` maps each record ID to its sequence.
        ``by_label`` maps lower-cased header tokens -- the record ID, the full
        description, and each whitespace-separated token -- to a sequence, so
        a prompt's ``Protein_Label`` (for example ``rpoS``) resolves. This
        mirrors ``gene_completion.build_reference_lookup``, first token wins.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the FASTA contains no records.
    """
    from Bio import SeqIO

    if not Path(path).exists():
        raise FileNotFoundError(f"Reference FASTA not found: {path}")

    by_id: dict[str, str] = {}
    by_label: dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        sequence = str(record.seq).replace("*", "")
        by_id[record.id] = sequence
        description = record.description.lower()
        keys = {record.id.lower(), description}
        keys.update(token.strip("[](),") for token in description.replace("/", " ").split())
        for key in keys:
            if key and key not in by_label:
                by_label[key] = sequence

    if not by_id:
        raise ValueError(f"Reference FASTA contained no records: {path}")
    return by_id, by_label


def build_constraint_chain(data: dict[str, Any]) -> Any:
    """Return a callback that builds this pipeline's ordered filter chain.

    Args:
        data: Parsed config mapping.

    Returns:
        A callable taking the run's ``Segment`` and ``Prompt``. The prompt is
        needed because each ORF must span the prompt DNA.
    """

    def build(segment: Segment, prompt: Prompt) -> list[Constraint]:
        qc_config = ProteinQCConfig(
            min_length=int(data.get("filter_min_length", 50)),
            max_length=int(data.get("filter_max_length", 1200)),
            filter_partial=bool(data.get("filter_partial_bool", False)),
            segmasker_threshold=float(data.get("segmasker_threshold", 0.1)),
            min_surviving_proteins=1,
            require_prompt_dna=prompt.sequence,
        )
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
    reference_by_id: dict[str, str],
    reference_by_label: dict[str, str],
    identity_threshold: float,
    protein_label_column: str,
    threads: int,
) -> pd.DataFrame:
    """Align each surviving protein to the references and score identity.

    Args:
        records: Proposals from the run.
        prompts: The prompts, used to recover per-prompt metadata.
        reference_by_id: Record ID to sequence, for the best-match search.
        reference_by_label: Label token to sequence, for the labelled reference.
        identity_threshold: Minimum best-match percent identity to score a
            protein at all.
        protein_label_column: Prompt CSV column naming the reference protein.
        threads: MAFFT CPU threads.

    Returns:
        One row per scored protein. Empty with the correct columns when
        nothing survived or nothing cleared ``identity_threshold``.
    """
    columns = [
        "protein_uid",
        "prompt_id",
        "protein_label",
        "length_percentage",
        "protein_length",
        "prompt_aa_length",
        "best_reference_id",
        "best_reference_identity",
        "full_sequence_identity",
        "non_prompt_sequence_identity",
        "sequence",
    ]
    prompts_by_id = {prompt.prompt_id: prompt for prompt in prompts}
    rows: list[dict[str, Any]] = []

    for protein in accepted_proteins(records):
        prompt = prompts_by_id[protein["prompt_id"]]
        prompt_aa = translate_prompt(prompt.sequence).replace("*", "")
        sequence = protein["sequence"]

        match = best_reference_match(sequence, reference_by_id, threads)
        if match is None or match.identity < identity_threshold:
            continue

        label = prompt.get(protein_label_column)
        reference = reference_by_label.get(label.lower()) if label else None
        if reference is None:
            print(
                f"WARNING: prompt {prompt.prompt_id}: no reference found for "
                f"{protein_label_column}={label!r}; reporting the best match only."
            )
            full_identity = match.identity
            non_prompt = 0.0
        else:
            full_identity = pairwise_identity(sequence, reference, threads)
            non_prompt = non_prompt_identity(sequence, reference, prompt_aa, threads)

        rows.append(
            {
                "protein_uid": protein["protein_uid"],
                "prompt_id": prompt.prompt_id,
                "protein_label": label,
                "length_percentage": prompt.get("Length_Percentage"),
                "protein_length": len(sequence),
                "prompt_aa_length": len(prompt_aa),
                "best_reference_id": match.reference_id,
                "best_reference_identity": match.identity,
                "full_sequence_identity": full_identity,
                "non_prompt_sequence_identity": non_prompt,
                "sequence": sequence,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate identity by protein label and prompt truncation level.

    Args:
        results: Output of :func:`score_identities`.

    Returns:
        Mean, standard deviation, and count of both identity measures per
        ``(protein_label, length_percentage)`` group.
    """
    if results.empty:
        return pd.DataFrame(
            columns=[
                "protein_label",
                "length_percentage",
                "full_identity_mean",
                "full_identity_std",
                "non_prompt_identity_mean",
                "non_prompt_identity_std",
                "count",
            ]
        )
    grouped = results.groupby(["protein_label", "length_percentage"], dropna=False).agg(
        full_identity_mean=("full_sequence_identity", "mean"),
        full_identity_std=("full_sequence_identity", "std"),
        non_prompt_identity_mean=("non_prompt_sequence_identity", "mean"),
        non_prompt_identity_std=("non_prompt_sequence_identity", "std"),
        count=("full_sequence_identity", "size"),
    )
    return grouped.reset_index().round(2)


def run_pipeline(config_path: Path, output_root: str | None = None) -> None:
    """Generate gene completions and score them against the references.

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
    settings = generation_settings(data, prepend_prompt=True)
    logger.info(
        "Gene completion: %d prompt(s) x %d sample(s) with %s",
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

    reference_by_id, reference_by_label = read_reference_fasta(Path(data["reference_seqs"]))
    results = score_identities(
        records,
        prompts,
        reference_by_id,
        reference_by_label,
        identity_threshold=float(data.get("seq_identity_match_threshold", 30.0)),
        protein_label_column=data.get("protein_label_column", "Protein_Label"),
        threads=int(data.get("mafft_threads", 1)),
    )
    results.to_csv(output_dir / "identity_results.csv", index=False)
    summarize(results).to_csv(output_dir / "summary_statistics.csv", index=False)

    logger.info("Scored %d completion(s); outputs in %s", len(results), output_dir)
    if results.empty:
        print(
            "WARNING: no completion cleared the identity threshold; "
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
