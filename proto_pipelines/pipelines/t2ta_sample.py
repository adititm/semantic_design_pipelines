"""Type II toxin-antitoxin generation, screening and cofolding.

The whole workflow is one ``RejectionSamplingOptimizer``. Its filters run in
declaration order and short-circuit, so each expensive stage only sees
generations that cleared the cheap ones -- AlphaFold 3 never folds a
generation that failed to produce two QC-clean ORFs:

    generate -> Prodigal + QC -> profile-HMM -> AF3 monomer screen -> cofold

Pairing rule: two proteins pair when they came from the same generation, and
are at most ``pair_max_identity`` identical to each other. Co-occurrence is
not evidence of a toxin-antitoxin interaction -- it only nominates a pair. The
cofold stage is where interface confidence is actually estimated, and the
MMseqs2 novelty filter afterwards is what removes near-copies of the prompt.

The generator is configurable (``generator: evo2`` by default; set ``evo1``
with a matching ``model_checkpoint`` to use Evo 1.5 instead).

Usage:
    python -m proto_pipelines.pipelines.t2ta_sample \
        --config proto_pipelines/configs/t2ta_sample.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from proto_language.core import Constraint, Segment

from proto_pipelines.af3 import (
    AlphaFold3MonomerScreenConfig,
    AlphaFold3RunConfig,
    af3_monomer_screen_constraint,
)
from proto_pipelines.cofold import (
    TACofoldConfig,
    apply_novelty_filter,
    ta_cofold_constraint,
)
from proto_pipelines.config import (
    af3_run_config,
    generation_settings,
    load_yaml,
    resolve_output_dir,
)
from proto_pipelines.hmm import ProfileHMMFilterConfig, profile_hmm_filter_constraint
from proto_pipelines.prompts import Prompt, read_prompts
from proto_pipelines.qc import ProteinQCConfig, prodigal_protein_qc_constraint
from proto_pipelines.reporting import (
    accepted_proteins,
    cofold_pairs,
    write_fasta,
    write_filter_summary,
    write_fold_scores,
    write_hmm_hits,
    write_proposal_table,
    write_raw_metadata,
    write_stage_tables,
)
from proto_pipelines.runner import run_prompts

logger = logging.getLogger(__name__)

ALLOWED_KEYS = {
    "input_prompts",
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
    "run_af3",
    "af3_plddt_threshold",
    "af3_ptm_threshold",
    "af3_max_proteins_per_proposal",
    "af3_use_msa",
    "af3_pair_heterocomplex_msas",
    "af3_msa_search_mode",
    "af3_msa_device",
    "af3_num_recycles",
    "af3_num_diffusion_samples",
    "af3_seeds",
    "af3_save_structures",
    "hmm_path",
    "hmm_evalue_threshold",
    "hmm_required_profiles",
    "hmm_min_matching_proteins",
    "hmm_annotate_only",
    "max_pair_identity",
    "mafft_threads",
    "run_cofold",
    "novelty_mmseqs_db",
    "novelty_max_identity",
    "cofold_pdockq2_threshold",
    "cofold_min_iptm",
    "cofold_min_avg_plddt",
    "cofold_max_pairs_per_proposal",
    "cofold_min_passing_pairs",
    "cofold_target_fasta",
    "cofold_af3_use_msa",
    "cofold_af3_num_recycles",
    "cofold_af3_num_diffusion_samples",
    "cofold_af3_save_structures",
}

QC_LABEL = "protein_qc"
HMM_LABEL = "profile_hmm"
AF3_LABEL = "af3_monomer_screen"
COFOLD_LABEL = "ta_cofold"

# Two proteins per generation are needed for an intra-generation pair.
MIN_PROTEINS_FOR_PAIRING = 2


def build_constraint_chain(data: dict[str, Any], output_dir: Path) -> Any:
    """Return a callback that builds this pipeline's ordered filter chain.

    Args:
        data: Parsed config mapping.
        output_dir: Pipeline output directory, used for AlphaFold 3 results.

    Returns:
        A callable taking the run's ``Segment`` and ``Prompt`` and returning
        its filters, cheapest first. This pipeline's filters do not depend on
        the prompt.
    """
    qc_config = ProteinQCConfig(
        min_length=int(data.get("filter_min_length", 50)),
        max_length=int(data.get("filter_max_length", 300)),
        filter_partial=bool(data.get("filter_partial_bool", True)),
        segmasker_threshold=float(data.get("segmasker_threshold", 0.1)),
        min_surviving_proteins=MIN_PROTEINS_FOR_PAIRING,
    )
    run_af3 = bool(data.get("run_af3", True))
    screen_config = AlphaFold3MonomerScreenConfig(
        qc_constraint_label=QC_LABEL,
        # Calibrated against random-DNA ORFs: keeps 92% of Evo ORFs, drops 46%
        # of junk. pTM is the load-bearing one (AUROC 0.854 vs 0.731).
        # See proto_pipelines/calibration/results/junk_vs_evo/.
        plddt_threshold=float(data.get("af3_plddt_threshold", 30.0)),
        ptm_threshold=float(data.get("af3_ptm_threshold", 0.20)),
        min_surviving_proteins=MIN_PROTEINS_FOR_PAIRING,
        max_proteins_per_proposal=int(data.get("af3_max_proteins_per_proposal", 8)),
        alphafold3=af3_run_config(
            data, output_dir=output_dir, subdirectory="af3_monomers", use_msa=False
        ),
    )

    hmm_path = data.get("hmm_path")
    hmm_config = (
        ProfileHMMFilterConfig(
            hmm_path=str(hmm_path),
            qc_constraint_label=QC_LABEL,
            evalue_threshold=float(data.get("hmm_evalue_threshold", 1.0)),
            required_profiles=list(data.get("hmm_required_profiles", [])),
            min_matching_proteins=int(data.get("hmm_min_matching_proteins", 1)),
            annotate_only=bool(data.get("hmm_annotate_only", False)),
        )
        if hmm_path
        else None
    )

    run_cofold = bool(data.get("run_cofold", True))
    targets: dict[str, str] = {}
    if data.get("cofold_target_fasta"):
        from proto_pipelines.calibration.build_tadb_set import read_fasta

        targets = {
            h.split()[0]: v
            for h, v in read_fasta(Path(data["cofold_target_fasta"])).items()
        }
        logger.info("Cofolding against %d fixed target(s)", len(targets))
    cofold_config = TACofoldConfig(
        source_constraint_label=AF3_LABEL if run_af3 else QC_LABEL,
        source_metadata_key="af3_proteins" if run_af3 else "qc_proteins",
        max_pair_identity=float(data.get("max_pair_identity", 70.0)),
        mafft_threads=int(data.get("mafft_threads", 1)),
        pdockq2_threshold=float(data.get("cofold_pdockq2_threshold", 0.23)),
        min_iptm=float(data.get("cofold_min_iptm", 0.55)),
        min_avg_plddt=float(data.get("cofold_min_avg_plddt", 80.0)),
        min_passing_pairs=int(data.get("cofold_min_passing_pairs", 1)),
        max_pairs_per_proposal=int(data.get("cofold_max_pairs_per_proposal", 6)),
        target_sequences=targets,
        alphafold3=AlphaFold3RunConfig(
            use_msa=bool(data.get("cofold_af3_use_msa", True)),
            pair_heterocomplex_msas=True,
            msa_device=data.get("af3_msa_device", "cuda"),
            num_recycles=int(data.get("cofold_af3_num_recycles", 10)),
            num_diffusion_samples=int(data.get("cofold_af3_num_diffusion_samples", 5)),
            output_dir=(
                str(output_dir / "af3_complexes")
                if data.get("cofold_af3_save_structures", True)
                else None
            ),
            device=data.get("device", "cuda"),
            verbose=bool(data.get("verbose", False)),
        ),
    )

    def build(segment: Segment, prompt: Prompt) -> list[Constraint]:
        constraints = [
            Constraint(
                inputs=[segment],
                function=prodigal_protein_qc_constraint,
                function_config=qc_config,
                label=QC_LABEL,
                threshold=0.5,
            )
        ]
        # Ordered between QC and folding: cheap relative to AlphaFold 3, and
        # it spends the fold budget on generations that look TA-like.
        if hmm_config is not None:
            constraints.append(
                Constraint(
                    inputs=[segment],
                    function=profile_hmm_filter_constraint,
                    function_config=hmm_config,
                    label=HMM_LABEL,
                    threshold=0.5,
                )
            )
        if run_af3:
            constraints.append(
                Constraint(
                    inputs=[segment],
                    function=af3_monomer_screen_constraint,
                    function_config=screen_config,
                    label=AF3_LABEL,
                    threshold=0.5,
                )
            )
        # Last in the chain: the optimizer short-circuits, so only generations
        # that already yielded two confident proteins reach the cofold.
        if run_cofold:
            constraints.append(
                Constraint(
                    inputs=[segment],
                    function=ta_cofold_constraint,
                    function_config=cofold_config,
                    label=COFOLD_LABEL,
                    threshold=0.5,
                )
            )
        return constraints

    return build


def run_pipeline(config_path: Path, output_root: str | None = None) -> None:
    """Sample Type II TA candidates and emit pairs for the cofold stage.

    Args:
        config_path: Path to the pipeline YAML config.
        output_root: Overrides where the run writes; see
            :func:`proto_pipelines.config.resolve_output_dir` for precedence.

    Returns:
        None. Writes ``generated_sequences.csv``, ``filter_summary.csv``,
        ``proteins.csv``, ``proteins.fasta``, ``protein_pairs.csv`` and
        ``cofold_targets.fasta`` into the configured ``output_dir``.
    """
    data = load_yaml(config_path, allowed_keys=ALLOWED_KEYS)
    output_dir = resolve_output_dir(data, cli_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(data["input_prompts"])
    settings = generation_settings(data, prepend_prompt=False)
    logger.info(
        "Type II TA: %d prompt(s) x %d sample(s) with %s",
        len(prompts),
        settings.n_sample_per_prompt,
        settings.model_checkpoint,
    )

    records = run_prompts(
        prompts,
        settings,
        build_constraint_chain(data, output_dir),
        checkpoint_path=output_dir / "checkpoint.jsonl",
    )

    # Must mirror build_constraint_chain's gating exactly: a filter missing
    # from this list is a filter whose rejections get reported as though they
    # came from something that is not a filter at all.
    labels = [QC_LABEL]
    if data.get("hmm_path"):
        labels.append(HMM_LABEL)
    if data.get("run_af3", True):
        labels.append(AF3_LABEL)
    if data.get("run_cofold", True):
        labels.append(COFOLD_LABEL)
    write_proposal_table(
        records,
        output_dir / "generated_sequences.csv",
        extra_columns={
            "model": settings.model_checkpoint,
            "temperature": settings.temperature,
            "top_k": settings.top_k,
        },
    )
    summary = write_filter_summary(records, labels, output_dir / "filter_summary.csv")
    write_hmm_hits(records, output_dir / "hmm_hits.csv")
    write_raw_metadata(records, output_dir / "raw_metadata.json")
    stages = write_stage_tables(records, output_dir)
    logger.info("Stage tables: %s", ", ".join(f"{k}={v}" for k, v in stages.items()))
    logger.info("Filter summary:\n%s", summary.to_string(index=False))

    # Every fold, passing or not: this is what the AlphaFold 3 cutoffs should
    # be chosen from, since the paper's ESMFold values do not transfer.
    folds = write_fold_scores(records, output_dir / "af3_fold_scores.csv")
    if not folds.empty:
        logger.info(
            "AlphaFold 3 folded %d protein(s); pLDDT %.1f-%.1f, pTM %.2f-%.2f, %d passed",
            len(folds),
            folds["avg_plddt"].min(),
            folds["avg_plddt"].max(),
            folds["ptm"].min(),
            folds["ptm"].max(),
            int(folds["passed_af3_screen"].sum()),
        )

    proteins = accepted_proteins(records)
    pd.DataFrame(proteins).to_csv(output_dir / "proteins.csv", index=False)
    write_fasta(
        ((protein["protein_uid"], protein["sequence"]) for protein in proteins),
        output_dir / "proteins.fasta",
    )

    # Pairs and their interface scores come out of the proposal metadata now;
    # the cofold ran inside the Program as the last filter.
    pairs = cofold_pairs(records)
    pairs.to_csv(output_dir / "cofold_pairs.csv", index=False)
    if not pairs.empty:
        passing = pairs[pairs["passed_gates"]]
        passing.to_csv(output_dir / "cofold_high_confidence.csv", index=False)
        write_fasta(
            (
                e
                for row in passing.itertuples()
                for e in (
                    (f"{row.uid_1} pdockq2={row.pdockq2:.3f}", row.seq_1),
                    (f"{row.uid_2} pdockq2={row.pdockq2:.3f}", row.seq_2),
                )
            ),
            output_dir / "cofold_high_confidence.fasta",
        )
        logger.info(
            "Cofolded %d pair(s); %d passed all gates (pDockQ2/ipTM/pLDDT)",
            len(pairs),
            int(pairs["passed_gates"].sum()),
        )
    else:
        print(
            "WARNING: no pairs were cofolded. Check filter_summary.csv for the "
            "stage that removed the generations."
        )

    # Novelty is a provenance policy, not evidence of function, so it runs on
    # gated survivors rather than as a filter in the Program.
    novelty_db = data.get("novelty_mmseqs_db")
    if not novelty_db:
        print(
            "NOTE: novelty_mmseqs_db is not set, so no MMseqs2 novelty filter "
            "ran and cofold_gated_with_novelty.csv was not written."
        )
    if novelty_db and not pairs.empty and pairs["passed_gates"].any():
        gated = apply_novelty_filter(
            pairs[pairs["passed_gates"]].rename(
                columns={"seq_1": "sequence_1", "seq_2": "sequence_2"}
            ),
            str(novelty_db),
            float(data.get("novelty_max_identity", 75.0)),
        )
        gated.to_csv(output_dir / "cofold_gated_with_novelty.csv", index=False)
        logger.info(
            "Novelty: %d/%d gated pair(s) below %.0f%% identity",
            int(gated["is_novel"].sum()),
            len(gated),
            float(data.get("novelty_max_identity", 75.0)),
        )

    logger.info(
        "Wrote %d protein(s) and %d scored pair(s) to %s",
        len(proteins),
        len(pairs),
        output_dir,
    )
    logger.info(
        "These gates filter for a plausible predicted complex, NOT a functional "
        "TA pair: on functionally labelled de novo pairs the one known "
        "non-functional pair was indistinguishable from two confirmed-functional "
        "ones, and this gate set rejects 3 of 9 confirmed-functional pairs. "
        "Rank within a generation rather than thresholding to select."
    )


def main() -> None:
    """Parse ``--config`` and run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", required=True, help="Path to the pipeline YAML config."
    )
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
