"""Anti-CRISPR candidate generation and structure screening.

Runs

    generate -> Prodigal + QC -> AlphaFold 3 monomer screen

Like the published workflow, this pipeline stops at structure prediction. The paper's
anti-CRISPR calls came from a separate PaCRISPR analysis that is not part of
this repository, and proto-tools has no Acr-prediction wrapper. Nothing here
labels a sequence as an anti-CRISPR: the output is a set of short, QC-clean,
confidently-folded proteins from Acr-context prompts, which is the input to
such an analysis rather than its result.

Usage:
    python -m proto_pipelines.pipelines.acr_sample --config configs/acr_sample.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from proto_language.core import Constraint, Segment

from proto_pipelines.acr import AcrEvidenceConfig, acr_evidence_constraint
from proto_pipelines.af3 import (
    AlphaFold3MonomerScreenConfig,
    af3_monomer_screen_constraint,
)
from proto_pipelines.config import (
    af3_run_config,
    generation_settings,
    load_yaml,
    resolve_output_dir,
)
from proto_pipelines.prompts import Prompt, read_prompts
from proto_pipelines.qc import ProteinQCConfig, prodigal_protein_qc_constraint
from proto_pipelines.reporting import (
    accepted_proteins,
    write_acr_evidence,
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
    "run_acr_evidence",
    "acr_hmm_path",
    "aca_hmm_path",
    "acr_hmm_evalue",
    "acr_acranker_model",
    "acr_acranker_shuffles",
    "acr_foldseek_ref_dir",
    "acr_foldseek_binary",
    "acr_combined_model",
    "acr_divergent_model",
    "acrnet_home",
    "acrnet_predict_property",
    "acrnet_psiblast",
    "acrnet_blast_db",
    "acrnet_workers",
    "acrnet_device",
    "acrnet_operating_points",
    "acr_source_key",
    "acr_require_af3_pass",
    "acr_prescreen_model",
    "acr_fold_fraction",
    "acr_prescreen_min_score",
    "run_acr_prescreen",
    "acr_min_score",
    "acr_min_qualifying_proteins",
}

QC_LABEL = "protein_qc"
AF3_LABEL = "af3_monomer_screen"
ACR_LABEL = "acr_evidence"
PRESCREEN_LABEL = "acr_prescreen"


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
        max_length=int(data.get("filter_max_length", 200)),
        filter_partial=bool(data.get("filter_partial_bool", True)),
        segmasker_threshold=float(data.get("segmasker_threshold", 0.1)),
        min_surviving_proteins=1,
    )
    run_af3 = bool(data.get("run_af3", True))
    screen_config = AlphaFold3MonomerScreenConfig(
        qc_constraint_label=QC_LABEL,
        # Calibrated against random-DNA ORFs: keeps 92% of Evo ORFs, drops 46%
        # of junk. pTM is the load-bearing one (AUROC 0.854 vs 0.731).
        # See proto_pipelines/calibration/results/junk_vs_evo/.
        plddt_threshold=float(data.get("af3_plddt_threshold", 30.0)),
        ptm_threshold=float(data.get("af3_ptm_threshold", 0.20)),
        min_surviving_proteins=1,
        max_proteins_per_proposal=int(data.get("af3_max_proteins_per_proposal", 8)),
        prescreen_constraint_label=PRESCREEN_LABEL if data.get("run_acr_prescreen", False) else "",
        fold_fraction=float(data.get("acr_fold_fraction", 1.0)),
        prescreen_min_score=float(data.get("acr_prescreen_min_score", 0.0)),
        alphafold3=af3_run_config(
            data, output_dir=output_dir, subdirectory="af3_monomers", use_msa=False
        ),
    )

    # The Acr callers read the AlphaFold 3 survivors, so this only makes
    # sense once the structure screen has run; without it there are no
    # structures for Foldseek and no pLDDT for the combined model.
    run_acr_evidence = bool(data.get("run_acr_evidence", True)) and run_af3
    # Sequence-only prescreen: HMM + AcRanker + AcrNET all take sequence, and
    # cost seconds per protein against AlphaFold 3's minutes. Ranking first
    # lets the fold step skip the least Acr-like tail. Foldseek and pLDDT are
    # deliberately absent -- both need a structure, which is what this stage
    # exists to avoid paying for. Sequence-only costs a little discrimination
    # (see docs/ACR_PIPELINE.md) but is reliable enough to order on; it gates
    # folding, never the final call, and an HMM hit always folds regardless.
    prescreen_config = AcrEvidenceConfig(
        source_constraint_label=QC_LABEL,
        source_metadata_key="qc_proteins",
        hmm_path=str(data.get("acr_hmm_path", "")),
        aca_hmm_path=str(data.get("aca_hmm_path", "")),
        hmm_evalue=float(data.get("acr_hmm_evalue", 1.0)),
        acranker_model=str(data.get("acr_acranker_model", "")),
        acranker_shuffles=int(data.get("acr_acranker_shuffles", 20)),
        foldseek_ref_dir="",          # no structures yet
        combined_model="",            # tier-1 model needs pLDDT and Foldseek
        divergent_model=str(data.get("acr_prescreen_model", "")),
        acrnet_home=str(data.get("acrnet_home", "")),
        acrnet_predict_property=str(data.get("acrnet_predict_property", "")),
        acrnet_psiblast=str(data.get("acrnet_psiblast", "")),
        acrnet_blast_db=str(data.get("acrnet_blast_db", "")),
        acrnet_workers=int(data.get("acrnet_workers", 8)),
        acrnet_device=str(data.get("acrnet_device", "cuda")),
        acrnet_operating_points=str(data.get("acrnet_operating_points", "")),
        min_score=0.0,
    )
    run_prescreen = bool(data.get("run_acr_prescreen", False)) and run_af3

    acr_config = AcrEvidenceConfig(
        source_constraint_label=AF3_LABEL,
        # Every folded ORF, not just the screen survivors: an Acr locus is
        # scored as a whole and a low-pLDDT partner is still evidence.
        source_metadata_key=str(data.get("acr_source_key", "af3_folded_proteins")),
        hmm_path=str(data.get("acr_hmm_path", "")),
        aca_hmm_path=str(data.get("aca_hmm_path", "")),
        hmm_evalue=float(data.get("acr_hmm_evalue", 1.0)),
        acranker_model=str(data.get("acr_acranker_model", "")),
        acranker_shuffles=int(data.get("acr_acranker_shuffles", 20)),
        foldseek_ref_dir=str(data.get("acr_foldseek_ref_dir", "")),
        foldseek_binary=str(data.get("acr_foldseek_binary", "")),
        combined_model=str(data.get("acr_combined_model", "")),
        divergent_model=str(data.get("acr_divergent_model", "")),
        acrnet_home=str(data.get("acrnet_home", "")),
        acrnet_predict_property=str(data.get("acrnet_predict_property", "")),
        acrnet_psiblast=str(data.get("acrnet_psiblast", "")),
        acrnet_blast_db=str(data.get("acrnet_blast_db", "")),
        acrnet_workers=int(data.get("acrnet_workers", 8)),
        acrnet_device=str(data.get("acrnet_device", "cuda")),
        acrnet_operating_points=str(data.get("acrnet_operating_points", "")),
        require_af3_pass=bool(data.get("acr_require_af3_pass", True)),
        min_score=float(data.get("acr_min_score", 0.0)),
        min_qualifying_proteins=int(data.get("acr_min_qualifying_proteins", 1)),
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
        if run_prescreen:
            constraints.append(
                Constraint(
                    inputs=[segment],
                    function=acr_evidence_constraint,
                    function_config=prescreen_config,
                    label=PRESCREEN_LABEL,
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
        if run_acr_evidence:
            constraints.append(
                Constraint(
                    inputs=[segment],
                    function=acr_evidence_constraint,
                    function_config=acr_config,
                    label=ACR_LABEL,
                    threshold=0.5,
                )
            )
        return constraints

    return build


def run_pipeline(config_path: Path, output_root: str | None = None) -> None:
    """Sample and screen anti-CRISPR candidates.

    Args:
        config_path: Path to the pipeline YAML config.
        output_root: Overrides where the run writes; see
            :func:`proto_pipelines.config.resolve_output_dir` for precedence.

    Returns:
        None. Writes ``generated_sequences.csv``, ``filter_summary.csv``,
        ``proteins.csv`` and ``proteins.fasta`` into the configured
        ``output_dir``.
    """
    data = load_yaml(config_path, allowed_keys=ALLOWED_KEYS)
    output_dir = resolve_output_dir(data, cli_root=output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(data["input_prompts"])
    settings = generation_settings(data, prepend_prompt=False)
    logger.info(
        "Acr: %d prompt(s) x %d sample(s) with %s",
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

    # Must match build_constraint_chain's order exactly: the summary's
    # evaluated counts only fall monotonically if the stages are listed in the
    # order the optimizer applied them.
    labels = [QC_LABEL]
    if data.get("run_acr_prescreen", False) and data.get("run_af3", True):
        labels.append(PRESCREEN_LABEL)
    if data.get("run_af3", True):
        labels.append(AF3_LABEL)
    if data.get("run_acr_evidence", True) and data.get("run_af3", True):
        labels.append(ACR_LABEL)
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
    acr_rows = write_acr_evidence(records, output_dir / "acr_evidence.csv")
    logger.info("Acr evidence rows: %d", len(acr_rows))
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

    logger.info("Wrote %d candidate protein(s) to %s", len(proteins), output_dir)
    logger.info(
        "These are structure-screened candidates from Acr-context prompts. "
        "An anti-CRISPR call requires an independent predictor or assay."
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
