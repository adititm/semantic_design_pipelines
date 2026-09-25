"""Anti-CRISPR evidence scoring for generated proteins.

The published workflow stopped at structure prediction and called
anti-CRISPRs with PaCRISPR. **PaCRISPR is no longer available** -- it was a
web server with no offline release, and it is not part of the published code.
That step is therefore not reproducible as written. This module replaces that step with five independent callers --
profile HMM, AcRanker, AcrNET, Foldseek TM-score and AlphaFold 3 pLDDT --
combined by logistic models fitted on a labelled calibration set.

Which model applies depends on two regimes, because both change what a
score means rather than merely shifting it:

* **HMM hit or not.** Most known Acrs hit no Pfam family, so profile
  absence is not evidence against; a separate model drops the HMM term
  instead of reading it as a negative.
* **PSSM or not.** A missing PSI-BLAST profile inflates AcrNET's output
  toward 1.0 regardless of the protein, and unprofilable ORFs are exactly
  the divergent candidates this pipeline exists to find. Scores are read
  against the regime that produced them.

Thresholds, per-regime tier tables, cross-validated AUROCs and how to
re-derive any of them: ``docs/ACR_PIPELINE.md``.

Two things shape how the output should be read:

* **Acr and Aca cannot be cleanly separated, and should not be.** Every
  caller confuses them, because the confusion is real: AcrIIA1, AcrIIA13,
  AcrIIA15 and AcrIF24 carry HTH domains and act as their own operon
  repressors, and Aca proteins are HTH regulators. The callers recover a
  documented overlap rather than failing. ``acr_locus_score`` therefore
  targets "protein from an Acr locus"; ``hmm_profiles`` says which side it
  looks like.
* **Nothing here proves a sequence is an anti-CRISPR.** These are
  similarity and confidence measures -- level-1 evidence. A high score
  means "worth testing", never "is an Acr".

**Scores are not comparable across runs.** AlphaFold 3 sampling is not
deterministic, and Foldseek inherits and amplifies that: one sequence
folded three times moved ``acr_locus_score`` from 0.048 to 0.477. Rank
within a run; do not read a threshold crossing across runs as meaningful.
Pointing successive runs at the same AlphaFold 3 output directory makes
repeat folds both free and deterministic, since ``job_name_for`` is a
content hash and the score cache sits beside the structures.

"""

from __future__ import annotations

import bisect
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any

from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput
from proto_language.utils.base import BaseConfig, ConfigField

logger = logging.getLogger(__name__)

#: Reduced 7-group alphabet, verbatim from AcRanker's server2.twomerFromSeq.
_GROUPS = {
    "A": "1",
    "V": "1",
    "G": "1",
    "I": "2",
    "L": "2",
    "F": "2",
    "P": "2",
    "Y": "3",
    "M": "3",
    "T": "3",
    "S": "3",
    "H": "4",
    "N": "4",
    "Q": "4",
    "W": "4",
    "R": "5",
    "K": "5",
    "D": "6",
    "E": "6",
    "C": "7",
}
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

#: Calibration CSVs and pipeline records name some columns differently. A
#: model feature that does not resolve to a record key must fail loudly: the
#: earlier ``record.get(name)`` silently substituted 0.0, which zeroed the
#: Foldseek term in every run while still producing a plausible-looking score.
_FEATURE_ALIASES = {"best_tmscore": "foldseek_tmscore"}


class AcrEvidenceConfig(BaseConfig):
    """Configuration for :func:`acr_evidence_constraint`.

    Attributes:
        source_constraint_label: Upstream constraint supplying proteins.
        source_metadata_key: Metadata key holding the protein list.
        hmm_path: Acr/Aca profile database. Empty disables the HMM caller.
        hmm_evalue: Sequence-level E-value cap for the scan.
        acranker_model: AcRanker booster JSON. Empty disables that caller.
        acranker_shuffles: Shuffles per sequence for the composition-matched
            ``z``. Zero disables it.
        foldseek_ref_dir: Directory of reference Acr chain PDBs. Empty
            disables the structural caller.
        foldseek_max_self_identity: Drop Foldseek hits at or above this
            identity, so a candidate cannot match its own source structure.
        combined_model: Logistic-model JSON from calibration. Empty reports
            the per-caller scores without combining them.
        min_score: ``acr_locus_score`` a generation must reach to pass.
            ``0.0`` records evidence without rejecting anything, which is the
            default: the callers are calibrated on natural Acrs, and their
            behaviour on generated sequence is not yet characterised.
        min_qualifying_proteins: Proteins that must clear ``min_score``.
    """

    source_constraint_label: str = ConfigField(
        default="af3_monomer_screen",
        title="Source Constraint",
        description="Constraint whose metadata supplies the proteins to score.",
    )
    source_metadata_key: str = ConfigField(
        default="af3_proteins", title="Source Key", description="Protein list key."
    )
    hmm_path: str = ConfigField(
        default="",
        title="Acr HMM",
        description="Acr/Aca profile database; empty disables the caller.",
    )
    aca_hmm_path: str = ConfigField(
        default="",
        title="Aca HMM",
        description=(
            "HTH families enriched in Aca proteins (>=10x over generic bacterial "
            "transcription factors). Detects 66% of Aca at an 8.3% false-positive "
            "rate on 2625 generic bacterial TFs -- these are Aca-ENRICHED, not "
            "Aca-exclusive. Empty disables the caller."
        ),
    )
    hmm_evalue: float = ConfigField(
        default=1.0,
        gt=0,
        title="HMM E-value",
        description="Sequence-level E-value cap; lenient, since strictness selects against divergence.",
    )
    acranker_model: str = ConfigField(
        default="",
        title="AcRanker Model",
        description="Path to the AcRanker booster JSON; empty disables the caller.",
    )
    acranker_shuffles: int = ConfigField(
        default=20,
        ge=0,
        title="AcRanker Shuffles",
        description="Shuffles per sequence for the composition-matched z; 0 disables it.",
    )
    foldseek_ref_dir: str = ConfigField(
        default="",
        title="Foldseek Reference",
        description="Directory of reference Acr chain PDBs; empty disables the caller.",
    )
    foldseek_binary: str = ConfigField(
        default="",
        title="Foldseek Binary",
        description=(
            "foldseek executable. Empty resolves "
            "$PROTO_HOME/proto_tool_envs/foldseek_env/bin/foldseek, which "
            "proto-tools provisions."
        ),
    )
    foldseek_max_self_identity: float = ConfigField(
        default=0.90,
        ge=0,
        le=1,
        title="Max Self Identity",
        description="Drop hits at or above this identity so a query cannot match itself.",
    )
    combined_model: str = ConfigField(
        default="",
        title="Combined Model",
        description="Calibrated logistic model JSON; empty reports per-caller scores only.",
    )
    acrnet_home: str = ConfigField(
        default="",
        title="AcrNET Home",
        description="AcrNET checkout containing model.ckpt; empty disables the caller.",
    )
    acrnet_predict_property: str = ConfigField(
        default="",
        title="Predict_Property Home",
        description="RaptorX Predict_Property checkout for ss3/ss8/acc (no database needed).",
    )
    acrnet_psiblast: str = ConfigField(
        default="",
        title="psiblast Binary",
        description="psiblast executable; empty zeroes the PSSM block (1.00 -> 0.90 accuracy).",
    )
    acrnet_blast_db: str = ConfigField(
        default="",
        title="BLAST Database",
        description="BLAST database prefix for the PSSM; empty zeroes that block.",
    )
    acrnet_workers: int = ConfigField(
        default=8,
        ge=1,
        title="AcrNET Workers",
        description="Concurrent RaptorX/PSI-BLAST processes; PSI-BLAST scales by process, not thread.",
    )
    acrnet_device: str = ConfigField(
        default="cuda",
        title="ESM Device",
        description="Device for the ESM-1b embedding.",
    )
    acrnet_operating_points: str = ConfigField(
        default="",
        title="AcrNET Operating Points",
        description=(
            "Empirical tier bounds JSON. AcrNET's probability is saturated near "
            "1.0 -- 33% of known negatives exceed 0.9 -- so a bare 0.9x is not a "
            "positive call. Empty leaves acrnet_tier/acrnet_pctile unset."
        ),
    )
    divergent_model: str = ConfigField(
        default="",
        title="Divergent Model",
        description=(
            "Model used when the HMM finds nothing. Absence of a Pfam hit is not "
            "evidence against an Acr -- 49 of 64 known Acrs have none -- so the "
            "HMM feature must be dropped rather than read as a negative."
        ),
    )
    require_af3_pass: bool = ConfigField(
        default=True,
        title="Require AF3 Pass",
        description=(
            "Exclude proteins that failed the AlphaFold 3 pLDDT/pTM gate from "
            "counting as candidates. They stay in the evidence table and still "
            "contribute locus context -- an Aca partner is identified by "
            "sequence HMM and is useful even when its own fold is poor -- but "
            "a candidate you would actually test should have folded."
        ),
    )
    min_score: float = ConfigField(
        default=0.0,
        ge=0,
        le=1,
        title="Minimum Score",
        description="acr_locus_score a protein must reach; 0 records evidence without filtering.",
    )
    min_qualifying_proteins: int = ConfigField(
        default=1,
        ge=1,
        title="Minimum Qualifying Proteins",
        description="Proteins that must reach min_score for the generation to pass.",
    )


def _acranker_features(sequence: str) -> list[float]:
    """AcRanker's 412 features: composition, then 2-mer, then 3-mer blocks.

    Each block is L2-normalised, and both k-mer blocks divide by
    ``len(sequence) - 1`` -- reproducing AcRanker's own normalisation, which
    is technically wrong for 3-mers but is what the shipped model expects.

    Args:
        sequence: Protein sequence.

    Returns:
        The 412-dimensional feature vector.
    """
    import numpy as np

    def l2(vector: Any) -> Any:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    from itertools import product

    length = max(len(sequence), 1)
    composition = np.array([sequence.count(a) / length for a in _AMINO_ACIDS])
    blocks = [l2(composition)]
    for k in (2, 3):
        index = {"".join(p): i for i, p in enumerate(product("1234567", repeat=k))}
        counts = np.zeros(7**k)
        for start in range(len(sequence) - k + 1):
            kmer = sequence[start : start + k]
            if any(residue not in _GROUPS for residue in kmer):
                continue
            counts[index["".join(_GROUPS[r] for r in kmer)]] += 1
        blocks.append(l2(counts / max(len(sequence) - 1, 1)))
    return list(np.concatenate(blocks))


def _foldseek_binary(configured: str) -> str:
    """Resolve the foldseek executable, or raise saying where to look."""
    if configured:
        if not Path(configured).is_file():
            raise FileNotFoundError(f"foldseek_binary {configured!r} does not exist")
        return configured
    home = os.environ.get("PROTO_HOME")
    if not home:
        raise RuntimeError(
            "foldseek_binary is empty and PROTO_HOME is unset, so the "
            "proto-tools-provisioned foldseek cannot be located. Set one."
        )
    path = Path(home) / "proto_tool_envs/foldseek_env/bin/foldseek"
    if not path.is_file():
        raise FileNotFoundError(
            f"no foldseek at {path}; run the pipeline once so proto-tools "
            "provisions its tool env, or set foldseek_binary."
        )
    return str(path)


#: Standard M8 columns plus the TM-score fields. proto-tools' own
#: ``FoldseekHit`` parses only the first twelve -- it carries no TM column at
#: all -- but the model feature is a TM-score, and ranking on E-value instead
#: is worthless: as a feature it scored CV AUROC 0.9115 against 0.9106 for
#: dropping the structural term entirely, versus 0.9237 with a real TM. So
#: foldseek is invoked directly with the columns it does support.
_FOLDSEEK_TM_FORMAT = (
    "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,"
    "evalue,bits,qtmscore,alntmscore"
)


def _foldseek_tm_hits(
    structure: str,
    ref_dir: str,
    binary: str,
    evalue: float,
    max_seqs: int,
    threads: int,
) -> list[dict[str, Any]]:
    """Run ``foldseek easy-search`` asking for TM-score columns.

    ``qtmscore`` is the query-normalised TM-score -- the same quantity the
    shipped models were calibrated on.

    Returns:
        One dict per hit with ``target``, ``pident`` (fraction), ``evalue``,
        ``bits``, ``qtmscore`` and ``alntmscore``.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        out = Path(work) / "result.m8"
        command = [
            _foldseek_binary(binary),
            "easy-search",
            structure,
            ref_dir,
            str(out),
            str(Path(work) / "fs_tmp"),
            "--format-output",
            _FOLDSEEK_TM_FORMAT,
            "-e",
            str(evalue),
            "--max-seqs",
            str(max_seqs),
            # alignment-type 2 (3Di+AA) is proto-tools' default and is what
            # the shipped models were calibrated against. Type 1 (TM-align)
            # returns systematically HIGHER TM -- median +0.15 over 25
            # re-searched calibration chains -- so switching it would put the
            # best_tmscore feature on a different scale than the fits.
            "--alignment-type",
            "2",
            "--threads",
            str(max(1, threads)),
        ]
        # Let a foldseek failure propagate: a crashed search must not look
        # like a protein with no structural neighbours.
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not out.is_file():
            logger.warning("foldseek wrote no result file for %s", structure)
            return []
        hits = []
        for line in out.read_text().splitlines():
            row = line.split("\t")
            if len(row) < 14:
                continue
            hits.append(
                {
                    "target": row[1],
                    "pident": float(row[2]) / 100.0,
                    "evalue": float(row[10]),
                    "bits": float(row[11]),
                    "qtmscore": float(row[12]),
                    "alntmscore": float(row[13]),
                }
            )
        if not hits:
            logger.warning("foldseek returned no hits for %s", structure)
        return hits


def score_proteins(
    proteins: list[dict[str, Any]],
    config: AcrEvidenceConfig,
    acrnet_scores: dict[str, float] | None = None,
    acrnet_has_pssm: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Run every enabled caller over one generation's proteins.

    Each caller is independent and optional; a caller that is not configured
    contributes ``None`` rather than a zero, so "not run" stays
    distinguishable from "ran and found nothing".

    Args:
        proteins: Protein dicts carrying ``protein_id``, ``sequence`` and,
            when the AlphaFold 3 screen ran, ``avg_plddt``.
        config: Caller settings.
        acrnet_has_pssm: ``{protein_id: bool}`` -- whether PSI-BLAST built a
            real PSSM. Required to read ``acrnet_score``: a zeroed PSSM
            inflates it toward 1.0 regardless of class.
        acrnet_scores: ``{protein_id: score}`` from the batched AcrNET run.
            Passed in rather than computed here because AcrNET is extracted
            once for the whole proposal batch -- and it must be attached
            before the combined model runs, or its term is silently absent.

    Returns:
        One record per protein with the per-caller scores and, when a
        combined model is configured, ``acr_locus_score``.
    """
    import numpy as np

    records = [
        {
            "protein_id": p["protein_id"],
            "length": len(p["sequence"]),
            "avg_plddt": p.get("avg_plddt"),
            "ptm": p.get("ptm"),
            # None when the monomer screen did not run (sequence-only
            # prescreen stage), which must not be read as a failure.
            "passed_af3_screen": p.get("passed_af3_screen"),
            "hmm_score": None,
            "hmm_best_profile": None,
            "hmm_best_evalue": None,
            "acranker_raw": None,
            "acranker_z": None,
            "foldseek_tmscore": None,
            "foldseek_target": None,
            "aca_best_profile": None,
            "is_aca_like": None,
            "acrnet_score": (acrnet_scores or {}).get(p["protein_id"]),
            "acrnet_has_pssm": (acrnet_has_pssm or {}).get(p["protein_id"]),
            "acrnet_pctile": None,
            "acrnet_tier": None,
            "acrnet_regime": None,
        }
        for p in proteins
    ]

    if config.hmm_path:
        from proto_pipelines.hmm import _scan

        for record, protein in zip(records, proteins, strict=True):
            hits = _scan(protein["sequence"], config.hmm_path, config.hmm_evalue)
            if hits:
                record["hmm_best_profile"] = hits[0]["profile"]
                record["hmm_best_evalue"] = hits[0]["evalue"]
                record["hmm_score"] = -math.log10(max(hits[0]["evalue"], 1e-300))
            else:
                record["hmm_score"] = 0.0

    if config.aca_hmm_path:
        from proto_pipelines.hmm import _scan

        for record, protein in zip(records, proteins, strict=True):
            hits = _scan(protein["sequence"], config.aca_hmm_path, config.hmm_evalue)
            record["aca_best_profile"] = hits[0]["profile"] if hits else None
            record["is_aca_like"] = bool(hits)

    if config.acranker_model:
        import xgboost

        booster = xgboost.Booster()
        booster.load_model(config.acranker_model)
        rng = random.Random(0)
        for record, protein in zip(records, proteins, strict=True):
            sequence = protein["sequence"]
            rows = [_acranker_features(sequence)]
            for _ in range(config.acranker_shuffles):
                residues = list(sequence)
                rng.shuffle(residues)
                rows.append(_acranker_features("".join(residues)))
            scores = booster.predict(xgboost.DMatrix(np.array(rows)))
            record["acranker_raw"] = float(scores[0])
            if config.acranker_shuffles:
                null = scores[1:]
                spread = float(null.std()) or 1e-9
                record["acranker_z"] = float((scores[0] - null.mean()) / spread)

    if config.foldseek_ref_dir:
        for record, protein in zip(records, proteins, strict=True):
            path = protein.get("structure_path") or protein.get("af3_pdb")
            if not path or not Path(path).exists():
                continue
            hits = _foldseek_tm_hits(
                structure=str(path),
                ref_dir=config.foldseek_ref_dir,
                binary=config.foldseek_binary,
                evalue=10.0,
                max_seqs=500,
                threads=config.acrnet_workers,
            )
            kept = [h for h in hits if h["pident"] < config.foldseek_max_self_identity]
            best = max(kept, key=lambda h: h["qtmscore"], default=None)
            record["foldseek_tmscore"] = best["qtmscore"] if best else 0.0
            record["foldseek_target"] = best["target"] if best else None

    def apply(model: dict[str, Any], record: dict[str, Any]) -> float:
        values = []
        for name in model["features"]:
            key = name if name in record else _FEATURE_ALIASES.get(name, name)
            if key not in record:
                raise KeyError(
                    f"model feature {name!r} has no matching field on the "
                    f"evidence record (have: {sorted(record)}). Add an alias "
                    "rather than letting it default to zero."
                )
            value = record[key]
            # None means that caller was disabled, which is a real state; a
            # missing key is a bug and was raised above.
            values.append(0.0 if value is None else value)
        z = [
            (v - m) / s
            for v, m, s in zip(values, model["mean"], model["scale"], strict=True)
        ]
        logit = model["intercept"] + sum(
            c * v for c, v in zip(model["coef"], z, strict=True)
        )
        return 1.0 / (1.0 + math.exp(-logit))

    # AcrNET's probability is saturated AND regime-dependent. A missing
    # PSI-BLAST PSSM pushes it toward 1.0 whatever the protein is: zeroing
    # the PSSM on the same 63 phage proteins moved them from 19% to 78%
    # above 0.99, and dropped Acr-vs-phage AUROC from 0.918 to 0.791. The
    # score is therefore read against the regime that produced it, never
    # against one global table -- otherwise every ORF that PSI-BLAST cannot
    # profile, which includes the divergent candidates this pipeline exists
    # to find, is silently rewarded for being unprofilable.
    #
    # AcrNET is not useless without a PSSM (AUROC 0.806 in that regime), so
    # the caller is kept and recalibrated rather than dropped.
    if config.acrnet_operating_points:
        ops = json.loads(Path(config.acrnet_operating_points).read_text())
        for record in records:
            score = record.get("acrnet_score")
            if score is None:
                continue
            has_pssm = record.get("acrnet_has_pssm")
            if has_pssm is None:
                raise ValueError(
                    f"{record['protein_id']}: acrnet_score is set but "
                    "acrnet_has_pssm is not, so the score cannot be assigned "
                    "to a calibration regime. Pass acrnet_has_pssm."
                )
            name = "has_pssm" if has_pssm else "no_pssm"
            regime = ops["regimes"][name]
            record["acrnet_regime"] = name
            quantiles = sorted(float(v) for v in regime["negative_quantiles"].values())
            record["acrnet_pctile"] = bisect.bisect_right(quantiles, score) / len(
                quantiles
            )
            for tier_def in regime["tiers"]:
                if tier_def["lower"] <= score < tier_def["upper"]:
                    record["acrnet_tier"] = tier_def["name"]
                    break
            else:
                raise ValueError(
                    f"acrnet_score {score!r} fell outside every tier of regime "
                    f"{name!r} in {config.acrnet_operating_points}; tiers must "
                    "span [0, 1]."
                )

    # Two models, because a profile hit and its absence mean different
    # things. A hit is near-certain evidence, but most known Acrs have no
    # Pfam hit at all -- so for those the full model would penalise a
    # missing feature that carries no information. The divergent model drops
    # the HMM term rather than reading it as a negative. Cross-validated
    # AUROCs for both are in docs/ACR_PIPELINE.md and in each model JSON.
    combined = (
        json.loads(Path(config.combined_model).read_text())
        if config.combined_model
        else None
    )
    divergent = (
        json.loads(Path(config.divergent_model).read_text())
        if config.divergent_model
        else None
    )
    for record in records:
        has_hmm_hit = bool(record.get("hmm_best_profile"))
        record["acr_tier"] = "hmm_hit" if has_hmm_hit else "divergent"
        model = combined if (has_hmm_hit and combined) else (divergent or combined)
        if model is not None:
            record["acr_locus_score"] = apply(model, record)

    # A candidate is a protein you would put on a bench: it must clear the
    # score AND have survived the AlphaFold 3 gate. Gate failures stay in the
    # table, flagged, because they still carry locus context -- an Aca partner
    # is identified by sequence HMM and is informative even when its own fold
    # is poor. ``passed_af3_screen`` is None when no fold ran at all (the
    # sequence-only prescreen stage); that is not a failure.
    for record in records:
        record["is_candidate"] = bool(
            (record.get("acr_locus_score") or 0.0) >= config.min_score
            and not (
                config.require_af3_pass and record.get("passed_af3_screen") is False
            )
        )
    return records


@constraint(
    key="acr-evidence",
    label="Anti-CRISPR Evidence",
    config=AcrEvidenceConfig,
    description="Score proteins for anti-CRISPR evidence from HMM, AcRanker, Foldseek and pLDDT.",
    tools_called=["pyhmmer-hmmscan", "foldseek-search"],
    category="protein_quality",
    supported_sequence_types=["dna"],
)
def acr_evidence_constraint(
    input_sequences: list[tuple[Any, ...]], config: AcrEvidenceConfig
) -> list[ConstraintOutput]:
    """Record anti-CRISPR evidence for each generation's proteins.

    Args:
        input_sequences: One tuple per proposal holding a DNA ``Sequence``
            carrying upstream protein metadata.
        config: Caller and gating settings.

    Returns:
        One ``ConstraintOutput`` per proposal. Passes when at least
        ``min_qualifying_proteins`` reach ``min_score``; with the default
        ``min_score`` of 0 every proposal passes and the evidence is recorded
        without filtering.

    Raises:
        KeyError: If the source constraint produced no metadata.
    """
    # Collect every protein across the whole batch first: AcrNET's feature
    # stack (RaptorX, PSI-BLAST, ESM-1b) is far cheaper per protein when the
    # pools and the GPU pass are shared across proposals than when each
    # proposal pays the startup cost on its own.
    per_proposal: list[list[dict[str, Any]]] = []
    for sequence_tuple in input_sequences:
        sequence = sequence_tuple[0]
        entry = sequence._constraints_metadata.get(config.source_constraint_label)
        if entry is None:
            raise KeyError(
                f"acr_evidence: no metadata from {config.source_constraint_label!r}; "
                "declare that constraint before this one."
            )
        per_proposal.append(entry.get("data", {}).get(config.source_metadata_key) or [])

    acrnet_scores: dict[str, float] = {}
    acrnet_has_pssm: dict[str, bool] = {}
    if config.acrnet_home and config.acrnet_predict_property:
        from proto_pipelines import acrnet as acrnet_module

        flat = [
            {"protein_id": f"p{i}_{p['protein_id']}", "sequence": p["sequence"]}
            for i, proteins in enumerate(per_proposal)
            for p in proteins
        ]
        if flat:
            features = acrnet_module.extract_features(
                flat,
                config.acrnet_predict_property,
                esm_device=config.acrnet_device,
                psiblast_bin=config.acrnet_psiblast,
                blast_db=config.acrnet_blast_db,
                workers=config.acrnet_workers,
            )
            for item in flat:
                if item["protein_id"] in features:
                    features[item["protein_id"]]["sequence"] = item["sequence"]
            acrnet_scores = acrnet_module.score(
                features, str(Path(config.acrnet_home) / "model.ckpt")
            )
            # A missing PSI-BLAST PSSM inflates AcrNET toward 1.0, so the
            # score is only interpretable against the regime that produced
            # it. Divergent ORFs are exactly the ones PSI-BLAST cannot
            # profile, so this is the common case here, not an edge case.
            acrnet_has_pssm = {
                protein_id: bool(feature["pssm"].any())
                for protein_id, feature in features.items()
                if "pssm" in feature
            }

    outputs: list[ConstraintOutput] = []
    for index, proteins in enumerate(per_proposal):
        # Re-key to plain protein_id: the batch used a proposal-prefixed id
        # so names could not collide across proposals.
        local = {
            p["protein_id"]: acrnet_scores[f"p{index}_{p['protein_id']}"]
            for p in proteins
            if f"p{index}_{p['protein_id']}" in acrnet_scores
        }
        local_pssm = {
            p["protein_id"]: acrnet_has_pssm[f"p{index}_{p['protein_id']}"]
            for p in proteins
            if f"p{index}_{p['protein_id']}" in acrnet_has_pssm
        }
        records = (
            score_proteins(proteins, config, local, local_pssm) if proteins else []
        )
        qualifying = sum(1 for r in records if r.get("is_candidate"))
        # Guilt by association. An Acr operon carries an Acr and its Aca
        # repressor, and Aca are far better covered by Pfam than Acrs, so the
        # partner is often the detectable half.
        #
        # An HTH hit ALONE is weak -- these families also fire on generic
        # bacterial transcription factors. What carries information is the
        # co-occurrence: an HTH-family protein in the same generation as an
        # Acr-family protein. Hence the flag is a conjunction, and is named
        # aca_like rather than aca.
        aca_like = [r["protein_id"] for r in records if r.get("is_aca_like")]
        acr_like = [
            r["protein_id"]
            for r in records
            if r.get("hmm_best_profile") and not r.get("is_aca_like")
        ]
        passed = config.min_score <= 0.0 or qualifying >= config.min_qualifying_proteins
        outputs.append(
            ConstraintOutput(
                score=0.0 if passed else 1.0,
                metadata={
                    "acr_evidence": records or None,
                    "acr_protein_count": len(records),
                    "acr_qualifying_count": qualifying,
                    "acr_best_score": max(
                        (r.get("acr_locus_score") or 0.0 for r in records), default=0.0
                    ),
                    "aca_like_proteins": aca_like or None,
                    "acr_family_proteins": acr_like or None,
                    "locus_has_acr_and_aca_like": bool(aca_like and acr_like),
                },
            )
        )
    return outputs
