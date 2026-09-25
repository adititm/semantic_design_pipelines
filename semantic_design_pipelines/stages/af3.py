"""AlphaFold 3 structure screens for monomers and complexes.

``af3_monomer_screen_constraint``
    Folds every QC-surviving ORF and gates on pLDDT/pTM. Runs
    single-sequence: this is a triage step over many ORFs, and an MSA here
    would rank sequences partly by how many homologs they have, which is
    the opposite of useful for de novo designs.

``score_complex``
    Cofolds two chains with a taxonomy-paired MMseqs2 MSA and returns
    AlphaFold 3's ipTM/pTM/pLDDT/PAE alongside pDockQ2 and pDockQ v1. Used by
    the cofold constraint in :mod:`semantic_design_pipelines.stages.cofold`.

Interpreting the numbers
------------------------
* pLDDT is on a **0-100** scale here. Cutoffs quoted on a 0-1 scale elsewhere
  are not comparable, and the two AlphaFold 3 modes are not comparable with
  each other either: single-sequence pLDDT runs far below MSA-mode pLDDT.
* **pDockQ2** (Zhu 2023) is the interface gate. It uses the PAE matrix and is
  applicable to AlphaFold 3 output; its 0.23 default is the DockQ
  "Acceptable" quality class.
* **pDockQ v1** (Bryant 2022) is computed from the same structure and
  reported for comparability with published tables, but never gates: its
  sigmoid was fit on AlphaFold2-Multimer confidences.

A structural confidence score describes the prediction, not toxin activity,
neutralisation, or anti-CRISPR function. See "Calibration results" in
``README.md`` for what these scores were and were not able to separate.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from proto_language.constraint.constraint_registry import constraint
from proto_language.core import ConstraintOutput
from proto_language.utils.base import BaseConfig, ConfigField
from proto_tools import (
    AlphaFold3Config,
    AlphaFold3Input,
    Complex,
    Mmseqs2HomologySearchConfig,
    PDockQ2Config,
    PDockQ2Input,
    run_alphafold3,
    run_pdockq2,
)
from proto_tools.entities.structures.selection import (
    ChainSelection,
    SingleChainSelection,
)

logger = logging.getLogger(__name__)

# Bryant, Pozzati & Elofsson 2022 (Nat Commun 13:1265), fit on AF2-Multimer.
PDOCKQ_V1_L = 0.724
PDOCKQ_V1_X0 = 152.611
PDOCKQ_V1_K = 0.052
PDOCKQ_V1_B = 0.018
PDOCKQ_V1_CONTACT_CUTOFF = 8.0


class AlphaFold3RunConfig(BaseConfig):
    """Shared AlphaFold 3 execution settings.

    Attributes:
        use_msa: Run MMseqs2 homology search for each protein chain. Off for
            monomer triage, on for the complex screen.
        pair_heterocomplex_msas: Taxonomy-pair the per-chain MSAs of a
            heterocomplex. Only meaningful when ``use_msa`` is set.
        msa_search_mode: MMseqs2 search mode passed to the homology search.
        num_recycles: AlphaFold 3 recycling iterations.
        num_diffusion_samples: Diffusion samples per seed.
        seeds: Seeds to average over.
        output_dir: Directory to persist AlphaFold 3 result folders into. When
            ``None`` the predictions run in a temporary directory that is
            deleted afterwards; structures are still returned in memory.
        device: Device string handed to AlphaFold 3.
        verbose: Forward AlphaFold 3's own progress output.
    """

    use_msa: bool = ConfigField(
        default=False,
        title="Use MSA",
        description="Run MMseqs2 homology search for each protein chain.",
    )
    pair_heterocomplex_msas: bool = ConfigField(
        default=True,
        title="Pair Heterocomplex MSAs",
        description="Taxonomy-pair per-chain MSAs across chains of a heterocomplex.",
    )
    msa_search_mode: str = ConfigField(
        default="local",
        title="MSA Search Mode",
        description="MMseqs2 homology search mode.",
    )
    msa_device: str = ConfigField(
        default="cpu",
        title="MSA Search Device",
        description=(
            "Device for the MMseqs2 homology search. 'cuda' runs MMseqs2-GPU and needs a "
            "'.idx_pad' index on the dataset; 'cpu' is the portable default."
        ),
    )
    num_recycles: int = ConfigField(
        default=10,
        ge=1,
        title="Recycles",
        description="AlphaFold 3 recycling iterations.",
    )
    num_diffusion_samples: int = ConfigField(
        default=5,
        ge=1,
        title="Diffusion Samples",
        description="Diffusion samples per seed.",
    )
    seeds: list[int] = ConfigField(
        default=[0], title="Seeds", description="AlphaFold 3 seeds to average over."
    )
    output_dir: str | None = ConfigField(
        default=None,
        title="Output Directory",
        description="Where to persist AlphaFold 3 result folders; None uses a temp dir.",
    )
    device: str = ConfigField(
        default="cuda", title="Device", description="Device for AlphaFold 3."
    )
    verbose: bool = ConfigField(
        default=False,
        title="Verbose",
        description="Forward AlphaFold 3 progress output.",
    )


def job_name_for(prefix: str, chains: list[str]) -> str:
    """Build a collision-free, reproducible AlphaFold 3 job name.

    The constraint is called once per proposal batch per prompt, so an index
    local to the batch is not unique across a run: with ``output_dir`` set,
    two prompts' results would land in the same folder and overwrite each
    other. Hashing the chain sequences makes the name unique to the input and
    stable across reruns, which also means a persisted result directory can be
    matched back to its sequences.

    Args:
        prefix: Short stage name, e.g. ``"monomer"`` or ``"pair"``.
        chains: The chain sequences being folded, in chain order.

    Returns:
        ``<prefix>_<10-hex-char digest>``.
    """
    digest = hashlib.sha1("\x00".join(chains).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _cache_path(run_config: AlphaFold3RunConfig, job_name: str) -> Path | None:
    """Sidecar path holding a finished score for ``job_name``, if caching is on.

    Caching keys off the persisted output directory, so it is only available
    when structures are being saved. ``job_name`` is already a content hash of
    the chains, which is what makes the sidecar safe to trust.

    Args:
        run_config: AlphaFold 3 execution settings.
        job_name: Content-hashed job name.

    Returns:
        The sidecar path, or ``None`` when results are not persisted.
    """
    if run_config.output_dir is None:
        return None
    return Path(run_config.output_dir) / ".scores" / f"{job_name}.json"


def _load_cached(path: Path | None) -> dict[str, Any] | None:
    """Read a cached score, treating any unreadable sidecar as a miss.

    A truncated sidecar (killed mid-write) must not poison the run, so a
    corrupt file is reported and recomputed rather than raised on.
    """
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        print(f"WARNING: ignoring unreadable score cache {path}: {error}")
        return None


def _store_cached(path: Path | None, payload: dict[str, Any]) -> None:
    """Write a score sidecar atomically so a kill cannot leave a partial file."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(path)


def fold(chains: list[str], job_name: str, run_config: AlphaFold3RunConfig) -> Any:
    """Predict one complex with AlphaFold 3 and return its ``Structure``.

    Each call folds a single complex so that persisted output directories do
    not collide; AlphaFold 3 dispatches one complex at a time regardless.

    Args:
        chains: Protein chain sequences, in the order they should be assigned
            chain IDs A, B, ...
        job_name: Filesystem-safe name for this prediction.
        run_config: AlphaFold 3 execution settings.

    Returns:
        The predicted ``Structure``, carrying per-residue pLDDT in its B-factor
        column and ``avg_plddt``/``ptm``/``iptm``/``avg_pae``/``pae`` on
        ``.metrics``.
    """
    output_prefix = None
    if run_config.output_dir is not None:
        directory = Path(run_config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        output_prefix = str(directory / job_name)

    af3_config = AlphaFold3Config(
        name=job_name,
        output_dir=output_prefix,
        use_msa=run_config.use_msa,
        pair_heterocomplex_msas=run_config.pair_heterocomplex_msas,
        msa_search_config=(
            # device="cuda" engages MMseqs2-GPU. Leaving it at the tool's "cpu"
            # default runs a 365 GB uniref30 search on CPU while the GPU sits
            # idle until the fold step, which dominates wall time for a sweep.
            Mmseqs2HomologySearchConfig(
                search_mode=run_config.msa_search_mode, device=run_config.msa_device
            )
            if run_config.use_msa
            else None
        ),
        include_pae_matrix=True,
        num_recycles=run_config.num_recycles,
        num_diffusion_samples=run_config.num_diffusion_samples,
        seeds=run_config.seeds,
        device=run_config.device,
        verbose=run_config.verbose,
    )
    result = run_alphafold3(
        AlphaFold3Input(complexes=[Complex(chains=list(chains))]),
        af3_config,
    )
    if not result.structures:
        raise RuntimeError(f"AlphaFold 3 returned no structure for job {job_name!r}")
    return result.structures[0]


# ---------------------------------------------------------------------------
# pDockQ (Bryant 2022) -- reported for comparability with the published tables
# ---------------------------------------------------------------------------


def _pdockq_v1_coordinates(
    pdb_text: str,
) -> tuple[dict[str, list[list[float]]], np.ndarray]:
    """Extract per-chain CB coordinates (CA for glycine) and per-residue pLDDT.

    Reproduces the parser used by the cofold stage so the reported
    pDockQ matches the published pipeline's definition exactly.
    """
    chain_coords: dict[str, list[list[float]]] = {}
    plddt_by_residue: dict[str, list[float]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        residue_name = line[17:20].strip()
        if atom_name != "CB" and not (atom_name == "CA" and residue_name == "GLY"):
            continue
        chain = line[21]
        residue_number = int(line[22:26])
        chain_coords.setdefault(chain, []).append(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        )
        plddt_by_residue.setdefault(f"{chain}{residue_number}", []).append(
            float(line[60:66])
        )
    plddt = np.array([float(np.mean(values)) for values in plddt_by_residue.values()])
    return chain_coords, plddt


def pdockq_v1(pdb_text: str) -> dict[str, float]:
    """Compute the Bryant 2022 pDockQ for the first two chains of a complex.

    The sigmoid was fit on AlphaFold2-Multimer confidences. Applying it to
    AlphaFold 3 pLDDT is a calibration transfer that has not been validated, so
    this value is for comparison with the paper only and must not be used as a
    gate.

    Args:
        pdb_text: PDB text with per-residue pLDDT in the B-factor column.

    Returns:
        ``pdockq_v1``, ``if_plddt``, ``if_contacts``, and ``avg_plddt``. All
        zero when fewer than two chains are present or no interface contacts
        fall within 8 A.
    """
    chain_coords, plddt = _pdockq_v1_coordinates(pdb_text)
    chains = list(chain_coords)
    empty = {"pdockq_v1": 0.0, "if_plddt": 0.0, "if_contacts": 0, "avg_plddt": 0.0}
    if len(chains) < 2 or plddt.size == 0:
        return empty

    coords1 = np.array(chain_coords[chains[0]])
    coords2 = np.array(chain_coords[chains[1]])
    stacked = np.append(coords1, coords2, axis=0)
    differences = stacked[:, np.newaxis, :] - stacked[np.newaxis, :, :]
    distances = np.sqrt(np.sum(differences**2, axis=-1))
    split = len(coords1)
    contacts = np.argwhere(distances[:split, split:] <= PDOCKQ_V1_CONTACT_CUTOFF)
    if contacts.size == 0:
        return {**empty, "avg_plddt": float(plddt.mean())}

    interface_plddt = float(
        np.average(
            np.concatenate(
                [plddt[np.unique(contacts[:, 0])], plddt[np.unique(contacts[:, 1])]]
            )
        )
    )
    n_contacts = int(contacts.shape[0])
    x = interface_plddt * math.log10(n_contacts + 1)
    score = (
        PDOCKQ_V1_L / (1 + math.exp(-PDOCKQ_V1_K * (x - PDOCKQ_V1_X0))) + PDOCKQ_V1_B
    )
    return {
        "pdockq_v1": float(score),
        "if_plddt": interface_plddt,
        "if_contacts": n_contacts,
        "avg_plddt": float(plddt.mean()),
    }


# ---------------------------------------------------------------------------
# Monomer screen
# ---------------------------------------------------------------------------


class AlphaFold3MonomerScreenConfig(BaseConfig):
    """Configuration for :func:`af3_monomer_screen_constraint`.

    Attributes:
        qc_constraint_label: Label of the upstream QC constraint whose
            ``qc_proteins`` metadata supplies the proteins to fold.
        plddt_threshold: Minimum AlphaFold 3 ``avg_plddt``, on the 0-100 scale.
        ptm_threshold: Minimum AlphaFold 3 ``ptm``, on the 0-1 scale.
        min_surviving_proteins: Proteins that must clear both cutoffs for the
            proposal to pass.
        max_proteins_per_proposal: Cap on how many QC survivors are folded per
            proposal. Guards against one generation with many ORFs dominating
            the GPU budget. ``0`` disables the cap.
        hmm_constraint_label: Upstream profile-HMM constraint, when one ran.
            Proteins it marked ``qualifies`` are folded before the rest, so
            the cap cannot discard the protein that earned the generation its
            place. Ignored when that constraint produced no metadata.
        alphafold3: AlphaFold 3 execution settings.
    """

    qc_constraint_label: str = ConfigField(
        default="protein_qc",
        title="QC Constraint Label",
        description="Upstream constraint label supplying qc_proteins metadata.",
    )
    plddt_threshold: float = ConfigField(
        default=70.0,
        ge=0.0,
        le=100.0,
        title="Minimum pLDDT (0-100)",
        description="AlphaFold 3 avg_plddt floor. Not transferable from the paper's ESMFold cutoff.",
    )
    ptm_threshold: float = ConfigField(
        default=0.5,
        ge=0.0,
        le=1.0,
        title="Minimum pTM",
        description="AlphaFold 3 pTM floor.",
    )
    min_surviving_proteins: int = ConfigField(
        default=1,
        ge=1,
        title="Minimum Surviving Proteins",
        description="Proteins that must pass.",
    )
    max_proteins_per_proposal: int = ConfigField(
        default=8,
        ge=0,
        title="Max Proteins Folded",
        description="Per-proposal fold cap; 0 = no cap.",
    )
    prescreen_constraint_label: str = ConfigField(
        default="",
        title="Prescreen Constraint",
        description=(
            "Upstream sequence-only Acr scorer. When set, ORFs are folded in its "
            "score order and `fold_fraction` may skip the tail. Sequence callers "
            "cost ~5 s/protein against AlphaFold 3's minutes, so ranking first is "
            "nearly free: folding the top 50% retains 94% of known Acrs."
        ),
    )
    prescreen_min_score: float = ConfigField(
        default=0.0,
        ge=0.0,
        le=1.0,
        title="Prescreen Minimum",
        description=(
            "Fold only ORFs whose prescreen score reaches this. Calibrated: 0.10 "
            "keeps 100% of known Acrs (and 100% of the divergent subset) while "
            "skipping 53% of negatives. Deliberately lenient -- it exists to save "
            "folds, not to select. 0.0 disables."
        ),
    )
    fold_fraction: float = ConfigField(
        default=1.0,
        gt=0.0,
        le=1.0,
        title="Fraction To Fold",
        description=(
            "Fraction of each proposal's ORFs to fold, taken in prescreen order. "
            "1.0 folds everything. Requires prescreen_constraint_label; ignored "
            "without it, since folding an arbitrary subset would be worse than "
            "folding all."
        ),
    )
    hmm_constraint_label: str = ConfigField(
        default="profile_hmm",
        title="Profile-HMM Constraint Label",
        description=(
            "Upstream HMM constraint, if any. Proteins it marked as qualifying are "
            "folded first when the per-proposal cap bites."
        ),
    )
    alphafold3: AlphaFold3RunConfig = ConfigField(
        default_factory=lambda: AlphaFold3RunConfig(use_msa=False),
        title="AlphaFold 3 Settings",
        description="Execution settings; single-sequence by default for triage.",
    )


def _prescreen_scores(sequence: Any, label: str) -> dict[str, float]:
    """Per-protein scores from an upstream sequence-only Acr prescreen.

    Args:
        sequence: The proposal ``Sequence`` carrying constraint metadata.
        label: Label of the prescreen constraint, if one ran.

    Returns:
        ``{protein_id: score}``, empty when no prescreen ran -- in which case
        ordering falls back to HMM-qualifying-then-longest on its own.
    """
    if not label:
        return {}
    entry = (sequence._constraints_metadata or {}).get(label)
    if not entry:
        return {}
    return {
        record["protein_id"]: record.get("acr_locus_score") or 0.0
        for record in entry.get("data", {}).get("acr_evidence") or []
    }


def _hmm_qualifying_ids(sequence: Any, hmm_constraint_label: str) -> set[str]:
    """Protein IDs the profile-HMM constraint marked as qualifying.

    Args:
        sequence: The proposal ``Sequence`` carrying constraint metadata.
        hmm_constraint_label: Label of the HMM constraint, if it ran.

    Returns:
        The qualifying IDs, or an empty set when no HMM constraint ran --
        in which case ordering falls back to longest-first on its own.
    """
    entry = (sequence._constraints_metadata or {}).get(hmm_constraint_label)
    if not entry:
        return set()
    return {
        hit["protein_id"]
        for hit in entry.get("data", {}).get("hmm_hits") or []
        if hit.get("qualifies")
    }


@constraint(
    key="af3-monomer-screen",
    label="AlphaFold 3 Monomer Screen",
    config=AlphaFold3MonomerScreenConfig,
    description="Fold each QC-surviving ORF of a generation with AlphaFold 3 and gate on pLDDT/pTM.",
    tools_called=["alphafold3-prediction"],
    category="protein_structure",
    supported_sequence_types=["dna"],
    uses_gpu=True,
)
def af3_monomer_screen_constraint(
    input_sequences: list[tuple[Any, ...]],
    config: AlphaFold3MonomerScreenConfig,
) -> list[ConstraintOutput]:
    """Fold each QC-surviving protein with AlphaFold 3 and gate on pLDDT/pTM.

    Args:
        input_sequences: One tuple per proposal, each holding a DNA
            ``Sequence`` that already carries QC metadata from an upstream
            constraint.
        config: Screen thresholds and AlphaFold 3 settings.

    Returns:
        One ``ConstraintOutput`` per proposal. Score is ``0.0`` when at least
        ``min_surviving_proteins`` proteins clear both cutoffs. Metadata
        carries ``af3_proteins`` (the survivors, which drive downstream
        pairing), ``af3_folded_proteins`` (**every** folded protein, each with
        ``avg_plddt``, ``ptm``, ``avg_pae``, ``passed_af3_screen`` and, when
        structures are persisted, ``af3_result_dir``), ``af3_protein_count``,
        and ``af3_folded_count``.
    """
    outputs: list[ConstraintOutput] = []
    for proposal_index, sequence_tuple in enumerate(input_sequences):
        sequence = sequence_tuple[0]
        qc_entry = sequence._constraints_metadata.get(config.qc_constraint_label)
        if qc_entry is None:
            raise KeyError(
                f"af3_monomer_screen: no metadata from constraint "
                f"{config.qc_constraint_label!r}; declare it before this screen."
            )
        proteins: list[dict[str, Any]] = (
            qc_entry.get("data", {}).get("qc_proteins") or []
        )

        if config.max_proteins_per_proposal:
            # Longest-first is only a tiebreak. A generation reaches this
            # screen because some protein looked TA-like to the HMM filter,
            # and that protein is often the shorter one -- capping on length
            # alone can fold two unremarkable ORFs and discard the only
            # candidate with any family signal, which is what the gate was
            # for. Qualifying proteins therefore sort first.
            qualifying = _hmm_qualifying_ids(sequence, config.hmm_constraint_label)
            prescore = _prescreen_scores(sequence, config.prescreen_constraint_label)
            proteins = sorted(
                proteins,
                key=lambda entry: (
                    entry["protein_id"] not in qualifying,
                    -prescore.get(entry["protein_id"], 0.0),
                    -entry["length"],
                ),
            )
            dropped = proteins[config.max_proteins_per_proposal :]
            proteins = proteins[: config.max_proteins_per_proposal]
        if config.prescreen_constraint_label and config.prescreen_min_score > 0.0:
            prescore = _prescreen_scores(sequence, config.prescreen_constraint_label)
            keep = [
                p
                for p in proteins
                if prescore.get(p["protein_id"], 1.0) >= config.prescreen_min_score
                or p["protein_id"]
                in _hmm_qualifying_ids(sequence, config.hmm_constraint_label)
            ]
            if len(keep) < len(proteins):
                logger.info(
                    "af3_monomer_screen: prescreen kept %d/%d ORFs at >=%.2f "
                    "(HMM-qualifying ORFs are always folded)",
                    len(keep),
                    len(proteins),
                    config.prescreen_min_score,
                )
            # A proposal with nothing above threshold still folds its best ORF:
            # returning no structures at all would look like a pipeline failure
            # rather than a weak generation.
            proteins = keep or proteins[:1]
        if config.prescreen_constraint_label and config.fold_fraction < 1.0:
            # Ordered by prescreen above, so the tail is the least Acr-like.
            keep = max(1, math.ceil(len(proteins) * config.fold_fraction))
            if keep < len(proteins):
                logger.info(
                    "af3_monomer_screen: folding %d/%d ORFs (fold_fraction=%.2f); "
                    "skipped ORFs get no structure and score on sequence alone",
                    keep,
                    len(proteins),
                    config.fold_fraction,
                )
            proteins = proteins[:keep]
        if config.max_proteins_per_proposal:
            dropped_qualifying = [
                entry["protein_id"]
                for entry in dropped
                if entry["protein_id"] in qualifying
            ]
            if dropped_qualifying:
                logger.warning(
                    "af3_monomer_screen: fold cap of %d dropped HMM-qualifying "
                    "protein(s) %s; raise af3_max_proteins_per_proposal to keep them.",
                    config.max_proteins_per_proposal,
                    ", ".join(dropped_qualifying),
                )

        folded: list[dict[str, Any]] = []
        survivors: list[dict[str, Any]] = []
        for protein in proteins:
            job_name = job_name_for("monomer", [protein["sequence"]])
            cache = _cache_path(config.alphafold3, job_name)
            confidences = _load_cached(cache)
            if confidences is None:
                structure = fold([protein["sequence"]], job_name, config.alphafold3)
                confidences = {
                    "avg_plddt": structure.metrics.get("avg_plddt"),
                    "ptm": structure.metrics.get("ptm"),
                    "avg_pae": structure.metrics.get("avg_pae"),
                }
                # Only the raw confidences are cached, never the verdict: the
                # thresholds are config, so a rerun with tighter gates must
                # re-decide rather than replay the old pass/fail.
                _store_cached(cache, confidences)
            avg_plddt = confidences.get("avg_plddt")
            ptm = confidences.get("ptm")
            passed = (
                avg_plddt is not None
                and ptm is not None
                and avg_plddt >= config.plddt_threshold
                and ptm >= config.ptm_threshold
            )
            record = {
                **protein,
                "avg_plddt": avg_plddt,
                "ptm": ptm,
                "avg_pae": confidences.get("avg_pae"),
                "af3_job": job_name,
                "passed_af3_screen": passed,
            }
            if config.alphafold3.output_dir is not None:
                record["af3_result_dir"] = (
                    f"{config.alphafold3.output_dir}/{job_name}_af3_results"
                )
                # Downstream structural callers need the PDB itself, not the
                # folder. AlphaFold 3 appends .1/.2 when a directory already
                # exists, so the path is globbed rather than constructed.
                matches = sorted(
                    Path(config.alphafold3.output_dir).glob(
                        f"{job_name}_af3_results*/{job_name}_0_af3.pdb"
                    )
                )
                record["structure_path"] = str(matches[-1]) if matches else None
            folded.append(record)
            if passed:
                survivors.append(record)

        if proteins and not survivors:
            logger.info(
                "af3_monomer_screen: proposal %d folded %d protein(s), none met "
                "pLDDT >= %.1f and pTM >= %.2f",
                proposal_index,
                len(proteins),
                config.plddt_threshold,
                config.ptm_threshold,
            )

        outputs.append(
            ConstraintOutput(
                score=0.0 if len(survivors) >= config.min_surviving_proteins else 1.0,
                metadata={
                    # Survivors drive the downstream pairing; every folded
                    # protein is kept alongside them so a run can be used to
                    # choose thresholds. Discarding the failures would make the
                    # "fold first, then set cutoffs" workflow impossible.
                    "af3_proteins": survivors or None,
                    "af3_folded_proteins": folded or None,
                    "af3_protein_count": len(survivors),
                    "af3_folded_count": len(proteins),
                },
            )
        )
    return outputs


# ---------------------------------------------------------------------------
# Complex screen
# ---------------------------------------------------------------------------


def score_complex(
    chain_a: str,
    chain_b: str,
    job_name: str,
    run_config: AlphaFold3RunConfig,
    pdockq2_distance_cutoff: float = 8.0,
) -> dict[str, Any]:
    """Cofold two protein chains with AlphaFold 3 and score the interface.

    Args:
        chain_a: Sequence assigned chain A.
        chain_b: Sequence assigned chain B.
        job_name: Filesystem-safe name for this prediction.
        run_config: AlphaFold 3 execution settings.
        pdockq2_distance_cutoff: CA-CA cutoff in angstroms for pDockQ2's
            interface definition. Defaults to 8.0, the value the published
            pDockQ2 sigmoid was calibrated at (proto-tools' own wrapper
            defaults to the more permissive 10.0).

    Returns:
        A flat dict of interface metrics: ``iptm``, ``ptm``, ``avg_plddt``,
        ``avg_pae``, ``pdockq2``, ``pdockq2_if_plddt``, ``pdockq2_if_pae``,
        ``pdockq2_contacts``, plus the reported-only ``pdockq_v1``,
        ``pdockq_v1_if_plddt``, and ``pdockq_v1_if_contacts``.
    """
    cache = _cache_path(run_config, job_name)
    cached = _load_cached(cache)
    if cached is not None:
        logger.info("Reusing cached score for %s", job_name)
        return cached

    structure = fold([chain_a, chain_b], job_name, run_config)
    chain_ids = structure.get_chain_ids()
    if len(chain_ids) < 2:
        raise RuntimeError(
            f"AlphaFold 3 job {job_name!r} returned {len(chain_ids)} chain(s); expected 2"
        )

    pdockq2 = run_pdockq2(
        PDockQ2Input(
            structure=structure,
            binder_chain=SingleChainSelection(chain=chain_ids[1]),
            target_chains=ChainSelection(chains=[chain_ids[0]]),
        ),
        PDockQ2Config(distance_cutoff=pdockq2_distance_cutoff),
    )
    legacy = pdockq_v1(structure.structure_pdb)

    result: dict[str, Any] = {
        "af3_job": job_name,
        "iptm": structure.metrics.get("iptm"),
        "ptm": structure.metrics.get("ptm"),
        "avg_plddt": structure.metrics.get("avg_plddt"),
        "avg_pae": structure.metrics.get("avg_pae"),
        "pdockq2": pdockq2.metrics.get("pdockq2"),
        "pdockq2_if_plddt": pdockq2.metrics.get("avg_interface_plddt"),
        "pdockq2_if_pae": pdockq2.metrics.get("avg_interface_pae"),
        "pdockq2_contacts": pdockq2.metrics.get("num_interface_contacts"),
        "pdockq_v1": legacy["pdockq_v1"],
        "pdockq_v1_if_plddt": legacy["if_plddt"],
        "pdockq_v1_if_contacts": legacy["if_contacts"],
    }
    if run_config.output_dir is not None:
        result["af3_result_dir"] = f"{run_config.output_dir}/{job_name}_af3_results"
    _store_cached(cache, result)
    return result
