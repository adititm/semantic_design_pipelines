"""AcrNET anti-CRISPR scoring, with the feature stack reproduced locally.

AcrNET consumes four inputs that its authors generated with web services:
RaptorX secondary structure and solvent accessibility, a PSI-BLAST PSSM
condensed into four POSSUM descriptors, and an ESM-1b embedding. All three
run offline here, and the substitution was validated end to end on AcrNET's
own 20-sequence test set: locally generated RaptorX + PSI-BLAST reproduces
the published 1.00 accuracy exactly (descriptor correlation with the shipped
features r=0.83-0.97).

Dropping the PSSM costs accuracy (1.00 -> 0.90) but removes the only
expensive step, so ``blast_db`` is optional. An MMseqs2-derived PSSM was
tried and is *worse than zeros* (0.80): the profile differs from PSI-BLAST in
content, not just scale, and the model is sensitive to that.

**Batching is what makes this affordable in-loop.** The optimizer hands a
whole batch of proposals to a constraint at once, so every protein across
every proposal is extracted together: RaptorX in a process pool (0.74 s each,
one core), PSI-BLAST in a process pool of few-thread workers (16 s at 16
threads but only 23 s at 4, so processes beat threads by ~3x), and ESM-1b in
one GPU pass with the model loaded once. Against AlphaFold 3 at 1-2 min per
protein this is roughly 5% overhead.

Scored on the 316-sequence calibration set, AcrNET reaches 0.883 AUROC
against all negatives and **0.874 on divergent Acrs** -- the best divergent
signal available, against 0.775 for the HMM/AcRanker/Foldseek tier. It is
still not a gate: it calls 77% of composition-matched shuffles anti-CRISPR
at median 0.937, and real Acrs beat their own shuffles by a median of only
+0.041. Use it to rank, never to threshold.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SS3_INDEX = {"C": 0, "E": 1, "H": 2}
SS8_INDEX = {"L": 0, "H": 1, "T": 2, "E": 3, "S": 4, "G": 5, "B": 6, "I": 7}
ACC_INDEX = {"E": 0, "B": 1, "M": 2}
AA_INDEX = {c: i for i, c in enumerate("AVGILFPYMTSHNQWRKDEC")}
PSSM_DESCRIPTORS = ("dpc_pssm", "pssm_ac", "pssm_composition", "rpssm")


def _read_simp(path: Path) -> str:
    """Third line of a Predict_Property ``*_simp`` file, or empty if absent."""
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    return lines[2].strip() if len(lines) > 2 else ""


def _run_raptorx(args: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    """Predict ss3/ss8/acc for one sequence with Predict_Property.

    Runs in no-profile mode: it needs no database, and on AcrNET's own test
    set the resulting features leave the model's predictions unchanged.

    Args:
        args: ``(protein_id, sequence, predict_property_home, work_dir)``.

    Returns:
        ``(protein_id, ss3, ss8, acc)``; the strings are empty on failure,
        which the caller treats as "cannot score" rather than as a result.
    """
    protein_id, sequence, home, work = args
    out = Path(work) / protein_id
    fasta = Path(work) / f"{protein_id}.fasta"
    fasta.write_text(f">{protein_id}\n{sequence}\n")
    try:
        subprocess.run(
            ["bash", str(Path(home) / "Predict_Property.sh"), "-i", str(fasta), "-o", str(out)],
            check=True, capture_output=True, timeout=300, cwd=home,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        logger.warning("Predict_Property failed for %s: %s", protein_id, type(error).__name__)
        return protein_id, "", "", ""
    return (
        protein_id,
        _read_simp(out / f"{protein_id}.ss3_simp"),
        _read_simp(out / f"{protein_id}.ss8_simp"),
        _read_simp(out / f"{protein_id}.acc_simp"),
    )


def _run_psiblast(args: tuple[str, str, str, str, str, int]) -> tuple[str, str]:
    """Build one PSI-BLAST PSSM.

    Args:
        args: ``(protein_id, sequence, psiblast_bin, blast_db, work_dir,
            threads)``. ``threads`` is deliberately small: PSI-BLAST scales
            poorly past ~4 threads, so throughput comes from running many
            processes rather than one wide one.

    Returns:
        ``(protein_id, pssm_path)``; the path is empty when PSI-BLAST found
        no hits, which happens for genuinely novel sequence.
    """
    protein_id, sequence, binary, database, work, threads = args
    fasta = Path(work) / f"{protein_id}.pb.fasta"
    pssm = Path(work) / f"{protein_id}.pssm"
    fasta.write_text(f">{protein_id}\n{sequence}\n")
    try:
        subprocess.run(
            [binary, "-query", str(fasta), "-db", database, "-num_iterations", "3",
             "-out_ascii_pssm", str(pssm), "-out", "/dev/null",
             "-num_threads", str(threads), "-evalue", "0.001"],
            check=True, capture_output=True, timeout=600,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        logger.warning("psiblast failed for %s: %s", protein_id, type(error).__name__)
        return protein_id, ""
    return protein_id, (str(pssm) if pssm.exists() else "")


def extract_features(
    proteins: list[dict[str, str]],
    predict_property_home: str,
    esm_device: str = "cuda",
    psiblast_bin: str = "",
    blast_db: str = "",
    workers: int = 8,
    psiblast_threads: int = 4,
) -> dict[str, dict[str, Any]]:
    """Build every AcrNET input for a whole batch of proteins at once.

    Args:
        proteins: ``{"protein_id", "sequence"}`` dicts, across all proposals.
        predict_property_home: Checkout of RaptorX Predict_Property.
        esm_device: Device for ESM-1b.
        psiblast_bin: ``psiblast`` executable; empty disables the PSSM and
            zeroes that feature block.
        blast_db: BLAST database prefix; empty disables the PSSM.
        workers: Concurrent RaptorX / PSI-BLAST processes.
        psiblast_threads: Threads per PSI-BLAST process.

    Returns:
        ``{protein_id: {"ss3", "ss8", "acc", "pssm", "embedding"}}``, omitting
        any protein whose secondary structure could not be predicted.
    """
    import numpy as np

    out: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as work:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [(p["protein_id"], p["sequence"], predict_property_home, work)
                    for p in proteins]
            for protein_id, ss3, ss8, acc in pool.map(_run_raptorx, jobs):
                if ss3:
                    out[protein_id] = {"ss3": ss3, "ss8": ss8, "acc": acc,
                                       "pssm": np.zeros(1110, dtype="float32")}
        logger.info("RaptorX: %d/%d proteins", len(out), len(proteins))

        if psiblast_bin and blast_db:
            import pssmpro.features as descriptors

            with ProcessPoolExecutor(max_workers=workers) as pool:
                jobs = [(p["protein_id"], p["sequence"], psiblast_bin, blast_db,
                         work, psiblast_threads)
                        for p in proteins if p["protein_id"] in out]
                built = 0
                for protein_id, path in pool.map(_run_psiblast, jobs):
                    if not path:
                        # Leaves this protein's 1110-dim PSSM block zeroed,
                        # which is a real state (no PSI-BLAST hits) but must
                        # not be silent -- it costs ~0.10 accuracy.
                        logger.warning(
                            "PSI-BLAST produced no PSSM for %s; its PSSM block "
                            "stays zeroed", protein_id,
                        )
                        continue
                    matrix = descriptors.read_pssm_matrix(path)
                    out[protein_id]["pssm"] = np.concatenate([
                        np.asarray(getattr(descriptors, d)(matrix)).ravel()
                        for d in PSSM_DESCRIPTORS
                    ])
                    built += 1
            logger.info("PSI-BLAST: %d/%d PSSMs (no hits leaves the block zeroed)",
                        built, len(out))

    if out:
        import esm as esm_lib
        import torch

        model, alphabet = esm_lib.pretrained.esm1b_t33_650M_UR50S()
        model = model.eval().to(esm_device)
        convert = alphabet.get_batch_converter()
        ids = list(out)
        by_id = {p["protein_id"]: p["sequence"] for p in proteins}
        for start in range(0, len(ids), 8):
            chunk = ids[start : start + 8]
            # ESM-1b's positional embedding stops at 1022 residues.
            batch = [(i, by_id[i][:1022]) for i in chunk]
            _, _, tokens = convert(batch)
            with torch.no_grad():
                reps = model(tokens.to(esm_device), repr_layers=[33])["representations"][33]
            for row, (protein_id, sequence) in enumerate(batch):
                out[protein_id]["embedding"] = (
                    reps[row, 1 : len(sequence) + 1].mean(0).cpu().numpy()
                )
    return out


def score(features: dict[str, dict[str, Any]], model_checkpoint: str) -> dict[str, float]:
    """Run AcrNET over pre-extracted features.

    Args:
        features: Output of :func:`extract_features`.
        model_checkpoint: AcrNET ``model.ckpt``.

    Returns:
        ``{protein_id: probability of the anti-CRISPR class}``.
    """
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pad_sequence

    from proto_pipelines.acrnet_model import AcrNET

    model = AcrNET()
    model.load_state_dict(torch.load(model_checkpoint, map_location="cpu"))
    model.eval()

    ids = [i for i, f in features.items() if "embedding" in f]
    scores: dict[str, float] = {}

    # Each protein is scored on its own, with no cross-protein padding.
    # AcrNET pads with index 0, which ``one_hot`` turns into a *valid* residue
    # ('A'/'C'/'L'/'E') rather than zeros, so padded positions enter the
    # convolution and its max-pool. Batching therefore made a protein's score
    # depend on which other proteins shared its batch: measured on the shipped
    # checkpoint, chunk-local padding moved the 20-dim CNN feature by up to
    # 0.304 and the final probability by up to 0.099 (median 0.0, p90 0.006).
    # Padding to a per-call global width does not fix it either, because that
    # width still depends on the set being scored.
    #
    # This deliberately diverges from upstream ``test.py``, which pads its
    # whole dataset in one call and so inherits the same dependence. A
    # pipeline scores arbitrary, varying sets of ORFs, so the score has to be
    # a function of the protein alone. The model is small enough that
    # one forward pass per protein is not a measurable cost next to RaptorX,
    # PSI-BLAST and ESM-1b.
    #
    # AcrNET's forward() calls an unconditional .squeeze() and then reads
    # size(2), so a batch of one loses its batch dimension and raises
    # IndexError; each protein is duplicated and the copy discarded.
    def one_hot(rows: list[Any], classes: int) -> Any:
        padded = pad_sequence(rows, batch_first=True)
        return functional.one_hot(torch.unsqueeze(padded, 1), num_classes=classes).float()

    for protein_id in ids:
        f = features[protein_id]
        length = len(f["ss3"])
        sequence = f.get("sequence", "A" * length)[:length]
        pair = lambda row: [torch.tensor(row), torch.tensor(row)]  # noqa: E731
        with torch.no_grad():
            output = model(
                seq=one_hot(pair([AA_INDEX.get(c, 0) for c in sequence]), 20),
                ss3=one_hot(pair([SS3_INDEX.get(c, 0) for c in f["ss3"]]), 3),
                ss8=one_hot(pair([SS8_INDEX.get(c, 0) for c in f["ss8"]]), 8),
                acc=one_hot(pair([ACC_INDEX.get(c, 2) for c in f["acc"]]), 3),
                dnn_feature=torch.tensor(
                    np.vstack([np.concatenate([f["pssm"], f["embedding"]])] * 2)
                ).float(),
            )
        scores[protein_id] = float(torch.exp(output)[0, 1])
    return scores
