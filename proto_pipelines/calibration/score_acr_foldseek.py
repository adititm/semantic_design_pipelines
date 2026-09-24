"""Score the Acr calibration set by structural similarity to known Acr folds.

Structure is the one signal that should survive sequence divergence: a
composition-matched shuffle of an Acr has the same residues in a different
order and should not fold like the parent. That makes Foldseek the natural
complement to AcRanker, whose discrimination is composition-driven, and to
the profile HMMs, which only recognise the handful of Acr families Pfam
covers.

**Self-hits must be excluded or the result is circular.** The positives are
the known Acrs, and the reference database is built from their deposited
structures, so an unfiltered search scores every positive ~1.0 against its
own crystal structure. Any hit whose target sequence is more than
``--max-self-identity`` identical to the query is therefore dropped, which
turns the question into "does this fold resemble some *other* Acr" -- the
question that actually transfers to a novel candidate.

Usage:
    python -m proto_pipelines.calibration.score_acr_foldseek \
        --chains acr_monomer_chains.csv --structures <fold dir> \
        --ref-dir ref_acr_chains --out acr_foldseek_scores.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proto_pipelines.calibration.score_acr_hmm import auroc

logger = logging.getLogger(__name__)


def identity(a: str, b: str) -> float:
    """Ungapped identity over the overlap of two sequences, as a fraction.

    Deliberately crude: this only has to catch a query matching its own
    deposited structure, where the sequences are effectively the same, not
    align remote homologues.

    Args:
        a: First sequence.
        b: Second sequence.

    Returns:
        Fraction identical over the shorter length, ``0.0`` if either is empty.
    """
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n], strict=False)) / n


def main() -> None:
    """Search every folded query against the Acr reference and report AUROC."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chains", required=True, type=Path,
                        help="acr_monomer_chains.csv, for labels and sequences.")
    parser.add_argument("--structures", required=True, type=Path,
                        help="Directory of folded query structures.")
    parser.add_argument("--ref-dir", required=True, type=Path,
                        help="Directory of reference Acr chain PDBs.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-self-identity", type=float, default=0.90,
                        help="Drop hits at or above this identity to the query.")
    parser.add_argument("--tmscore-threshold", type=float, default=0.0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Each proto call rebuilds the target DB, so a "
                             "full sweep is slow; shard it across an array.")
    args = parser.parse_args()

    from proto_tools import FoldseekSearchConfig, FoldseekSearchInput, run_foldseek_search

    chains = pd.read_csv(args.chains)

    # AlphaFold 3 names its outputs by the content hash of the sequence, not
    # by chain_uid, so the fold-result CSVs are the only thing that maps a
    # chain to its structure. Joining on filename silently matches nothing.
    shards = sorted(args.structures.glob("*.csv"))
    if not shards:
        raise FileNotFoundError(f"no fold-result CSVs in {args.structures}")
    folded = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    if "af3_job" not in folded.columns:
        raise ValueError(f"fold results in {args.structures} carry no af3_job column")
    job_of = dict(zip(folded["chain_uid"].astype(str), folded["af3_job"], strict=False))

    by_job: dict[str, Path] = {}
    for path in args.structures.rglob("*_af3.pdb"):
        by_job.setdefault(path.name.split("_0_af3")[0], path)
    structures = {
        uid: by_job[job] for uid, job in job_of.items() if job in by_job
    }
    if not structures:
        raise FileNotFoundError(
            f"no structure matched any af3_job under {args.structures}"
        )
    logger.info("Mapped %d/%d chains to structures via af3_job",
                len(structures), len(chains))

    config = FoldseekSearchConfig(
        search_mode="local", local_db=str(args.ref_dir),
        tmscore_threshold=args.tmscore_threshold, evalue=10.0, max_seqs=500,
    )

    if args.num_shards > 1:
        chains = chains.iloc[args.shard :: args.num_shards].reset_index(drop=True)
        logger.info("Shard %d/%d: %d chain(s)", args.shard, args.num_shards, len(chains))

    rows: list[dict[str, Any]] = []
    for position, row in enumerate(chains.itertuples(), start=1):
        path = structures.get(str(row.chain_uid))
        if path is None:
            rows.append({"chain_uid": row.chain_uid, "n_hits": None,
                         "best_tmscore": None, "best_target": None})
            continue
        result = run_foldseek_search(
            FoldseekSearchInput(structure=str(path)), config
        )
        # Foldseek reports sequence_identity per hit, so self-hits are
        # excluded exactly rather than by re-deriving identity here.
        kept = [
            hit for hit in (result.hits or [])
            if (hit.sequence_identity or 0.0) < args.max_self_identity
        ]
        best = max(kept, key=lambda h: h.query_tm_score or 0.0, default=None)
        rows.append({
            "chain_uid": row.chain_uid,
            "n_hits": len(kept),
            "n_hits_raw": len(result.hits or []),
            # query_tm_score normalises by the query, which is the right
            # question here: how much of *this* protein looks like a known Acr.
            "best_tmscore": (best.query_tm_score if best else 0.0) or 0.0,
            "best_aln_tm": (best.alignment_tm_score if best else 0.0) or 0.0,
            "best_evalue": best.evalue if best else None,
            "best_target": best.target_id if best else None,
            "best_seqid": best.sequence_identity if best else None,
        })
        if position % 5 == 0:
            logger.info("searched %d/%d", position, len(chains))
            # Flush incrementally: a preempted shard keeps what it scored.
            pd.DataFrame(rows).to_csv(args.out, index=False)

    frame = chains.merge(pd.DataFrame(rows), on="chain_uid", how="left")
    frame["best_tmscore"] = frame["best_tmscore"].fillna(0.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    positives = frame.loc[frame["label"] == 1, "best_tmscore"].to_numpy()
    print(f"\n  self-hits excluded at >={args.max_self_identity:.0%} identity")
    print(f"  {'class':<12} {'n':>4} {'median TM':>10} {'any hit':>8} {'AUROC':>7}")
    for name, group in frame.groupby("chain_class"):
        tm = group["best_tmscore"].to_numpy()
        au = "-" if name == "acr" else f"{auroc(positives, tm):.3f}"
        print(f"  {name:<12} {len(group):>4} {np.median(tm):>10.3f} "
              f"{(tm > 0).mean():>7.0%} {au:>7}")


if __name__ == "__main__":
    main()
