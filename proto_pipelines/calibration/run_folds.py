"""Score a labelled pair set with AlphaFold 3 + pDockQ2, shardable across jobs.

Scoring goes through :func:`proto_pipelines.af3.score_complex`, the same
function the cofold constraint calls during a design run -- a threshold derived
here is only meaningful if it was measured the way it will be applied.

Sharding exists because AlphaFold 3 with an MSA is slow and the pairs are
independent: ``--num-shards N`` splits the set N ways by row index and
``--shard i`` runs one slice, so a SLURM array covers the set in parallel.
Each shard writes its own CSV; :mod:`proto_pipelines.calibration.analyze`
concatenates them.

Usage:
    python -m proto_pipelines.calibration.run_folds \
        --pairs calibration_pairs_pilot.csv --out-dir results/ \
        --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from proto_pipelines.af3 import AlphaFold3RunConfig, job_name_for, score_complex

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("root_id", "protein_uid_1", "sequence_1", "protein_uid_2", "sequence_2")


def load_pairs(
    path: Path,
    min_residues: int = 0,
    max_residues: int = 10_000,
    max_pairs: int = 0,
) -> pd.DataFrame:
    """Read a pair CSV and drop complexes outside the size window.

    AlphaFold 3's cost scales steeply with total length, so very long pairs are
    excluded rather than left to stall a shard. Filtering is logged, not
    silent: a set that loses most of its rows here would otherwise look like a
    set that simply scored badly.

    Args:
        path: CSV with ``root_id``, ``protein_uid_1``/``sequence_1`` and
            ``protein_uid_2``/``sequence_2``.
        min_residues: Reject pairs shorter than this in total.
        max_residues: Reject pairs longer than this in total.
        max_pairs: Keep at most this many rows; ``0`` keeps all.

    Returns:
        The surviving pairs with a ``total_residues`` column added.

    Raises:
        ValueError: If a required column is missing, or every row is filtered
            out -- an empty pair set is a configuration error, not a result.
    """
    pairs = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in pairs.columns]
    if missing:
        raise ValueError(f"{path} is missing required column(s): {', '.join(missing)}")

    pairs["total_residues"] = (
        pairs["sequence_1"].str.len() + pairs["sequence_2"].str.len()
    )
    in_window = pairs["total_residues"].between(min_residues, max_residues)
    if not in_window.all():
        logger.warning(
            "Dropping %d of %d pair(s) outside [%d, %d] residues",
            (~in_window).sum(),
            len(pairs),
            min_residues,
            max_residues,
        )
    pairs = pairs[in_window].reset_index(drop=True)
    if pairs.empty:
        raise ValueError(
            f"No pairs left in {path} after the [{min_residues}, {max_residues}] "
            "residue filter; widen the window or check the input."
        )
    if max_pairs:
        pairs = pairs.head(max_pairs)
    return pairs


def run_shard(
    pairs: pd.DataFrame,
    shard: int,
    num_shards: int,
    run_config: AlphaFold3RunConfig,
    distance_cutoff: float,
    out_path: Path,
) -> pd.DataFrame:
    """Score this shard's slice of the pair set, writing incrementally.

    Results are flushed after every pair rather than at the end: an AlphaFold 3
    sweep is long enough that a preempted or timed-out job would otherwise lose
    everything it had computed.

    Args:
        pairs: The full labelled pair set.
        shard: Zero-based shard index.
        num_shards: Total shards.
        run_config: AlphaFold 3 execution settings.
        distance_cutoff: CA-CA cutoff for pDockQ2's interface definition.
        out_path: Destination CSV for this shard.

    Returns:
        The rows this shard scored.
    """
    slice_ = pairs.iloc[shard::num_shards].reset_index(drop=True)
    logger.info("Shard %d/%d: %d of %d pair(s)", shard, num_shards, len(slice_), len(pairs))

    done: set[str] = set()
    rows: list[dict[str, Any]] = []
    if out_path.exists():
        previous = pd.read_csv(out_path)
        rows = previous.to_dict("records")
        done = set(previous["root_id"].astype(str))
        logger.info("Resuming: %d pair(s) already scored in %s", len(done), out_path)

    for position, row in enumerate(slice_.itertuples(), start=1):
        if str(row.root_id) in done:
            continue
        job_name = job_name_for("cal", [row.sequence_1, row.sequence_2])
        print(
            f"[shard {shard}] {position}/{len(slice_)} {row.pair_class} "
            f"{row.protein_uid_1}+{row.protein_uid_2} ({row.total_residues} res)",
            flush=True,
        )
        metrics = score_complex(
            chain_a=row.sequence_1,
            chain_b=row.sequence_2,
            job_name=job_name,
            run_config=run_config,
            pdockq2_distance_cutoff=distance_cutoff,
        )
        record = {**row._asdict(), **metrics}
        record.pop("Index", None)
        rows.append(record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def main() -> None:
    """Parse arguments and score one shard."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", required=True, type=Path, help="Labelled pairs CSV")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for shard CSVs")
    parser.add_argument("--shard", type=int, default=0, help="Zero-based shard index")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--no-msa", action="store_true", help="Run single-sequence (much faster)")
    parser.add_argument(
        "--msa-device",
        default="cpu",
        help="MMseqs2 search device; 'cuda' uses MMseqs2-GPU (needs a .idx_pad index).",
    )
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--num-diffusion-samples", type=int, default=5)
    parser.add_argument("--distance-cutoff", type=float, default=8.0)
    args = parser.parse_args()

    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"--shard must be in [0, {args.num_shards}), got {args.shard}")

    pairs = load_pairs(args.pairs, min_residues=0, max_residues=10_000, max_pairs=0)
    for column in ("label", "pair_class"):
        if column not in pairs.columns:
            raise ValueError(f"{args.pairs} is missing the {column!r} column; not a labelled set.")

    run_config = AlphaFold3RunConfig(
        use_msa=not args.no_msa,
        pair_heterocomplex_msas=not args.no_msa,
        msa_device=args.msa_device,
        num_recycles=args.num_recycles,
        num_diffusion_samples=args.num_diffusion_samples,
        output_dir=str(args.out_dir / "structures"),
        verbose=False,
    )

    out_path = args.out_dir / f"scores_shard{args.shard:02d}.csv"
    scored = run_shard(
        pairs, args.shard, args.num_shards, run_config, args.distance_cutoff, out_path
    )
    logger.info("Shard %d wrote %d row(s) to %s", args.shard, len(scored), out_path)


if __name__ == "__main__":
    main()
