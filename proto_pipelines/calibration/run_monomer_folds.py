"""Fold single chains with AlphaFold 3 and record pLDDT/pTM, shardable.

Runs **single-sequence** by default because that is how
``af3_monomer_screen`` runs inside the pipelines. A threshold measured with
an MSA would not transfer: MSA-mode pLDDT is substantially higher.

Production fidelity (``num_recycles=10``, ``num_diffusion_samples=5``) is the
default here on purpose. Earlier smoke-run numbers were taken at 3/1, which
depresses confidence, so a threshold set from those would be too permissive.

Usage:
    python -m proto_pipelines.calibration.run_monomer_folds \
        --chains monomer_chains.csv --out-dir results/monomer --shard 0 --num-shards 4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from proto_pipelines.af3 import AlphaFold3RunConfig, fold, job_name_for

logger = logging.getLogger(__name__)


def run_shard(
    chains: pd.DataFrame, shard: int, num_shards: int,
    run_config: AlphaFold3RunConfig, out_path: Path,
) -> pd.DataFrame:
    """Fold this shard's chains, writing after each so a preemption resumes.

    Args:
        chains: Full labelled chain set.
        shard: Zero-based shard index.
        num_shards: Total shards.
        run_config: AlphaFold 3 settings.
        out_path: Destination CSV.

    Returns:
        Rows this shard folded.
    """
    slice_ = chains.iloc[shard::num_shards].reset_index(drop=True)
    logger.info("Shard %d/%d: %d of %d chain(s)", shard, num_shards, len(slice_), len(chains))

    rows: list[dict[str, Any]] = []
    done: set[str] = set()
    if out_path.exists():
        previous = pd.read_csv(out_path)
        rows = previous.to_dict("records")
        done = set(previous["chain_uid"].astype(str))
        logger.info("Resuming: %d chain(s) already folded", len(done))

    for position, row in enumerate(slice_.itertuples(), start=1):
        if str(row.chain_uid) in done:
            continue
        job_name = job_name_for("mono", [row.sequence])
        print(
            f"[shard {shard}] {position}/{len(slice_)} {row.chain_class} "
            f"{row.chain_uid} ({row.length} aa)",
            flush=True,
        )
        structure = fold([row.sequence], job_name, run_config)
        record = {**row._asdict()}
        record.pop("Index", None)
        record.update(
            {
                "avg_plddt": structure.metrics.get("avg_plddt"),
                "ptm": structure.metrics.get("ptm"),
                "avg_pae": structure.metrics.get("avg_pae"),
                "af3_job": job_name,
            }
        )
        rows.append(record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def main() -> None:
    """Fold one shard of the monomer calibration set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chains", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--use-msa", action="store_true", help="Not how the screen runs; for comparison only.")
    parser.add_argument("--msa-device", default="cuda")
    parser.add_argument("--num-recycles", type=int, default=10)
    parser.add_argument("--num-diffusion-samples", type=int, default=5)
    args = parser.parse_args()

    chains = pd.read_csv(args.chains)
    for column in ("chain_uid", "sequence", "label", "chain_class"):
        if column not in chains.columns:
            raise ValueError(f"{args.chains} is missing the {column!r} column")

    run_config = AlphaFold3RunConfig(
        use_msa=args.use_msa,
        msa_device=args.msa_device,
        num_recycles=args.num_recycles,
        num_diffusion_samples=args.num_diffusion_samples,
        output_dir=str(args.out_dir / "structures"),
        verbose=False,
    )
    out_path = args.out_dir / f"monomer_shard{args.shard:02d}.csv"
    folded = run_shard(chains, args.shard, args.num_shards, run_config, out_path)
    logger.info("Shard %d wrote %d row(s)", args.shard, len(folded))


if __name__ == "__main__":
    main()
