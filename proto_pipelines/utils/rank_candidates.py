"""Turn a ranked Acr candidate list into a defensible top-k selection.

``acr_locus_score`` is a logistic output, not a probability of being an
anti-CRISPR: it was fit at the calibration set's own class balance, which is
not your candidate pool's. This maps it through an isotonic calibration
fitted on out-of-fold scores, then rescales to whatever prior you actually
expect, so ``P(Acr)`` means something.

The useful quantity for choosing k is the **expected number of true Acrs in
the top k**, which is just the sum of calibrated probabilities over those k.
Pick k by the yield you want to test rather than by a score cutoff -- a
cutoff is exactly what the calibration says not to trust.

Usage:
    python -m proto_pipelines.utils.rank_candidates \
        --evidence acr_evidence.csv --prior 0.05 --out ranked.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CALIBRATION = Path("proto_pipelines/data/models/acr/acr_score_calibration.json")


def rescale_prior(probability: np.ndarray, fitted: float, target: float) -> np.ndarray:
    """Move a calibrated probability from one class prior to another.

    Args:
        probability: ``P(Acr | score)`` at the prior it was fitted on.
        fitted: Class prior of the calibration set.
        target: Expected Acr fraction in the pool being scored.

    Returns:
        ``P(Acr | score)`` at ``target``, via the prior-independent odds ratio.
    """
    eps = 1e-9
    odds = np.clip(probability, eps, 1 - eps) / (1 - np.clip(probability, eps, 1 - eps))
    likelihood_ratio = odds * (1 - fitted) / max(fitted, eps)
    prior_odds = target / max(1 - target, eps)
    posterior = likelihood_ratio * prior_odds
    return posterior / (1 + posterior)


def main() -> None:
    """Rank candidates and report expected yield at each cut."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--evidence",
        required=True,
        type=Path,
        help="acr_evidence.csv from a pipeline run.",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=0.05,
        help="Expected fraction of real Acrs among candidates.",
    )
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.evidence)
    model = json.loads(args.calibration.read_text())
    calibrated = np.interp(
        frame["acr_locus_score"].fillna(0.0),
        model["raw_score"],
        model["calibrated_precision"],
    )
    frame["p_acr"] = rescale_prior(calibrated, model["calibration_prior"], args.prior)
    # A tier-1 HMM hit is categorically stronger than any tier-2 rank: it had
    # zero false positives across 191 negatives, so it sorts above everything.
    frame["tier_rank"] = (frame.get("acr_tier", "divergent") != "hmm_hit").astype(int)
    frame = frame.sort_values(["tier_rank", "p_acr"], ascending=[True, False])
    frame["rank"] = range(1, len(frame) + 1)
    frame["expected_true_acrs_by_here"] = frame["p_acr"].cumsum()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    hmm_hits = int((frame["tier_rank"] == 0).sum())
    print(
        f"\n  {len(frame)} candidates, prior {args.prior:.0%}, "
        f"{hmm_hits} with an HMM hit (accept directly)"
    )
    print(f"  {'take top':>9} {'expected true Acrs':>19} {'expected precision':>19}")
    for k in (5, 10, 20, 50, len(frame)):
        if k > len(frame):
            continue
        expected = frame["expected_true_acrs_by_here"].iloc[k - 1]
        print(f"  {k:>9} {expected:>19.1f} {expected / k:>18.0%}")
    print(
        "\n  Choose k by the yield you are willing to test, not by a score "
        "cutoff:\n  the calibration shows thresholds do not transfer across runs."
    )


if __name__ == "__main__":
    main()
