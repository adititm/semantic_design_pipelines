"""Score the Acr calibration set with the Acr/Aca profile HMMs.

Reuses :func:`proto_pipelines.hmm._scan`, so the numbers here are produced by
exactly the code path the pipeline filter would use.

Unlike a ranker, an HMM scan is a *detector*: most sequences have no hit at
all. The reported statistic is therefore detection rate per class rather than
only AUROC, and the score used for AUROC is ``-log10(best E-value)`` with
no-hit sequences pinned to zero -- so a class where nothing is detected sits
at the floor instead of contributing noise.

The expected weakness is coverage, not specificity: Pfam carries only a
handful of Acr families, so this can only recognise Acrs related to those.
A low detection rate on the positives is the honest result, not a bug.

Usage:
    python -m proto_pipelines.calibration.score_acr_hmm \
        --sequences acr_calibration.csv --hmm acr_families.hmm \
        --out acr_hmm_scores.csv --evalue 1.0
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from proto_pipelines.hmm import _scan

logger = logging.getLogger(__name__)


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUROC via Mann-Whitney, ties counted as half.

    Ties matter here: most sequences score exactly zero (no hit), so a
    tie-blind implementation would badly misreport.

    Args:
        positive: Scores for the positive class.
        negative: Scores for the negative class.

    Returns:
        Area under the ROC curve, ``nan`` if either class is empty.
    """
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).sum()
    tied = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * tied) / (positive.size * negative.size))


def main() -> None:
    """Scan every sequence and report detection rate and AUROC per class."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--hmm", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--evalue", type=float, default=1.0,
                        help="Sequence-level E-value cap; lenient by default.")
    args = parser.parse_args()

    frame = pd.read_csv(args.sequences)
    scores, profiles, evalues = [], [], []
    for position, sequence in enumerate(frame["sequence"], start=1):
        hits = _scan(sequence, str(args.hmm), args.evalue)
        if hits:
            best = hits[0]
            evalues.append(best["evalue"])
            profiles.append(best["profile"])
            scores.append(-math.log10(max(best["evalue"], 1e-300)))
        else:
            evalues.append(None)
            profiles.append(None)
            scores.append(0.0)
        if position % 50 == 0:
            logger.info("scanned %d/%d", position, len(frame))

    frame["hmm_best_profile"] = profiles
    frame["hmm_best_evalue"] = evalues
    frame["hmm_score"] = scores
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    positives = frame.loc[frame["label"] == 1, "hmm_score"].to_numpy()
    detected = frame.assign(hit=frame["hmm_best_profile"].notna())
    print(f"\n  E-value cap: {args.evalue}")
    print(f"  {'class':<14} {'n':>4} {'detected':>9} {'rate':>7} {'AUROC vs acr':>13}")
    for name, group in detected.groupby("seq_class"):
        rate = group["hit"].mean()
        au = ("-" if name == "acr"
              else f"{auroc(positives, group['hmm_score'].to_numpy()):.3f}")
        print(f"  {name:<14} {len(group):>4} {int(group['hit'].sum()):>9} "
              f"{rate:>6.1%} {au:>13}")
    print("\n  profiles hit by the positives:")
    counts = frame.loc[frame["label"] == 1, "hmm_best_profile"].value_counts()
    for profile, count in counts.items():
        print(f"    {profile:<24} {count}")


if __name__ == "__main__":
    main()
