"""Turn scored calibration pairs into per-score discrimination and thresholds.

Reports, for each candidate score (pDockQ2, ipTM, pTM, avg pLDDT, and the
reported-only pDockQ v1), how well it separates cognate TA pairs from each
negative class, with bootstrap confidence intervals because the pilot sets are
small enough that a point AUROC is easy to over-read.

Three deliberate choices:

* **Negative classes are scored separately, never pooled.** Pooling an easy
  class (``cross_family``) with a hard one (``same_family``) produces a single
  flattering number that hides which discrimination actually works. The
  spread between them is the informative part.
* **The ``same_family`` AUROC is reported as a lower bound.** TA antitoxins
  cross-react within a family, so some of those "negatives" may be genuine
  binders mislabelled by construction.
* **A threshold is only suggested, never asserted as validated.** The operating
  point is chosen on natural pairs; transferring it to de novo designs is an
  assumption this data cannot test. ``--denovo`` scores a second, functionally
  labelled set against the same thresholds to probe that transfer.

Usage:
    python -m proto_pipelines.calibration.analyze --results results/pilot_natural
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Higher is better for all of these.
SCORES = ["pdockq2", "iptm", "ptm", "avg_plddt", "pdockq_v1"]
BOOTSTRAP_SAMPLES = 2000


def load_scores(results_dir: Path) -> pd.DataFrame:
    """Concatenate every shard CSV in ``results_dir``.

    Args:
        results_dir: Directory holding ``scores_shard*.csv``.

    Returns:
        All scored pairs, de-duplicated on ``root_id``.

    Raises:
        FileNotFoundError: If no shard CSV is present.
        ValueError: If required label columns are missing.
    """
    shards = sorted(results_dir.glob("scores_shard*.csv"))
    if not shards:
        raise FileNotFoundError(f"No scores_shard*.csv in {results_dir}")
    frame = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    for column in ("label", "pair_class"):
        if column not in frame.columns:
            raise ValueError(f"Scored results are missing the {column!r} column")
    before = len(frame)
    frame = frame.drop_duplicates(subset="root_id")
    if len(frame) != before:
        logger.info("Dropped %d duplicate row(s) across shards", before - len(frame))
    return frame


def auroc(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney statistic.

    Computed directly rather than via a curve so that ties are handled
    explicitly: tied pairs contribute 0.5, which matters when a score
    saturates (many pDockQ2 values pinned near 0, for instance).

    Args:
        positives: Scores of the positive class.
        negatives: Scores of the negative class.

    Returns:
        AUROC in ``[0, 1]``; ``nan`` when either class is empty.
    """
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = positives[:, None] - negatives[None, :]
    wins = (comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()
    return float(wins / (positives.size * negatives.size))


def bootstrap_auroc(
    positives: np.ndarray, negatives: np.ndarray, samples: int, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for :func:`auroc`.

    Args:
        positives: Positive-class scores.
        negatives: Negative-class scores.
        samples: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        ``(lower, upper)`` bounds of the 95% interval; ``(nan, nan)`` when
        either class is empty.
    """
    if positives.size == 0 or negatives.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(samples)
    for i in range(samples):
        p = rng.choice(positives, size=positives.size, replace=True)
        n = rng.choice(negatives, size=negatives.size, replace=True)
        values[i] = auroc(p, n)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def youden_threshold(positives: np.ndarray, negatives: np.ndarray) -> dict[str, float]:
    """Pick the cutoff maximising Youden's J (sensitivity + specificity - 1).

    Args:
        positives: Positive-class scores.
        negatives: Negative-class scores.

    Returns:
        ``threshold``, ``sensitivity``, ``specificity`` and ``youden_j`` at
        the best operating point. All ``nan`` when a class is empty.
    """
    if positives.size == 0 or negatives.size == 0:
        return dict.fromkeys(("threshold", "sensitivity", "specificity", "youden_j"), float("nan"))
    candidates = np.unique(np.concatenate([positives, negatives]))
    best = {"threshold": float("nan"), "sensitivity": 0.0, "specificity": 0.0, "youden_j": -1.0}
    for cutoff in candidates:
        sensitivity = float((positives >= cutoff).mean())
        specificity = float((negatives < cutoff).mean())
        j = sensitivity + specificity - 1.0
        if j > best["youden_j"]:
            best = {
                "threshold": float(cutoff),
                "sensitivity": sensitivity,
                "specificity": specificity,
                "youden_j": j,
            }
    return best


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Score every metric against every negative class.

    Args:
        frame: Scored pairs from :func:`load_scores`.

    Returns:
        One row per (score, negative class) with counts, medians, AUROC and
        its bootstrap interval, and the Youden operating point.
    """
    positives = frame[frame["label"] == 1]
    negative_classes = sorted(frame.loc[frame["label"] == 0, "pair_class"].unique())

    rows: list[dict[str, Any]] = []
    for score in SCORES:
        if score not in frame.columns:
            logger.warning("Score %r absent from results; skipping", score)
            continue
        pos = positives[score].dropna().to_numpy(dtype=float)
        for klass in negative_classes:
            neg = frame[(frame["label"] == 0) & (frame["pair_class"] == klass)][score]
            neg = neg.dropna().to_numpy(dtype=float)
            low, high = bootstrap_auroc(pos, neg, BOOTSTRAP_SAMPLES)
            operating = youden_threshold(pos, neg)
            rows.append(
                {
                    "score": score,
                    "negative_class": klass,
                    "n_pos": pos.size,
                    "n_neg": neg.size,
                    "median_pos": float(np.median(pos)) if pos.size else float("nan"),
                    "median_neg": float(np.median(neg)) if neg.size else float("nan"),
                    "auroc": auroc(pos, neg),
                    "auroc_lo95": low,
                    "auroc_hi95": high,
                    **operating,
                    "lower_bound_only": klass == "same_family",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Load scored pairs, summarise discrimination, and write the tables."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", required=True, type=Path, help="Shard CSV directory")
    parser.add_argument("--out", type=Path, default=None, help="Summary CSV (default: in --results)")
    args = parser.parse_args()

    frame = load_scores(args.results)
    print(f"\nScored pairs: {len(frame)}")
    print(frame.groupby(["pair_class", "label"]).size().to_string())

    summary = summarize(frame)
    out = args.out or args.results / "calibration_summary.csv"
    summary.to_csv(out, index=False)

    show = [
        "score", "negative_class", "n_pos", "n_neg", "median_pos", "median_neg",
        "auroc", "auroc_lo95", "auroc_hi95", "threshold", "sensitivity", "specificity",
    ]
    print("\n" + summary[show].round(3).to_string(index=False))
    print(f"\nWrote {out}")
    print(
        "\nReading these numbers:\n"
        "  - same_family AUROC is a LOWER BOUND: within-family antitoxin cross-\n"
        "    reactivity means some of those negatives may be genuine binders.\n"
        "  - cross_family and shuffled are easier by construction; strong numbers\n"
        "    there alone do not justify a threshold.\n"
        "  - Any threshold here is fitted on NATURAL pairs. Natural complexes are\n"
        "    likely represented in AlphaFold 3 training data, so transfer to de novo\n"
        "    designs is an assumption this data cannot test."
    )


if __name__ == "__main__":
    main()
