"""Score sequences with AcRanker's published model and report per-class AUROC.

AcRanker ships an XGBoost *ranker* (``objective="rank:pairwise"``) trained to
order a whole proteome by Acr-likeness. Two consequences shape this script:

* A rank within a proteome is meaningless for our unit of work -- a handful
  of ORFs from one generation -- so the raw margin score is used instead and
  the ranking is done once, globally, over everything scored.
* Ranker margins are not calibrated probabilities. Only the *ordering* is
  interpretable, which is why AUROC is the reported statistic and no
  score threshold is suggested here.

The features are recomputed rather than imported: AcRanker's ``server2.py``
is Python 2-era and imports ``sklearn.externals.joblib``, removed in
scikit-learn 0.23. The three feature blocks below reproduce
``prot_feats_seq`` exactly -- 20 L2-normalised amino-acid fractions, then
2-mer and 3-mer counts over a 7-group reduced alphabet, each divided by
``len(seq)-1`` and L2-normalised -- for 412 features total.

Usage:
    python -m proto_pipelines.calibration.score_acranker \
        --sequences acr_calibration.csv \
        --model calibration/data/acr/acranker_booster.json \
        --out acranker_scores.csv

The shipped ``xgb_rank.pickle`` predates XGBoost 1.0 and no modern release
will unpickle it. ``acranker_booster.json`` is that same booster, extracted
from the pickle and re-exported by XGBoost 1.7.6 -- 412 features, 120 rounds,
unchanged weights.
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
#: AcRanker's 7-group reduced alphabet, verbatim from server2.twomerFromSeq.
GROUPS = {
    "A": "1", "V": "1", "G": "1", "I": "2", "L": "2", "F": "2", "P": "2",
    "Y": "3", "M": "3", "T": "3", "S": "3", "H": "4", "N": "4", "Q": "4",
    "W": "4", "R": "5", "K": "5", "D": "6", "E": "6", "C": "7",
}


def _l2(vector: np.ndarray) -> np.ndarray:
    """L2-normalise, leaving an all-zero vector untouched rather than NaN."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _kmer_block(sequence: str, k: int) -> np.ndarray:
    """Reduced-alphabet k-mer frequencies, in AcRanker's index order.

    Residues outside the 20 standard amino acids are skipped, which is the
    behaviour AcRanker falls back to; its own except-branch substitutes the
    most frequent group, but that path only triggers on a KeyError and is not
    reachable for clean sequences.

    Args:
        sequence: Protein sequence.
        k: k-mer length (2 or 3).

    Returns:
        A ``7**k`` vector of counts divided by ``len(sequence) - 1``.
    """
    index = {"".join(p): i for i, p in enumerate(product("1234567", repeat=k))}
    counts = np.zeros(7**k)
    for start in range(len(sequence) - k + 1):
        kmer = sequence[start : start + k]
        if any(residue not in GROUPS for residue in kmer):
            continue
        counts[index["".join(GROUPS[residue] for residue in kmer)]] += 1
    denominator = max(len(sequence) - 1, 1)
    return counts / denominator


def features(sequence: str) -> np.ndarray:
    """Build AcRanker's 412-dimensional feature vector for one sequence.

    Args:
        sequence: Protein sequence.

    Returns:
        Concatenated composition, 2-mer and 3-mer blocks, each L2-normalised.
    """
    length = max(len(sequence), 1)
    composition = np.array([sequence.count(a) / length for a in AMINO_ACIDS])
    return np.concatenate(
        [_l2(composition), _l2(_kmer_block(sequence, 2)), _l2(_kmer_block(sequence, 3))]
    )


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUROC via the Mann-Whitney statistic, with ties counted as half.

    Args:
        positive: Scores for the positive class.
        negative: Scores for the negative class.

    Returns:
        Area under the ROC curve, or ``float("nan")`` if either class is empty.
    """
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).sum()
    tied = (positive[:, None] == negative[None, :]).sum()
    return float((greater + 0.5 * tied) / (positive.size * negative.size))


def main() -> None:
    """Score the calibration set and print AUROC against each negative class."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    import xgboost

    frame = pd.read_csv(args.sequences)
    booster = xgboost.Booster()
    booster.load_model(str(args.model))

    matrix = np.vstack([features(s) for s in frame["sequence"]])
    logger.info("Feature matrix: %s", matrix.shape)
    if booster.num_features() != matrix.shape[1]:
        raise ValueError(
            f"model expects {booster.num_features()} features, built "
            f"{matrix.shape[1]}; the feature reimplementation has drifted"
        )
    frame["acranker_score"] = booster.predict(xgboost.DMatrix(matrix))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    positives = frame.loc[frame["label"] == 1, "acranker_score"].to_numpy()
    print(f"\n  positives: n={positives.size} median={np.median(positives):.4f}")
    print(f"  {'negative class':<14} {'n':>4} {'median':>9} {'AUROC':>7}")
    for name, group in frame[frame["label"] == 0].groupby("seq_class"):
        scores = group["acranker_score"].to_numpy()
        print(f"  {name:<14} {scores.size:>4} {np.median(scores):>9.4f} "
              f"{auroc(positives, scores):>7.3f}")
    everything = frame.loc[frame["label"] == 0, "acranker_score"].to_numpy()
    print(f"  {'ALL negatives':<14} {everything.size:>4} "
          f"{np.median(everything):>9.4f} {auroc(positives, everything):>7.3f}")


if __name__ == "__main__":
    main()
