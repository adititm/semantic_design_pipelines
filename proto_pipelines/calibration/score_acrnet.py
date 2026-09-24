"""Score sequences with AcrNET's published model and report the distribution.

Written because the shipped ``acrnet_scores.csv`` was originally produced by
an ad-hoc script, leaving no way to regenerate it after a change to
:func:`proto_pipelines.acrnet.score`.

Two properties of AcrNET shape what this reports:

* **The probability was never validated upstream.** ``test.py`` in
  ``banma12956/AcrNET`` converts the model output with ``argmax(dim=-1)``, a
  hard 0/1 call, and the paper reports accuracy/precision/recall/F1/MCC plus
  ROC *curves* -- no numeric AUROC, no calibration analysis, no stated
  threshold. The probability is a byproduct, so this script reports the
  empirical operating points rather than trusting the scale.
* **Scores are batch-invariant here but not upstream.** AcrNET pads with
  index 0, which one-hots to a *valid* residue rather than zeros, so padded
  positions enter the convolution and its max-pool. ``score`` therefore runs
  one protein at a time; see the comment there.

Usage:
    python -m proto_pipelines.calibration.score_acrnet \
        --sequences proto_pipelines/calibration/data/acr/acr_calibration.csv \
        --predict-property .scratch_acr/Predict_Property \
        --psiblast .scratch_acr/blast/ncbi-blast-2.17.0+/bin/psiblast \
        --blast-db .scratch_acr/blast/db/uniref30 \
        --checkpoint proto_pipelines/calibration/data/acr/acrnet/model.ckpt \
        --out proto_pipelines/calibration/results/acr/acrnet_scores.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEGATIVE_CLASSES = ("phage", "shuffled", "random_orf")


def main() -> int:
    """Extract AcrNET features, score, and write the per-sequence table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--predict-property", required=True)
    parser.add_argument("--psiblast", default="")
    parser.add_argument("--blast-db", default="")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--dump-features", type=Path, default=None,
        help="Write extracted features to this .npz so a scoring change can "
             "be compared on identical inputs instead of re-extracting.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from proto_pipelines.acrnet import extract_features, score

    frame = pd.read_csv(args.sequences)
    for column in ("seq_id", "sequence", "seq_class"):
        if column not in frame.columns:
            raise KeyError(f"{args.sequences} lacks required column {column!r}")

    proteins = [
        {"protein_id": str(row.seq_id), "sequence": str(row.sequence)}
        for row in frame.itertuples()
    ]
    logger.info("extracting AcrNET features for %d sequences", len(proteins))
    features = extract_features(
        proteins, args.predict_property, esm_device=args.device,
        psiblast_bin=args.psiblast, blast_db=args.blast_db, workers=args.workers,
    )
    if not features:
        raise RuntimeError(
            f"extract_features returned nothing for {len(proteins)} input "
            "sequences -- RaptorX/PSI-BLAST/ESM almost certainly failed."
        )
    missing = [p["protein_id"] for p in proteins if p["protein_id"] not in features]
    if missing:
        logger.warning("no features for %d sequence(s): %s",
                       len(missing), ", ".join(missing[:10]))

    if args.dump_features:
        import pickle

        args.dump_features.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_features.open("wb") as handle:
            pickle.dump(features, handle)
        logger.info("dumped features for %d proteins to %s",
                    len(features), args.dump_features)

    scores = score(features, args.checkpoint)
    frame["acrnet_score"] = frame.seq_id.astype(str).map(scores)
    if frame.acrnet_score.isna().all():
        raise RuntimeError("every sequence scored NaN -- seq_id join failed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    logger.info("wrote %s (%d rows)", args.out, len(frame))

    scored = frame.dropna(subset=["acrnet_score"])
    positives = scored[scored.seq_class == "acr"].acrnet_score.values
    negatives = scored[scored.seq_class.isin(NEGATIVE_CLASSES)].acrnet_score.values
    if len(positives) and len(negatives):
        from scipy.stats import mannwhitneyu

        auroc = mannwhitneyu(positives, negatives).statistic / (
            len(positives) * len(negatives)
        )
        print(f"\n  n_acr={len(positives)}  n_negative={len(negatives)}")
        print(f"  standalone AUROC = {auroc:.4f}")
        print(f"\n  {'tier':<16}{'bound':<16}{'Acrs':>8}{'negatives':>11}{'LR':>8}")
        for name, low, high in [("veto", 0.0, 0.01), ("uninformative", 0.01, 0.99),
                                ("moderate", 0.99, 0.999), ("strong", 0.999, 1.01)]:
            in_acr = float(((positives >= low) & (positives < high)).mean())
            in_neg = float(((negatives >= low) & (negatives < high)).mean())
            ratio = f"{in_acr / in_neg:.2f}" if in_neg > 0 else "inf"
            print(f"  {name:<16}{f'{low}-{high}':<16}{in_acr:>7.1%}{in_neg:>10.1%}{ratio:>8}")
        # Upstream's actual output is argmax, i.e. this threshold.
        print(f"\n  at upstream's argmax (>=0.5) operating point:")
        for name in ("acr", *NEGATIVE_CLASSES):
            subset = scored[scored.seq_class == name].acrnet_score
            if len(subset):
                print(f"    {name:<12} called positive: {(subset >= 0.5).mean():>5.0%} "
                      f"(n={len(subset)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
