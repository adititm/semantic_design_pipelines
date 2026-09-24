# Optimisation round notes — September 2026

Findings from the round that replaced PaCRISPR, built the five-caller Acr
stack and verified the tree against official `evo-design` `main`. Kept for
provenance. **Nothing here is needed to use the pipelines**; current
behaviour is documented in `../README.md`, `../docs/ACR_PIPELINE.md` and
`../docs/CALIBRATION.md`.

## Discarded: the original AcrNET calibration

`acrnet_scores.csv` (standalone AUROC 0.8831) was produced by an ad-hoc
script and proved **irreproducible** — neither per-protein nor chunked
scoring regenerates it from the cached features (max delta 0.9988). Every
figure derived from it was withdrawn, including a four-tier table whose top
tier claimed LR 10.82. `calibration/score_acrnet.py` exists so this cannot
recur: the table is now regenerable, and `--dump-features` makes a scoring
change comparable on identical inputs.

## The PSSM confound

PSI-BLAST profiles 63/64 Acrs and 63/63 phage proteins, but 0/64 shuffled
and 0/64 random ORFs. A zeroed PSSM inflates AcrNET's output on its own:
zeroing it on the *same* 63 phage proteins moved them from 19% to 78% above
0.99, and dropped Acr-vs-phage AUROC 0.918 → 0.791. A `has_pssm` flag alone
separates the calibration classes at AUROC 0.825.

Consequences, all now in the shipped behaviour:

* Tiers are per-regime. Positives for `no_pssm` come from **ablation** —
  real Acrs with the PSSM zeroed, the same transformation the pipeline
  applies when PSI-BLAST returns nothing. Only 1 of 64 Acrs natively lacks
  one, so there was no other way to populate that regime.
* Models consume `acrnet_pctile` (percentile within regime), not the raw
  probability. A `raw + has_pssm` model scored higher (0.9567 vs 0.9237) and
  was **rejected**: it rewards a protein for being profilable, backwards for
  divergent Acrs.
* AcrNET is not useless without a PSSM (AUROC 0.806 in that regime), so the
  caller was recalibrated rather than dropped.

## Padding was real but not the cause

AcrNET pads with index 0, which one-hots to a valid residue (`A`/`C`/`L`/`E`)
rather than zeros, so padded positions enter the convolution and its
max-pool. Chunked scoring therefore leaked batch membership. Measured on
identical features, though, per-protein vs chunk-of-32 differs by a median
of 0.0000 (max 0.1513) — the PSSM regime, not padding, drove the spread.
Scoring is now one protein at a time; a per-call global pad width does *not*
fix it, since that width still depends on the set being scored.

## Verified against upstream AcrNET

`acrnet_model.py` is byte-identical to `banma12956/AcrNET`'s `model.py`, and
all four index maps match — including the non-alphabetical
`AVGILFPYMTSHNQWRKDEC`, where alphabetical order would silently scramble the
sequence channel. Upstream's `test.py` uses `argmax(dim=-1)` and never the
probability; the paper reports accuracy/precision/recall/F1/MCC and ROC
curves but no numeric AUROC, no calibration analysis and no threshold. The
probability is an unvalidated byproduct, which is why tiers exist.

## Clean-environment findings

Building a fresh env from official `main` (proto-language `1a56693e`,
proto-tools `55339880`) exposed four defects invisible on a workstation
where everything happened to be installed:

1. `torch`, `fair-esm`, `pssmpro`, `xgboost` were undeclared — now the
   `acr` extra.
2. `pssmpro>=1.0` was an invented constraint; only 0.0.1/0.0.2 exist.
3. Unpinned `torch>=2.0` resolved to a CUDA 13 build and died at GPU init on
   a 12.6 driver.
4. `FoldseekHit` on official main has no TM field — see below.

Development had been running against a personal branch 1243 commits behind
and 1483 ahead of `evo/main`, which is also why an internally inconsistent
`codonfm` import broke the parity suite there but not on matched `main`s.

## Foldseek TM-score

proto-tools parses a plain 12-column M8 row, so `FoldseekHit` carries no TM
at all. `acr.py` therefore calls the provisioned binary directly with
`--format-output …,qtmscore,alntmscore`. Two things were checked:

* **E-value is not a substitute.** As a model feature it scored CV AUROC
  0.9115, against 0.9106 for dropping the structural term entirely and
  0.9237 with a real TM. Swapping it in would have silently deleted the
  caller.
* **Alignment type matters.** `--alignment-type 1` (TM-align) returns
  systematically higher TM — median +0.15 over 25 re-searched calibration
  chains — which would put the feature on a different scale than the fits.
  Type 2 (3Di+AA), foldseek's default, reproduces the calibration: r=0.983,
  median delta 0.0000, 16/25 exact.

## Prescreen threshold

Moved 0.20 → 0.15 on the refit model: 95% of known Acrs retained folding 28%
of negatives, against 92%/23% at 0.20 and 94%/35% for the superseded model.
Re-verified for over-enrichment on the 300-sequence diverse panel — the only
set here with a real identity gradient, since all 64 calibration Acrs are
AcrDB entries hitting it at 50% identity. Spearman(identity, prescreen
score) = +0.012 (p=0.84), and -0.099 (p=0.14) excluding the no-alignment
bin; sequences <30% identical to a known Acr pass slightly *more* often than
those ≥50%.
