# Anti-CRISPR calibration assets

`acr_families_default.hmm` (16 profiles) is the shipped Acr/Aca profile set:
12 curated Pfam Acr families plus `DUF2829`, `DUF1374`, `PcfK` and
`STIV_B116-like`, derived by scanning 179 known Acrs against full Pfam at
E<1e-5 and refusing HTH and Cas families.

`acr_families.hmm` (12 profiles) is the curated-only subset, kept for
provenance.

Two other variants were built, measured and rejected, and are not shipped:

* **29 families**, everything hit at E<0.01 — raised sensitivity 17%→28% but
  produced **82% false positives on Aca proteins** (AUROC 0.217, inverted),
  because it admitted HTH families and Aca proteins are HTH regulators.
* **6 families**, evidence-only at E<1e-5 — dropped curated Acr families that
  the scanned set happened not to hit strongly, falling to 9% sensitivity.

`ref_db/` is a prebuilt Foldseek database over 189 Acr chains extracted from
112 PDB entries (Cas9/Cascade chains excluded by sequence matching, not by
entry title). The chains themselves are gitignored; `ref_acr_manifest.json`
records the 115 source entities so they can be regenerated.

## Two Foldseek references, and which to use

`ref_db/acr_ref` — 189 experimentally determined Acr chains from the PDB.
**This is the calibration reference**, and every number reported for the
structural caller was measured against it. It is kept precisely because it
contains no predicted structures.

`ref_db_ext/acr_ref_ext` — the same 189 plus 31 AlphaFold 3 models of known
Acrs that reached pLDDT >= 50. **This is the production reference**, and what
the shipped configs point at. On the 19 held-out Acrs it raises Foldseek
AUROC vs phage from 0.723 to 0.743, improving 8/19 chains while leaving the
phage median essentially unchanged (0.235 -> 0.240), so it broadens coverage
rather than inflating everything.

Do not re-run the calibration against `ref_db_ext`: those 31 models are folds
of the calibration positives, so a query would match its own structure and
the AUROC would be meaningless. Use `ref_db` for any re-calibration.
