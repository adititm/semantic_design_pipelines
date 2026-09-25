# Anti-CRISPR screening pipeline

The published workflow called anti-CRISPRs with PaCRISPR. **PaCRISPR is no
longer available**, so that step cannot be reproduced as written. It is
replaced here by five callers that run offline. The fitted models they use
are in `data/models/acr/` and are loaded directly; how they were derived is not part of this repository.

    python -m semantic_design_pipelines.pipelines.acr_sample \
        --config semantic_design_pipelines/configs/acr_sample.yaml

## Ordering

**1. Generate** — Evo 2 (`evo2_7b`) from Acr-context prompts, 11 prompts x 5 samples.

**2. ORF calling + QC** — Prodigal, then the paper's predicates: length 50-200 aa,
`partial=00`, k-mer repetitiveness, >=12 unique AAs, segmasker <=0.1. Every
surviving ORF continues; the locus is scored as a whole.

**3. Profile-HMM annotation** 16 Acr/Aca families.
Automatically advance sequences that have hits to known Acr or Aca families to
the AF3 stage. These sequences are unlikely to be de novo Acrs, but may be
strong, diverged candidates.

**4. Sequence-only prescreen**
`run_acr_prescreen: true`. HMM, AcRanker and AcrNET all take sequence only
and are run first to rank the ORFs. `acr_prescreen_min_score` (default 0.15)
drops the least Acr-like tail and `acr_fold_fraction` caps how many survive;
**an ORF with an HMM hit always folds regardless of score.** The threshold was
chosen to retain nearly all known Acrs while skipping most negatives, and
checked not to favour sequences resembling known Acrs. Being sequence-only, it
is reliable enough to order on but not to accept on, simply gating folding.

**5. AlphaFold 3 — every surviving ORF, MSA on.**
`af3_max_proteins_per_proposal: 0`, `af3_use_msa: true`,
`af3_msa_device: cuda`, with `af3_plddt_threshold` / `af3_ptm_threshold`
gating. Foldseek ranks on these folds, so fold quality matters more than
triage speed.

**6. Five-caller evidence.** Batched across every protein in every proposal —
RaptorX 0.74 s/seq, PSI-BLAST ~2.9 s/seq on 32 cores, ESM-1b in one GPU pass,
plus HMM, AcRanker and Foldseek.

**7. Rank within the run and take the top-k.** Tier 1 accepts directly; tier 2
ranks.

## The five callers

Each is independent and optional -- an empty config path disables that
caller rather than failing -- and each contributes a column to
`acr_evidence.csv`.

| caller | what it contributes |
|---|---|
| profile HMM (16 Acr/Aca families) | family identity. High precision, low recall: a hit is near-certain, a miss means nothing |
| AcRanker | composition and reduced-alphabet k-mers, as a raw score and a self-shuffle z |
| AcrNET | deep model over secondary structure, PSSM and ESM-1b |
| Foldseek | TM-score against known Acr chains |
| AlphaFold 3 pLDDT | foldability, reused from the monomer screen |

## Two models, not one

Which model scores a protein depends on whether the profile HMM hit it.
Most known Acrs hit no Pfam family, so absence is not evidence against; a
separate model drops the HMM term rather than reading a missing feature as
a negative. `acr_tier` records which applied.

A second split applies to AcrNET: a missing PSI-BLAST PSSM inflates its
output toward 1.0 regardless of the protein, and the ORFs PSI-BLAST cannot
profile are exactly the divergent candidates this pipeline exists to find.
Scores are therefore read against the regime that produced them, recorded
in `acrnet_regime`, and the models consume a within-regime percentile
(`acrnet_pctile`) rather than the raw probability.

**Do not threshold `acrnet_score` directly.** It is not a calibrated
confidence; use `acrnet_tier`, which is regime-aware.

## Choosing how many to test

`acr_locus_score` is a logistic output, **not** P(Acr). It was fit at the
calibration set's class balance, which is nothing like the fraction of real
Acrs in a pool of generated sequence. Reading it as a probability, or
thresholding it, will overstate.

`utils/rank_candidates.py` does the conversion: isotonic calibration, then
a prior-independent likelihood-ratio rescale to whatever base rate you
actually expect.

```bash
python -m semantic_design_pipelines.utils.rank_candidates \
    --evidence outputs/.../acr_evidence.csv --prior 0.05 --out ranked.csv
```

It reports the **expected number of true Acrs in the top k**, which is the
sum of calibrated probabilities over those k. Choose k by the yield you are
willing to test, not by a score cutoff.

Ranking alone always returns k things whether or not any are real. The 
expected-yield number is the stopping rule. For example, on a run of four
generated ORFs that all scored poorly, it reports an expected yield of 0.0 
rather than handing back a top-4. An expected yield near zero means test
nothing, not test the best of a bad batch.

## Reading `acr_evidence.csv`

One row per ORF with every caller side by side.

* `acr_tier` — `hmm_hit` is qualitatively stronger evidence than a
  `divergent` rank.
* `hmm_best_profile` — which family, and whether Acr or Aca.
* **An Aca-like partner is a positive signal.** Acr and Aca cannot be
  separated: AcrIIA1, AcrIIA13, AcrIIA15 and AcrIF24 carry HTH domains and
  repress their own operons. An Acr-context prompt generates both, so an
  Acr + Aca pair strengthens the locus call.
* **Agreement across callers** beats any single high score, because they fail
  in orthogonal ways.
