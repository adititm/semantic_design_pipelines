# Anti-CRISPR screening pipeline

The published workflow called anti-CRISPRs with PaCRISPR. **PaCRISPR is no
longer available** -- a web server with no offline release, not part of the
published code -- so that step cannot be reproduced as written. It is
replaced here by five callers that run offline. The fitted models they use
are in `data/models/acr/` and are loaded directly; how they were derived is not part of this repository.

    python -m proto_pipelines.pipelines.acr_sample \
        --config proto_pipelines/configs/acr_sample.yaml

## Ordering

**1. Generate** — Evo 2 (`evo2_7b`) from Acr-context prompts, 11 prompts x 5 samples.

**2. ORF calling + QC** — Prodigal, then the paper's predicates: length 50-200 aa,
`partial=00`, k-mer repetitiveness, >=12 unique AAs, segmasker <=0.1. Every
surviving ORF continues; the locus is scored as a whole.

**3. Profile-HMM annotation — annotate, do not gate.** 16 Acr/Aca families.
0 false positives across 191 negatives, 0.7% on 145 defense genes. Only ~23%
of Acrs trigger it, so a miss means nothing and a hit is near-certain.

**4. Sequence-only prescreen — order the ORFs before paying for folds.**
`run_acr_prescreen: true`. HMM, AcRanker and AcrNET all take sequence only
(~5 s/protein) while AlphaFold 3 with an MSA costs minutes, so they run
first and rank the ORFs. `acr_prescreen_min_score` (default 0.15) drops the
least Acr-like tail and `acr_fold_fraction` caps how many survive; **an ORF
with an HMM hit always folds regardless of score.** The threshold was chosen
to retain nearly all known Acrs while skipping most negatives, and checked
not to favour sequences resembling known Acrs. Being sequence-only it is
reliable enough to order on but not to accept on — it gates folding, never
the final call.

**5. AlphaFold 3 — every surviving ORF, MSA on.**
`af3_max_proteins_per_proposal: 0`, `af3_use_msa: true`,
`af3_msa_device: cuda`, with `af3_plddt_threshold` / `af3_ptm_threshold`
gating. Foldseek ranks on these folds, so fold quality matters more than
triage speed.

**6. Five-caller evidence.** Batched across every protein in every proposal —
RaptorX 0.74 s/seq, PSI-BLAST ~2.9 s/seq on 32 cores, ESM-1b in one GPU pass,
plus HMM, AcRanker and Foldseek. About 5% of AlphaFold 3's cost, so it runs
in-loop.

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
| AlphaFold 3 pLDDT | foldability, reused from the monomer screen at no extra cost |

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

## Why rank rather than threshold

A threshold calibrated on natural Acrs selects for resemblance to known Acrs,
which is the opposite of the pipeline's purpose. Two further reasons, both
weaker than they first appear:

* **Composition controls.** AcrNET calls most composition-matched shuffles
  anti-CRISPR — but that is now explained: shuffles get no PSI-BLAST PSSM, and
  a zeroed PSSM inflates the score by itself (see *Reading an AcrNET score*).
  Scored within the `no_pssm` regime they are unremarkable. The comparison
  against real phage proteins is the operationally relevant one.
* **AlphaFold 3 non-determinism.** With a fixed seed and identical inputs the
  same sequence gave pLDDT 62.1-68.4 — GPU kernel non-determinism, not
  sampling, and not fixable by seeding. Modest (~6 points), measured at one
  diffusion sample; production uses five.

**Reuse the AlphaFold 3 output directory across runs.** `job_name_for` is a
content hash and the score cache sits beside the structures, so repeat folds
become free *and* deterministic. This is the single biggest lever on score
stability.

## Choosing how many to test

`acr_locus_score` is a logistic output, **not** P(Acr). It was fit at the
calibration set's class balance, which is nothing like the fraction of real
Acrs in a pool of generated sequence. Reading it as a probability, or
thresholding it, will overstate.

`utils/rank_candidates.py` does the conversion: isotonic calibration, then
a prior-independent likelihood-ratio rescale to whatever base rate you
actually expect.

```bash
python -m proto_pipelines.utils.rank_candidates \
    --evidence outputs/.../acr_evidence.csv --prior 0.05 --out ranked.csv
```

It reports the **expected number of true Acrs in the top k**, which is the
sum of calibrated probabilities over those k. Choose k by the yield you are
willing to test, not by a score cutoff.

This is also the answer to "does ranking let junk through?". Ranking alone
always returns k things whether or not any are real. The expected-yield
number is the stopping rule: on a run of four generated ORFs that all
scored in the veto tier, it reports an expected yield of 0.0 at a 5% prior
rather than handing back a top-4. **An expected yield near zero means test
nothing, not test the best of a bad batch.**

The prior is yours to supply and the output is only as good as it. If you
do not know the Acr rate in your pool, treat the ordering as usable and the
absolute numbers as not.

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

## Honest limits

* **Nothing here proves a sequence is an anti-CRISPR.** These are
  similarity and confidence measures. A high score means "worth testing",
  never "is an Acr".
* **Acr and Aca are not cleanly separable, and should not be.** Several
  known Acrs carry HTH domains and act as their own operon repressors, so
  the callers recover a documented overlap rather than failing.
  `acr_locus_score` targets "protein from an Acr locus"; `hmm_profiles`
  says which side it looks like.
* **Scores are not comparable across runs.** AlphaFold 3 sampling is not
  deterministic and Foldseek inherits that. Rank within a run. Pointing
  successive runs at the same AlphaFold 3 output directory makes repeat
  folds both free and deterministic.
* **The callers were calibrated on natural Acrs.** Their behaviour on
  generated sequence is uncharacterised, which is why the shipped configs
  record evidence rather than gating on it (`acr_min_score: 0.0`).
