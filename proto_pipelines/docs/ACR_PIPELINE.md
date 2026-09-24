# Anti-CRISPR screening pipeline

Replaces the paper's PaCRISPR step with five callers that run offline.
Calibrated on 316 labelled sequences plus 145 never-seen DefenseFinder
negatives.

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
first and rank the ORFs. `acr_prescreen_min_score` (0.15 — keeps 95% of known Acrs, folds 28%
of negatives; Spearman(identity to a known Acr, score) = +0.012, p=0.84, so
it does not reward resemblance) drops the least
Acr-like tail and `acr_fold_fraction` caps how many survive; **an ORF with an
HMM hit always folds regardless of score.** The prescreen model is
sequence-only (AUROC 0.911 vs 0.924 with Foldseek), which is reliable enough
to order on but not to accept on — it gates folding, never the final call.

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

| caller | contributes | vs phage | vs shuffles |
|---|---|---|---|
| AcrNET | deep model over SS + PSSM + ESM-1b | 0.918 | 0.806 |
| Foldseek | TM vs 220 known Acr chains | 0.691 | 0.748 |
| AcRanker | acidic / DNA-mimic composition | 0.801 | 0.567 |
| profile HMM | family identity, near-zero FP | — | — |
| AF3 pLDDT | foldability (tier 1 only) | 0.639 | 0.865 |

## Two tiers

| tier | trigger | model | AUROC | action |
|---|---|---|---|---|
| 1 | HMM hit (~15/64) | all five | 0.950 | accept |
| 2 | no HMM hit (49/64) | AcrNET + Foldseek + AcRanker | 0.924 | rank |

**49 of 64 known Acrs have no Pfam hit**, so tier 2 is the normal path. HMM
absence is not evidence against: the tier-1 model penalises the missing
feature and collapses to 0.64 if misapplied, which is why the tiers are
separate models rather than one.

## What ranking buys you

Cross-validated, divergent Acrs vs phage proteins (44% prior):

| take top | Acrs found | precision | enrichment |
|---|---|---|---|
| 5 | 5 | 100% | 2.3x |
| 10 | 9 | 90% | 2.1x |
| 20 | 17 | 85% | 1.9x |

## Reading an AcrNET score

AcrNET's probability is **saturated near 1.0 and regime-dependent**, so a bare
`0.9x` means nothing on its own. Two separate problems, both measured:

**1. A missing PSSM inflates the score.** PSI-BLAST profiles 63/64 Acrs and
63/63 phage proteins, but **0/64 shuffled and 0/64 random ORFs**. Zeroing the
PSSM on the *same* 63 phage proteins moved them from 19% to **78%** above 0.99
and dropped Acr-vs-phage AUROC from 0.918 to 0.791. So the raw score partly
reports *whether PSI-BLAST found hits* rather than Acr-likeness — and the ORFs
it cannot profile are exactly the divergent candidates this pipeline exists to
find.

**2. It is still informative without one** (AUROC 0.806 in that regime), so the
caller is recalibrated, not dropped. The signal sits in the extreme tail.

Scores are therefore read against the regime that produced them
(`data/models/acr/acrnet_operating_points.json`), and `acr.py` emits
`acrnet_regime`, `acrnet_tier` and `acrnet_pctile` per protein:

| regime | tier | bound | Acrs | negatives | LR |
|---|---|---|---|---|---|
| **has_pssm** | veto | `< 0.01` | 1.6% | 30.2% | **0.05** |
| AUROC 0.918 | uninformative | `0.01 - 0.9` | 4.8% | 44.4% | 0.11 |
| n=63 vs 63 | weak | `0.9 - 0.99` | 6.3% | 6.3% | 1.00 |
| | moderate | `0.99 - 0.999` | 14.3% | 12.7% | 1.12 |
| | strong | `>= 0.999` | 73.0% | 6.3% | **11.50** |
| **no_pssm** | veto | `< 0.99` | 3.2% | 17.8% | **0.18** |
| AUROC 0.806 | uninformative | `0.99 - 0.999` | 9.5% | 28.8% | 0.33 |
| n=63 vs 191 | weak | `0.999 - 0.9999` | 25.4% | 36.6% | 0.69 |
| | moderate | `0.9999 - 0.99999` | 44.4% | 15.7% | 2.83 |
| | strong | `>= 0.99999` | 17.5% | 1.1% | **16.68** |

`0.9995` is **strong (94th percentile) with a PSSM but only weak (55th)
without one**. Synthetic negatives are retained but scored only within
`no_pssm`, where their ~0.999 is unremarkable rather than Acr-like.

`no_pssm` positives are real Acrs with the PSSM block **zeroed by ablation** —
the identical input transformation the pipeline applies when PSI-BLAST returns
nothing. Only 1 of 64 Acrs natively lacks a PSSM, so there is no other way to
populate that regime with positives.

### The models consume the percentile, not the raw score

| acrnet feature in the divergent model | CV AUROC |
|---|---|
| raw `acrnet_score` | 0.7897 |
| **within-regime `acrnet_pctile`** | **0.9237** |
| raw + `has_pssm` flag | 0.9567 — *rejected* |
| `has_pssm` alone (control) | 0.8247 |
| acrnet removed (control) | 0.7755 |

The `has_pssm` flag variant scores highest and is **deliberately rejected**:
`has_pssm` alone reaches 0.8247, so that model would reward a protein for
being profilable — backwards for divergent Acrs. The percentile neutralises
the regime instead: negatives sit at median percentile **0.50 / 0.51** in the
two regimes, and within `has_pssm` raw (0.9181) and percentile (0.9174) are
equivalent, so it removes the artifact without inventing signal.

Shipped models: divergent **0.9237**, prescreen **0.9106**, combined
**0.9501** (`cv_auroc` in each JSON).

### Implementation notes (verified against upstream)

Checked against `banma12956/AcrNET` rather than assumed:

* `acrnet_model.py` is byte-identical to upstream `model.py`, including the
  final `F.log_softmax`; `torch.exp(output)[:, 1]` is the correct conversion.
* All four index maps match upstream exactly, including the
  **non-alphabetical** `AVGILFPYMTSHNQWRKDEC`. Alphabetical ordering would
  silently scramble the sequence channel and raise no error.
* Upstream's `test.py` converts the output with `argmax(dim=-1)` and never
  uses the probability. The paper reports accuracy/precision/recall/F1/MCC
  and ROC *curves*, but no numeric AUROC, no calibration analysis and no
  stated threshold — the probability is an unvalidated byproduct.

**Padding was a real confound and is fixed, but it was not the cause of the
above.** AcrNET pads with index 0, which one-hots to a *valid* residue
(`A`/`C`/`L`/`E`) rather than zeros, so padded positions enter the convolution
and its max-pool. Chunked scoring therefore leaked batch membership. Measured
on identical features, however, per-protein vs chunk-of-32 differs by a median
of **0.0000** (max 0.1513, 0 rows above 0.5) and AUROC 0.7698 vs 0.7686 — the
PSSM regime, not padding, drives the spread. `acrnet.score` now runs **one
protein at a time**, which makes the score a function of the protein alone;
this deliberately diverges from upstream, which pads its whole dataset in one
call and inherits the same dependence. Cost ~1 ms per protein. Guarded by the
`AcrNET score is batch-invariant` parity check.

**A superseded `acrnet_scores.csv` (AUROC 0.8831) was discarded as
irreproducible**: neither per-protein nor chunked scoring regenerates it from
the cached features (max delta 0.9988), and it was produced by an ad-hoc
script with no committed entrypoint. a committed regeneration entrypoint now exists in the research tree
so the table can always be regenerated, and `--dump-features` makes a scoring
change comparable on identical inputs.

## Why rank rather than threshold

A threshold calibrated on natural Acrs selects for resemblance to known Acrs,
which is the opposite of the pipeline's purpose. Two further reasons, both
weaker than they first appear:

* **Composition controls.** AcrNET calls most composition-matched shuffles
  anti-CRISPR — but that is now explained: shuffles get no PSI-BLAST PSSM, and
  a zeroed PSSM inflates the score by itself (see *Reading an AcrNET score*).
  Scored within the `no_pssm` regime they are unremarkable. The phage
  comparison (0.918) remains the operationally relevant one.
* **AlphaFold 3 non-determinism.** With a fixed seed and identical inputs the
  same sequence gave pLDDT 62.1-68.4 — GPU kernel non-determinism, not
  sampling, and not fixable by seeding. Modest (~6 points), measured at one
  diffusion sample; production uses five.

**Reuse the AlphaFold 3 output directory across runs.** `job_name_for` is a
content hash and the score cache sits beside the structures, so repeat folds
become free *and* deterministic. This is the single biggest lever on score
stability.

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

## Divergence probe: how far does this generalise?

All three Acr databases were offline (AcrDB maintenance, anti-CRISPRdb 502,
AcrHub DNS), so the probe was built locally from the 65,255 predicted Acrs in
`AcrDatabase.faa` (AcrHub + AcrCatalog predictions), binned by MMseqs
identity to the canonical 64 and sampled 75 per bin.

**AcRanker degrades gracefully.** AUROC vs phage across the gradient:

| set | n | AUROC vs phage |
|---|---|---|
| canonical 64 | 64 | 0.801 |
| >=50% identity to known | 75 | 0.780 |
| 30-50% | 75 | 0.751 |
| <30% | 75 | 0.786 |
| no alignment at all | 75 | 0.741 |

0.801 -> 0.741 across the full range, consistent with it reading a
mechanistic property (acidic / DNA-mimic composition) rather than memorising.

**AcrNET splits sharply by label provenance, not by divergence:**

Regenerated with the corrected per-protein scoring and regime-aware
calibration (regenerated in the research tree; features cached in
`results/acr/acrnet_features{,_heldout,_diverse}.pkl`). The reference is the
same 63 phage proteins, all with real PSSMs.

| set | n | PSSM | median | AUROC vs phage | 95% CI |
|---|---|---|---|---|---|
| canonical 64 (validated) | 64 | 98% | 1.000 | **0.9191** | 0.865 – 0.963 |
| 19 held-out **validated** Acrs | 19 | 100% | 1.000 | **0.9340** | 0.852 – 0.989 |
| 300 **predicted** Acrs | 300 | 97% | 0.975 | **0.7499** | 0.671 – 0.823 |

All three panels are 97–100% PSSM-covered against a 100%-covered reference,
so these sit almost entirely inside the `has_pssm` regime. Scoring them on
the within-regime percentile instead moves nothing (0.9183 / 0.9319 /
0.7376) — the regime correction changes results only where the confound
actually operates, which is a useful check that it is not doing anything
spurious here.

The held-out validated set — which was never used to derive any model here —
scores identically to the canonical 64. So the drop on the predicted panel is
most plausibly the *predictions'* false-positive rate rather than AcrNET
failing on divergent sequence, and it does not by itself indicate
memorisation.

**What stays unresolved.** Both validated sets are *published* Acrs, and
AcrNET (2023) was trained on published Acrs, so both could sit inside its
training data. A leak-free test needs Acrs first described after 2023;
UniProt `date_created` cannot supply that (H9C181 is AcrF8, a long-known Acr,
with a 2025 entry date). Treat **0.919-0.934 as an upper bound** for
genuinely novel sequence, and note that on this evidence AcRanker is the more
*trustworthy* caller on divergent input even though AcrNET has the stronger
headline number.


## Honest limits

* **Plan against 0.919-0.934 (vs phage), not 0.977.** DefenseFinder genes
  turned out *easier* than phage — Acrs are phage proteins and share that
  compositional background. Phage is the hard negative.
* **Likely optimistic.** AcrNET was trained on published Acrs and the
  positives are the canonical published set; its training data is not in the
  repo, so overlap could not be checked. The 0.998 median on positives is a
  memorisation signature.
* **Tier 2 rests on 49 positives and 63 phage negatives**, not bootstrapped
  at that size.
* **All level-1 evidence** — similarity and confidence measures. A high score
  means *worth testing*, never *is an anti-CRISPR*.

The decisive experiment has not been run: score a batch of Evo candidates,
test the top-k, and see whether the ranking holds on sequences no caller was
trained on.
