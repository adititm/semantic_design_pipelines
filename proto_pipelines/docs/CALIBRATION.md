# Calibration

How the shipped thresholds and models were derived, and what the measurements
do and do not support. **None of this is needed to run a pipeline** -- the
fitted models in `calibration/data/` are loaded directly. Read this when you
want to re-derive them, change a gate, or judge how much weight a number
carries.

The scripts that produced these live in `calibration/`; the intermediates
they write to `calibration/results/` are gitignored (~1.6 GB) and
reproducible. One-off SLURM jobs for those steps are in `archive/slurm/`.

For the anti-CRISPR pipeline specifically, see [ACR_PIPELINE.md](ACR_PIPELINE.md).

## Threshold calibration (`calibration/`)

The cutoffs in the configs are placeholders. `calibration/` is the machinery
for replacing them with measured ones, in two arms that answer different
questions.

**Arm 1 - natural cognate pairs (`build_tadb_set.py`).** Cognate
toxin/antitoxin partners from [TADB 3.0](https://bioinfo-mml.sjtu.edu.cn/TADB3/)'s
experimentally validated type II loci, where `T{n}` and `AT{n}` are the two
genes of one locus. The label is a literature annotation, independent of any
structure prediction - which is the point, since calibrating a structure
threshold on a structure-filtered candidate list would be circular. The builder
verifies that every locus's toxin and antitoxin come from the same organism and
**raises** if any disagree, rather than trusting the naming convention.

Three negative classes are emitted and scored **separately**, never pooled:

| class | construction | what it is for |
| --- | --- | --- |
| `cross_family` | antitoxin from a different toxin family | easy; good numbers here alone justify nothing |
| `same_family` | antitoxin from another locus in the same family | the honest test - composition, length and fold matched |
| `shuffled` | cognate antitoxin, residues permuted | sanity floor; a scorer that fails here is not reading structure |

`same_family` performance is reported as a **lower bound**: TA antitoxins
cross-react within families (this paper's own EvoAT2 neutralises three
different toxins), so some of those negatives may be genuine binders.

**Arm 2 - de novo functional pairs (`build_denovo_set.py`).** The paper's
growth-rescue outcomes on designed sequences. These labels are functional and
in the right domain, which makes this the arm a transferable threshold would
have to come from. Two safeguards are enforced in code, not prose: rows whose
outcome is `UNKNOWN` in `data/rescue_matrix.csv` are **skipped, never assumed**,
and the identity of `RelE` must be passed explicitly via `--rele-variant`
because this repository contains two different proteins that could be meant
(the natural homolog EvoRele1 was derived from, at 70.8% identity to it, and
E. coli RelE `P0C077` at 29.1%).

**Why both.** Natural complexes are very likely represented in AlphaFold 3's
training data, so a cutoff fitted on Arm 1 need not transfer to Evo output.
Arm 1 establishes whether a score discriminates cognate from non-cognate at
all - a prerequisite, not a calibration. Arm 2 tests the transfer. As shipped
Arm 2 has 9 positives and **1** negative, which is a spot-check; the builder
says so.

```bash
# Arm 1: build, score across a SLURM array, analyse
python -m proto_pipelines.calibration.build_tadb_set \
    --toxins proto_pipelines/calibration/data/type_II_T_exp.fas \
    --antitoxins proto_pipelines/calibration/data/type_II_AT_exp.fas \
    --out proto_pipelines/calibration/data/calibration_pairs_pilot.csv --n-per-class 20
sbatch --export=ALL,PAIRS=...,OUTDIR=...,NUM_SHARDS=8 \
    proto_pipelines/slurm/calibration_folds.sbatch
python -m proto_pipelines.calibration.analyze \
    --results proto_pipelines/calibration/results/pilot_natural
```

`analyze.py` reports AUROC per negative class with a percentile bootstrap
interval, plus the Youden operating point. Scoring goes through the same
`af3.score_complex` the cofold stage calls, so a threshold is measured exactly
the way it will be applied.

## Calibration results (measured)

Run on 80 natural TA pairs and 10 de novo pairs with experimental rescue
labels. Raw data in `calibration/results/`; summaries in
`pilot_natural/calibration_summary.csv` and `denovo/denovo_summary.csv`.

### What the scores can do

Against implausible complexes, every score works well (n=20 per class):

| score | vs cross-family | vs shuffled | vs same-family |
| --- | --- | --- | --- |
| pDockQ2 | **1.000** | **1.000** | 0.725 [0.558-0.865] |
| ipTM | 0.998 | 1.000 | 0.704 [0.525-0.848] |
| avg pLDDT | 0.943 | 0.993 | 0.670 [0.498-0.832] |
| pTM | 0.945 | 0.950 | 0.651 [0.470-0.812] |
| pDockQ v1 | 0.945 | 0.932 | 0.648 [0.470-0.805] |

The `shuffled` control separating at AUROC 1.000 confirms the scorers read
structure rather than amino-acid composition.

### What they cannot do

**Partner-level discrimination.** Against same-family non-cognate pairs every
score falls to 0.65-0.73 with confidence intervals reaching chance. At its
best operating point pDockQ2 reaches 95% sensitivity at 45% specificity.
pDockQ2's edge over ipTM is not separable at this n.

**Functional discrimination.** On the de novo arm the one experimentally
confirmed *non*-functional pair (MazF+EvoAT4: pDockQ2 0.017, ipTM 0.31,
pLDDT 81.0) is indistinguishable from two confirmed *functional* pairs
(YoeB+EvoAT4 0.018/0.31/78.3 and YoeB+EvoAT2 0.014/0.19/72.0). On both pLDDT
and pTM the negative sits **interior** to the positive distribution, so no
cut can separate them. The internal control — one antitoxin, four toxins —
splits the set cleanly but along fold-family compatibility, not rescue.

A caveat that is not resolvable from this data: growth rescue does not
require a tight 1:1 interface. A low score on a rescuing pair may be a
correct prediction of a weak direct interface rather than a prediction
failure.

### Chosen gates and their cost

`pdockq2 >= 0.23` AND `iptm >= 0.55` AND `avg_plddt >= 80`:

| set | retained |
| --- | --- |
| natural cognate | 19/20 |
| same-family negatives admitted | 12/20 |
| cross-family negatives admitted | 2/20 |
| shuffled negatives admitted | 0/20 |
| **de novo confirmed-functional** | **6/9** |

The ipTM and pLDDT gates are free in false-negative terms: they discard no
confirmed-functional pair that pDockQ2 >= 0.23 keeps, while cutting
cross-family admissions from 4/20 to 2/20.

`min_iptm` is placed by gap, not by optimum. The de novo positives cluster at
ipTM 0.74-0.86 and the rejects at 0.19-0.31, so any cut in ~[0.32, 0.74]
scores identically here. 0.55 sits mid-gap with +0.19 to the nearest
confirmed-functional pair; 0.70 would leave +0.04, which is smaller than the
seed-to-seed variation observed when the same pair was folded twice. The
extra specificity that buys is one cross-family pair.

`min_avg_plddt` is non-binding: across 78.5-81.5 it changes no outcome on any
class. It is a guard against degenerate predictions, not a discriminator.

**The gate set still rejects 3 of 9 confirmed-functional de novo pairs.**
That is the honest cost, and it is why these are documented as a plausibility
filter rather than a hit-caller. Expect real designs below the cut.

### Monomer gates: calibrated, and the answer is "do not gate"

The monomer screen is applied to Evo output, so it was calibrated on Evo
output: 62 designed chains from the paper's synthesis-order set (no BLAST
match, so genuinely novel) each against a shuffle of itself. A natural-protein
arm (25 TA chains + shuffles) was run alongside as a reference.

| set (62-101 aa, length-matched) | real median | shuffled median | pLDDT AUROC | pTM AUROC |
| --- | --- | --- | --- | --- |
| natural TA chains | 60.6 / 0.44 | 46.2 / 0.28 | **0.827** | **0.893** |
| **Evo designs** | 63.8 / 0.43 | **64.7** / 0.40 | **0.475** | 0.629 |

**Scrambling a natural protein collapses its predicted structure. Scrambling
an Evo design does not.** Length-matched, so this is a property of the
population and not of chain size - the natural arm actually separates *better*
when restricted to the same window.

Single-sequence AlphaFold 3 confidence on these designs is therefore not
reading sequence order: pLDDT is at chance (0.475) at distinguishing a design
from a scramble of itself, and pTM is weak. A score that ignores the property
being selected for cannot gate on it. Note the designs also score *higher*
than natural proteins (63.8 vs 60.6) while carrying less order-dependent
signal, so a naive reading calls them better folded.

Two further facts against gating here:

* The natural *E. coli* toxins MazF (pLDDT 38.9) and YoeB (41.5) score below
  **every one** of the 62 designs. Any cut that looks sensible on designs
  discards real, characterised toxins - and both YoeB pairs rescued
  experimentally.
* In `t2ta_sample` both members of a pair must survive, so monomer losses
  compound quadratically into the pairing step.

**Shipped: `af3_plddt_threshold: 0.0`, `af3_ptm_threshold: 0.0`** - the screen
runs but does not gate. The folds still populate `af3_fold_scores.csv`, which
is how you would calibrate a cut for your own generations if one turns out to
discriminate there. Set `run_af3: false` to skip the step and save the GPU
time outright.

The originally shipped placeholders (`70.0/0.5` and `60.0/0.4`) rejected every
observed de novo monomer, so they were wrong in the other direction.

A caveat on the positives: these designs passed the paper's ESMFold screen
before being ordered, so they are enriched for predicted foldability. That
biases *towards* finding separation, which makes the null result stronger
rather than weaker.

### Selecting binders: rank, do not threshold

For picking likely binding partners the decisive result is that **ranking
beats thresholding**, because the realistic false positive is a same-family
non-partner, not a random protein.

Absolute cutoffs against same-family decoys top out at ~2:1 enrichment
(pDockQ2 >= 0.85 gives 10/20 recall with 5/20 decoys admitted). Ranking the
candidate partners of a single toxin does far better:

| task | pDockQ2 | pDockQ v1 | ipTM | chance |
| --- | --- | --- | --- | --- |
| top-1 of 4 (all decoy types) | 80% | 80% | 80% | 25% |
| vs same-family decoy only | 80% | 85% | 80% | 50% |

Top-1-of-4 at 80% is binomial p = 3.9e-07; the paired sign test on the hard
same-family comparison gives p = 0.012 (pDockQ2) and p = 0.0026 (pDockQ v1).

So: use the gates to remove implausible complexes, then **rank within each
generation and take the top-k**. `t2ta_sample` enumerates pairs from one
generation, so the comparison is already matched.

Note that pDockQ v1 has the *worst* unpaired AUROC (0.648) yet the *best*
paired performance. A score can be badly calibrated in absolute terms - and
so useless for thresholding - while still preserving rank order within a
matched comparison. That is why it stays in the output table despite never
gating.

Ranking selects likely cognate partners. It does not predict function; the
de novo arm showed no score here does.
