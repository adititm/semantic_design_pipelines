# Type II toxin–antitoxin screening pipeline

Generates candidate type II TA systems and screens them by structure: each
generation is checked for a toxin and an antitoxin that fold, pair, and form
a plausible interface.

    python -m proto_pipelines.pipelines.t2ta_sample \
        --config proto_pipelines/configs/t2ta_sample.yaml

Unlike the anti-CRISPR pipeline, this one has an interface score that
genuinely discriminates, so it **gates** rather than only ranking. Defaults
and the effect of changing them are in [SETTINGS.md](SETTINGS.md).

## Ordering

**1. Generate** — Evo samples from TA-context prompts. `n_tokens: 2000`
gives room for two ORFs plus intergenic sequence; the Acr pipeline uses half
that because it needs only one.

**2. ORF calling + QC (`protein_qc`)** — Prodigal, then the published
predicates: length 50–300 aa, `partial=00`, k-mer repetitiveness,
≥12 unique amino acids, segmasker ≤0.1. Wider than the Acr window because a
toxin and its antitoxin differ in size.

**3. Profile-HMM filter (`profile_hmm`) — this one gates.** 219 TA family
profiles derived by scanning known type II TA proteins against Pfam. A
proposal must contain at least one protein hitting a TA family
(`hmm_min_matching_proteins: 1`) or it is rejected. Set
`hmm_annotate_only: true` to record hits without gating.

This is the opposite choice from the Acr pipeline, and deliberately: Pfam
covers TA families well, whereas most known Acrs hit no family at all, so
there an HMM miss means nothing.

**4. Monomer screen (`af3_monomer_screen`)** — AlphaFold 3 folds the QC
survivors; chains that fold too poorly to pair are dropped.

**5. Pairing and cofold (`ta_cofold`)** — surviving chains are enumerated
into pairs, filtered for redundancy and size, then cofolded. Each complex is
scored with pDockQ2, ipTM, pTM and average pLDDT.

Pair enumeration is capped (`cofold_max_pairs_per_proposal: 6`) because the
pair count grows quadratically with surviving chains and cofolding is the
most expensive step in either pipeline.

## The interface gate

A pair passes on **all three** of:

```
pdockq2  >= 0.23
iptm     >= 0.55
avg_plddt >= 80
```

Measured cost of that combination:

| set | retained |
| --- | --- |
| natural cognate pairs | 19/20 |
| same-family negatives admitted | 12/20 |
| cross-family negatives admitted | 2/20 |
| shuffled negatives admitted | **0/20** |
| de novo confirmed-functional | 6/9 |

Two things worth understanding before changing these.

**Same-family negatives are the hard case.** 12 of 20 get through. A toxin
paired with a non-cognate antitoxin from the same family still forms a
plausible-looking interface, because the interface is a family property.
The gate separates TA-like from not-TA-like, not cognate from non-cognate.

**`min_iptm` is placed by gap, not by optimum.** De novo positives cluster
at ipTM 0.74–0.86 and rejects at 0.19–0.31, so any cut in roughly
[0.32, 0.74] scores identically on this data. 0.55 sits mid-gap. Tightening
toward 0.70 leaves only +0.04 to the nearest confirmed-functional pair,
which is smaller than the seed-to-seed variation seen when the same pair was
folded twice — so it would start rejecting on noise.

The ipTM and pLDDT gates are free in false-negative terms: they discard no
confirmed-functional pair that `pdockq2 >= 0.23` keeps, while halving
cross-family admissions.

## Novelty filters

| key | default | what it does |
| --- | --- | --- |
| `max_pair_identity` | `70.0` | Drops pairs whose two chains are too similar to each other, i.e. a protein paired with a near-copy of itself. |
| `novelty_max_identity` | `75.0` | Drops candidates too close to the natural TA proteins in the reference set, so a passing pair is not a rediscovery. |

## Reading the output

| file | contents |
| --- | --- |
| `generated_sequences.csv` | One row per proposal with its outcome |
| `qc_proteins.csv` | Every ORF, with `passed_qc` and `rejected_by` |
| `hmm_proteins.csv` | Profile-HMM hits per protein |
| `af3_fold_scores.csv` | Monomer pLDDT/pTM |
| `cofold_pairs.csv` | Every pair scored, with pDockQ2, ipTM, pTM, pLDDT and pass/fail |
| `filter_summary.csv` | Per-stage in/out/rejected counts, in execution order |

## Honest limits

* **A passing pair is a structural prediction, not a functional TA system.**
  The de novo arm showed no score here predicts functional rescue; the
  calibration measured which pairs *look* like TA complexes.
* **Cognate specificity is not tested.** Same-family non-cognate pairs pass
  at 12/20. If you need cognate pairing, this pipeline does not establish it.
* **AlphaFold 3 sampling is not deterministic** and `seed` does not fix it.
  Reuse the same AlphaFold 3 output directory across runs to make repeat
  folds both free and reproducible.
* **The de novo arm is n=9 confirmed-functional pairs.** Every threshold
  placed against it carries that uncertainty; the natural-pair arm (n=20) is
  what the separation figures rest on.
