# Pipeline stages and settings

What each stage does, what it is set to by default, and what changing it
costs. Every figure below was measured on the calibration sets; where a
setting has not been measured, this says so rather than guessing.

Settings are YAML keys. Unknown or retired keys are **rejected at startup**
rather than ignored, so a typo fails immediately instead of silently
running with a default.

---

## 1. Generation

Evo samples DNA from each prompt. Nothing downstream is generator-specific.

| key | default | effect of changing it |
| --- | --- | --- |
| `generator` | `evo2` | `evo1` for the Evo 1.5 checkpoints the paper used. The checkpoint must belong to the family; mismatching them fails at startup. |
| `model_name` | `evo2_7b` | Selects the architecture. With `model_local_path` set it must still name the variant your weights derive from. |
| `model_local_path` | `""` | Load local weights instead of the HuggingFace download. Evo 2 only. Empty uses the released weights. |
| `n_tokens` | `1000` | New tokens per sample. Below ~600 the generated region often cannot hold a complete ORF at the QC length bounds. |
| `temperature` | `0.8` | Higher diversifies and lowers per-sample quality; lower collapses toward the prompt's local context. |
| `top_k` | `4` | The published setting. Raising it widens the sampled distribution. Untested here. |
| `n_sample_per_prompt` | `5` | Linear in cost and in candidate count. |
| `seed` | `null` | Seeds the optimizer and Evo sampling. **Does not make AlphaFold 3 deterministic** — see "reproducibility" below. |

## 2. ORF calling and protein QC (`protein_qc`)

Prodigal calls ORFs, then the published predicates filter them. Rejections
are recorded per ORF in `qc_proteins.csv` with the reason.

| key | default | effect of changing it |
| --- | --- | --- |
| `filter_min_length` / `filter_max_length` | `50` / `200` | Amino acids. The Acr window; most characterised Acrs are 50–150 aa. Widening admits more ORFs and proportionally more compute downstream. |
| `filter_partial_bool` | `true` | Requires Prodigal `partial=00`, i.e. both ends within the generated region. Turning it off admits truncated ORFs whose folds are not meaningful. |
| `segmasker_threshold` | `0.1` | Maximum low-complexity fraction. Raising it admits repeat-rich sequence that folds poorly and inflates composition-based callers. |

## 3. Sequence prescreen (`acr_prescreen`, Acr pipeline only)

HMM, AcRanker and AcrNET run on sequence alone and rank the ORFs, so the
fold step can skip the least Acr-like tail. **This gates folding, never the
final call.**

| key | default | effect of changing it |
| --- | --- | --- |
| `run_acr_prescreen` | `true` | Off folds every QC survivor. Correct but slower. |
| `acr_prescreen_min_score` | `0.15` | Keeps 95% of known Acrs and 95% of a never-used held-out set, while folding 27% of phage. Raising to 0.25 keeps ~88%; lowering approaches "fold everything". An ORF with an HMM hit **always** folds regardless. |
| `acr_fold_fraction` | `1.0` | Caps folds per proposal as a fraction of survivors, after prescreen ordering. Below 1.0 trades recall for compute. |

A proposal whose ORFs all score below threshold still folds its single best
ORF, so a weak generation yields a structure rather than looking like a
pipeline failure.

## 4. Structure screen (`af3_monomer_screen`)

AlphaFold 3 folds the survivors; low-confidence folds stop being candidates.

| key | default | effect of changing it |
| --- | --- | --- |
| `af3_use_msa` | `false` | **Measured, not assumed.** An MSA raises absolute pLDDT hugely (Acr median 49 → 87) but does not improve Foldseek TM discrimination (AUROC 0.643 without vs 0.635 with; 95% CI on the difference [-0.091, +0.070]). It *does* confound the gate: Spearman(MSA depth, pLDDT) is +0.40 with an MSA and −0.19 without, so deep- and shallow-MSA Acrs differ by 27 pLDDT points under an MSA and ~5 without. Divergent candidates are the shallow-MSA ones. Turning it on also adds the dominant share of fold runtime. |
| `af3_plddt_threshold` | `30.0` | Real Acrs fold badly — median pLDDT 49.4 single-sequence — so this sits near their 1st–5th percentile and discards 2/64. It is also the most efficient point available: 27% of junk removed per 3% of Acrs lost (8.5×), against 4.7× at 35 and 2.5× at 40. |
| `af3_ptm_threshold` | `0.20` | The load-bearing half of the gate; pTM discriminates better than pLDDT (AUROC 0.617 vs 0.548 against phage). |
| `af3_max_proteins_per_proposal` | `8` | Caps folds per proposal. HMM-qualifying ORFs sort first and a warning fires if the cap drops one. |
| `af3_num_recycles` / `af3_num_diffusion_samples` | `10` / `5` | More of each raises confidence and cost. **The gate was calibrated at 3/1**, so the effective gate is marginally more lenient than the figures above. Unmeasured. |

The gate is a **junk floor, not a discriminator**: fold confidence barely
separates Acrs from real phage proteins. On Evo output versus random-DNA
ORFs it keeps 92% of Evo ORFs while removing 46% of random ones — but it
cannot tell a poor Evo ORF from a good one. That is the five callers' job.

## 5. Anti-CRISPR evidence (`acr_evidence`)

All five callers run over every folded ORF. Results go to
`acr_evidence.csv`, one row per protein.

| key | default | effect of changing it |
| --- | --- | --- |
| `acr_min_score` | `0.0` | **Record evidence, reject nothing.** The callers were calibrated on natural Acrs, so gating on them selects for resemblance to known Acrs — the opposite of the point. Raise only if you accept that bias. |
| `acr_require_af3_pass` | `true` | Proteins that failed the fold gate stay in the table and still contribute locus context — an Aca partner is identified by sequence HMM and is useful even with a poor fold — but cannot be candidates. |
| `acr_hmm_evalue` | `1.0` | Permissive by design: the profile set has 0 false positives across 191 negatives, so recall is the binding constraint, not precision. |
| `acr_acranker_shuffles` | `20` | Self-shuffle null for `acranker_z`. More is slower and slightly less noisy. `acranker_z` is weak (AUROC 0.605 vs phage) and is **not** in either shipped model. |
| `acrnet_psiblast` / `acrnet_blast_db` | set | Optional. Without a PSSM AcrNET still ranks, but its scores inflate and are read against the weaker `no_pssm` calibration. Supply the database if you can. |
| `acrnet_workers` | `8` | Concurrent RaptorX/PSI-BLAST processes. PSI-BLAST scales by process, not thread. |
| `acrnet_device` | `cuda` | **Local torch device only.** ESM-1b is loaded directly rather than dispatched as a proto tool, so `proto`/`modal` raise here. |

### Caller strength, measured against phage

Natural Acrs versus natural phage proteins, which is the operationally
relevant comparison:

| caller | AUROC vs phage |
| --- | --- |
| AcrNET (within-regime percentile) | 0.918 |
| AcRanker raw | 0.801 |
| AcRanker z | 0.605 |
| profile HMM | 23% sensitivity at **0 false positives** |
| prescreen model, held-out Acrs | **0.937** |

The held-out figure uses 19 Acrs never involved in any derivation, so there
is no generalisation gap on natural sequence. Performance on *generated*
sequence is uncharacterised — there is no labelled set for it.

## 6. Ranking

`acr_locus_score` is a logistic output, not P(Acr). Use
`utils/rank_candidates.py` to calibrate it to your own base rate and get an
expected yield; see [ACR_PIPELINE.md](ACR_PIPELINE.md).

## Reproducibility

AlphaFold 3 sampling is not deterministic and `seed` does not fix it: the
same sequence folded three times moved `acr_locus_score` from 0.048 to
0.477. Rank within a run; do not compare absolute scores across runs.
Pointing successive runs at the same AlphaFold 3 output directory makes
repeat folds both free and deterministic, because the job name is a content
hash and the score cache sits beside the structures.
