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
| `acr_prescreen_min_score` | `0.15` | Keeps 95% of known Acrs and 95% of a never-used held-out set, while folding 27% of other defense proteins. Raising to 0.25 keeps ~88%; lowering approaches "fold everything". An ORF with an HMM hit **always** folds regardless. |
| `acr_fold_fraction` | `1.0` | Caps folds per proposal as a fraction of survivors, after prescreen ordering. Below 1.0 trades recall for compute. |

A proposal whose ORFs all score below threshold still folds its single best
ORF, so a weak generation yields a structure rather than looking like a
pipeline failure.

## 3b. Profile-HMM filter (`profile_hmm`, TA pipeline only)

| key | default | effect of changing it |
| --- | --- | --- |
| `hmm_min_matching_proteins` | `1` | A proposal must contain at least one protein hitting a TA family, or it is rejected. |
| `hmm_evalue_threshold` | `1.0` | Permissive; the family set is specific enough that recall binds. |
| `hmm_annotate_only` | `false` | `true` records hits without rejecting anything. |
| `hmm_required_profiles` | `[]` | Restrict to named families; empty accepts any TA family. |

## 4. Structure screen (`af3_monomer_screen`)

AlphaFold 3 folds the survivors; low-confidence folds stop being candidates.

| key | default | effect of changing it |
| --- | --- | --- |
| `af3_use_msa` | `false` | An MSA raises absolute pLDDT hugely (Acr median 49 → 87) but does not improve Foldseek TM discrimination (AUROC 0.643 without vs 0.635 with; 95% CI on the difference [-0.091, +0.070]). Deep- and shallow-MSA Acrs differ by 27 pLDDT points under an MSA and ~5 without. Divergent candidates are the shallow-MSA ones, therefore, MSA is deliberately turned off as to not bias towards selecting candidates that are highly similar to known Acrs. Turning it on also adds the dominant share of fold runtime. |
| `af3_plddt_threshold` | `30.0` | Real Acrs fold badly (median pLDDT 39.4). While low, the pLDDT and pTM thresholds were selected as they was the most efficient point available and remained permissive to both de novo Acrs and natural Acrs while removing 8x junk/scrambled sequences over natural Acrs. |
| `af3_ptm_threshold` | `0.20` | pTM discriminates better than pLDDT. |
| `af3_max_proteins_per_proposal` | `8` | Caps folds per proposal. HMM-qualifying ORFs sort first and a warning fires if the cap drops one. |
| `af3_num_recycles` / `af3_num_diffusion_samples` | `10` / `5` | More of each raises confidence and cost. |

This gate is serves to help remove obviously junk sequences, but is not a discriminator,
as fold confidence barely separates Acrs from other phage proteins.

## 5. Anti-CRISPR evidence (`acr_evidence`)

All five callers run over every folded ORF. Results go to
`acr_evidence.csv`, one row per protein.

| key | default | effect of changing it |
| --- | --- | --- |
| `acr_min_score` | `0.0` | Gating on these callers tends to select for resemblance to known Acrs, which may not be desirable. Raise only if you accept that bias. |
| `acr_require_af3_pass` | `true` | Proteins that failed the fold gate stay in the table and still contribute locus context — an Aca partner is identified by sequence HMM and is useful even with a poor fold — but cannot be candidates. |
| `acr_hmm_evalue` | `1.0` | Permissive by design, tighten to select for greater similarity to known Acrs. |
| `acr_acranker_shuffles` | `20` | Not used by default. Self-shuffle null for `acranker_z`. More is slower and slightly less noisy. `acranker_z` is a relatively weak differentiator and as such is not in either present model. |
| `acrnet_psiblast` / `acrnet_blast_db` | set | Optional. Without a PSSM AcrNET still ranks, but its scores inflate and are read against the weaker `no_pssm` calibration. Supply the database if you can. |
| `acrnet_workers` | `8` | Concurrent RaptorX/PSI-BLAST processes. PSI-BLAST scales by process, not thread. |
| `acrnet_device` | `cuda` | **Local torch device only.** ESM-1b is loaded directly rather than dispatched as a proto tool, so `proto`/`modal` raise here. |

## 5a. Pairing and cofold (`ta_cofold`, TA pipeline only)

Surviving chains are enumerated into pairs and cofolded; the interface is
scored with pDockQ2, ipTM, pTM and average pLDDT. Unlike the Acr callers,
this gate genuinely discriminates, so it rejects rather than only ranking.

| key | default | effect of changing it |
| --- | --- | --- |
| `cofold_min_iptm` | `0.55` | Placed by gap, not optimum: de novo positives sit at 0.74-0.86 and rejects at 0.19-0.31, so any cut in ~[0.32, 0.74] scored roughly the same. 0.70 would leave +0.04 to the nearest functional pair, below the seed-to-seed variation, so could result in rejection on noise. Raise to select for more confident interfaces, at the risk of penalizing more de novo sequences |
| `cofold_min_avg_plddt` | `80.0` | Raise to select for more confident sequences. |
| `cofold_min_passing_pairs` | `1` | Pairs a proposal needs to be accepted. |
| `cofold_max_pairs_per_proposal` | `6` | Pair count grows quadratically with surviving chains and cofolding is the most expensive step; this caps it. |
| `max_pair_identity` | `70.0` | Drops pairs whose chains are near-copies of each other. |
| `novelty_max_identity` | `75.0` | Drops candidates too close to natural TA proteins, so a pass is not a rediscovery. |

## 6. Ranking

`acr_locus_score` is a logistic output, not P(Acr). Use
`utils/rank_candidates.py` to calibrate it to your own base rate and get an
expected yield; see [ACR_PIPELINE.md](ACR_PIPELINE.md).
