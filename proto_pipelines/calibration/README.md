# Calibration

Scripts that **derive** the shipped models. None of this runs at pipeline
time: the fitted artefacts in `data/` are loaded directly, and a clone can
run every workflow without executing anything here.

Read `../docs/CALIBRATION.md` for what the measurements support, and
`../docs/ACR_PIPELINE.md` for the anti-CRISPR stack specifically.

## Layout

| path | tracked | what |
| --- | --- | --- |
| `data/` | yes | The artefacts the pipeline loads — fitted logistic models, profile-HMM family subsets, the AcRanker booster, the AcrNET checkpoint, the Foldseek reference database, labelled sequence sets. |
| `results/` | **no** | Intermediates: AF3 structures, per-caller score tables, cached AcrNET features. ~1.6 GB, gitignored, reproducible from the scripts below. |
| `*.py` | yes | The scripts. Each has a usage block in its docstring. |

## What the scripts do

**Set construction** — `build_acr_set.py`, `build_tadb_set.py`,
`build_monomer_set.py`, `build_denovo_set.py`, `build_denovo_monomer_set.py`,
`build_junk_monomer_set.py` assemble labelled sequence sets with their
negative classes.

**Family discovery** — `discover_ta_families.py` and `build_ta_hmm_subset.py`
scan Pfam to derive the profile-HMM subsets that ship in `data/`. These need
Pfam-A, which is gitignored and downloaded on demand.

**Scoring** — `score_acrnet.py`, `score_acranker.py`, `score_acr_hmm.py`,
`score_acr_foldseek.py` run one caller across a labelled set and report its
discrimination. `score_acrnet.py` also takes `--dump-features`, which caches
extracted features so a *scoring* change can be compared on identical inputs
rather than re-extracting — use it before changing anything in `acrnet.py`.

**Folding** — `run_folds.py` (pairs, pDockQ2) and `run_monomer_folds.py`
(single chains, pLDDT/pTM) are shardable across a SLURM array.

**Analysis** — `analyze.py` turns scored sets into thresholds;
`rank_candidates.py` turns a ranked candidate list into a defensible top-k.

## Regenerating

`../slurm/acrnet_rescore.sbatch` is the worked example. Point `--sequences`
at another labelled set to score it; the one-off jobs that produced the
shipped assets are in `../archive/slurm/` for reference.

Anything you regenerate lands in `results/`, which is gitignored — copy the
fitted artefact into `data/` deliberately, and update the parity check that
pins its feature order.

`*.hmm` and `*.ckpt` under `data/` are tracked in Git LFS (patterns, so a
regenerated profile set is captured whatever it is named). Run
`git lfs install` once per machine; without it a clone gets ~130-byte
pointer files and pyhmmer reports a malformed profile rather than a missing
one.
