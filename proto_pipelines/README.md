# Semantic design pipelines

Runnable, configurable scripts for the pipelines in [*Semantic design of
functional de novo genes from a genomic language
model*](https://www.nature.com/articles/s41586-025-09749-7).

Each workflow is a single YAML-configured entry point that runs end to end —
sequence generation, ORF calling, quality filtering, profile-HMM screening,
structure prediction and interface scoring — with every tool provisioned
automatically into its own environment. Nothing to install per tool, no
binary paths to configure, and every prompt and reference file needed is
bundled here.

```bash
python -m proto_pipelines.pipelines.t2ta_sample        --config proto_pipelines/configs/t2ta_sample.yaml
python -m proto_pipelines.pipelines.acr_sample         --config proto_pipelines/configs/acr_sample.yaml
python -m proto_pipelines.pipelines.gene_completion    --config proto_pipelines/configs/gene_completion.yaml
python -m proto_pipelines.pipelines.operon_completion  --config proto_pipelines/configs/operon_completion.yaml
```

Run from the repository root. Structure prediction needs a GPU; everything
before it is CPU-only.

## Quickstart

```bash
git lfs install && git lfs pull      # profile HMMs and the AcrNET checkpoint

conda create -y -p ./envs/proto python=3.11 && conda activate ./envs/proto
export PYTHONNOUSERSITE=1
pip install "git+https://github.com/evo-design/proto-language.git"
pip install -e "./proto_pipelines[acr]"
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121   # match your driver

export PROTO_HOME=/path/you/own/proto_home   # the runner derives the rest

python proto_pipelines/tests/test_parity.py     # 18 checks, no GPU needed
proto_pipelines/scripts/run_pipeline.sh acr_sample \
    proto_pipelines/configs/smoke/acr_sample_smoke.yaml
```

The parity suite is the install check and needs neither a GPU nor AF3
weights. The smoke config is a two-prompt run that exercises every stage; it
took 54 minutes on one H100, most of it AlphaFold 3 MSAs and first-use
tool-environment builds.

**No GPU?** Set `device: modal` (or `proto`) in the config and the heavy tools
run on connected compute — no local accelerator, AF3 weights or MSA database
required. **Slurm is optional**: nothing in the pipelines needs it, and
`run_pipeline.sh` covers every mode. See
[Execution modes](#execution-modes) and full [Setup](#setup) below.

## The four workflows

| workflow | what it does |
| --- | --- |
| `t2ta_sample` | Type II toxin–antitoxin: generate → QC → TA profile-HMM → monomer fold → pair → cofold + interface scoring |
| `acr_sample` | Anti-CRISPR candidates: generate → QC → sequence prescreen → monomer fold → five-caller Acr/Aca evidence → ranked candidates. Replaces the published PaCRISPR step, which has no usable offline release, with callers that run locally ([docs/ACR_PIPELINE.md](docs/ACR_PIPELINE.md)) |
| `gene_completion` | How closely a truncated gene is completed, by MAFFT identity to a reference |
| `operon_completion` | Whether the downstream operon genes are produced |

Type III TA is not included: it needs Infernal (`cmscan`) and Tandem Repeat
Finder, neither of which has a managed-environment wrapper.

## Generator

`evo2` / `evo2_7b` by default. Both the family and the checkpoint are config
keys:

```yaml
generator: evo2          # evo2 | evo1
model_name: evo2_7b      # evo2_40b, evo2_1b_base, evo-1.5-8k-base, ...
```

Evo 2 returns only `(sequence, logits, kv_cache)` in this proto-tools build
and writes no generator metadata, so the `evo_score` column is empty under
it; the run warns once per prompt. Evo 1.5 populates it. Nothing else
differs between the two.

```yaml
```

The published work used **Evo 1.5**, so reproducing its numbers means setting
`generator: evo1` and `model_name: evo-1.5-8k-base`. Note also that every
threshold shipped here was measured on Evo 1.5 output; on Evo 2 they are
an assumption until re-measured.

## Bundled data

```
data/prompts/      t2ta, acr, gene-completion, operon-completion, t3ta prompt CSVs
data/reference/    rpoS and modABC reference proteins for the completion workflows
data/models/       fitted models + profile HMMs the pipelines load
```

Every prompt from the paper's four supported workflows is included, so each
`configs/*.yaml` runs as shipped with no extra downloads. The t3ta prompts are
bundled for reference only — there is no t3ta script here.

## Structure filtering, in one paragraph

Structure confidence gates plausibility, not function. On functionally
labelled de novo pairs the one known non-functional pair was indistinguishable
from two confirmed-functional ones, and the shipped gate set rejects 3 of 9
confirmed-functional pairs. For selection, **rank within a generation and take
the top-k** (80% top-1-of-4 versus 25% chance) rather than thresholding, which
tops out near 2:1 enrichment. The calibration behind every number is below.

## Anti-CRISPR calling (`acr.py`)

> Full pipeline ordering, calibration tables and operating guidance:
> **[docs/ACR_PIPELINE.md](docs/ACR_PIPELINE.md)**


The paper called Acrs with PaCRISPR, a web server that cannot be run offline
and is not part of the released code. It is replaced here by four independent
callers combined by a logistic model calibrated on 316 labelled sequences.

**Calibration set** (in the research tree): 64
experimentally named Acrs spanning CRISPR types I, II, III, V and VI, against
four negative classes — composition-matched shuffles, random-DNA ORFs,
length-matched non-Acr **bacteriophage** proteins, and **Aca** proteins. All
classes are caliper-matched on length (medians 106–109 aa), because the Type
II TA calibration found an apparent effect that was length alone.

| caller | what it adds | measured |
|---|---|---|
| profile HMM (16 families) | precision | 23% sensitivity, **0/191 false positives** |
| AcRanker (raw + self-shuffle z) | composition/mechanism | 0.80 vs phage alone |
| Foldseek vs 189 PDB Acr chains | fold, survives shuffling | **0.75 vs shuffles** — best single |
| AlphaFold 3 pLDDT | foldability | 0.87 vs junk |

**Scoring is two-tier, because the two regimes behave completely
differently.** A profile-HMM hit is near-certain (0 false positives across 191
negatives) and the full five-feature model reaches **0.98** AUROC on that
subset. But **49 of 64 known Acrs have no Pfam hit at all**, and for those the
full model *penalises* the missing feature and collapses to **0.64** -- it
reads HMM absence as evidence against. The divergent tier drops the HMM term
and uses `acranker_raw + best_tmscore`, reaching **0.775** vs phage. AlphaFold
pLDDT is deliberately excluded from that tier: it helps against junk but
*hurts* against real proteins (0.775 -> 0.740).

| regime | n | model | AUROC vs phage |
|---|---|---|---|
| HMM hit (canonical) | 15 | 5-feature | **0.98** |
| no HMM hit (divergent) | 49 | AcRanker + Foldseek | **0.775** |

Cross-validated overall (5-fold, 5 seeds): Acr vs other phage proteins 0.836
-- but that figure is dominated by the canonical subset and should not be
quoted for divergent candidates. On 19 held-out Acrs never used in any
derivation: 0.905 vs all non-Acr negatives. Those held-out Acrs scored
*higher* than the calibration positives (median 0.905 vs 0.623), i.e. that
set is more canonical, so 0.905 is optimistic for divergent sequence.

A fifth caller was tested and rejected: MMseqs2 against 67k predicted Acrs
(`AcrDatabase.faa`) reached only 0.59-0.69 once close homologues were
excluded, and made the Aca confusion worse (0.27).

Three findings determine how to read the output.

**Acr and Aca cannot be separated, and should not be.** Every caller confuses
them — naive HMM expansion produced 82% Aca false positives, and Foldseek
scores Aca *above* Acrs (0.373). The cause is real biology, not a bug:
AcrIIA1, AcrIIA13, AcrIIA15 and AcrIF24 carry HTH domains and repress their
own operons, and Aca proteins are HTH regulators. The Aca hits trace to
AcrIIA6/AcrIIA15 reference structures. `acr_locus_score` therefore targets
"protein from an Acr locus"; `hmm_profiles` says which side it looks like.
Since an Acr-context prompt generates both, an Aca-like partner is a
*positive* signal for the locus.

**Much of the apparent signal is foldability.** pLDDT alone separates Acrs
from shuffles/random at 0.87 — better than any Acr-specific caller. The
Acr-specific question is the comparison against other *real* proteins
(phage), where Foldseek and AcRanker lead and pLDDT contributes least.

**Pfam under-covers Acrs.** Only 18/64 (28%) of known Acrs hit *any* Pfam
family at E<0.01, so ~28% is the ceiling for any Pfam-based caller. The
shipped set reaches 23%, and expanding further costs specificity.

Gating is **off by default** (`acr_min_score: 0.0`): the callers are
calibrated on natural Acrs, so gating on them selects for resemblance to
known Acrs — the opposite of what the pipeline is for. Evidence is recorded
to `acr_evidence.csv` with every caller's score side by side; raise the
threshold only after inspecting your own run.

**AlphaFold 3 sampling is not deterministic, and it propagates.** The same
sequence folded in three runs gave pLDDT 67.6 / 62.2 / 68.4; Foldseek TM went
0.40 / 0.42 / 0.00, moving `acr_locus_score` from 0.048 to 0.477. Rank within
a run, never compare absolute scores across runs, and reuse the AlphaFold 3
output directory so content-hashed folds are shared (which makes them
deterministic as well as free). Measured at smoke settings (1 diffusion
sample); the production config's 5 samples should damp this, unquantified.

None of this proves a sequence is an anti-CRISPR. These are similarity and
confidence measures — level-1 evidence throughout.


## Execution modes

The pipelines do not care where the heavy tools run. One script covers every
case:

```bash
scripts/run_pipeline.sh acr_sample configs/smoke/acr_sample_smoke.yaml
```

It resolves the repo root from its own location, derives the asset roots from
`PROTO_HOME`, runs the parity checks, and refuses to start a folding pipeline
without AlphaFold 3 weights unless you are folding remotely.

| mode | `device:` | what you need |
| --- | --- | --- |
| **Local GPU** | `cuda` | A GPU, AF3 weights, and the MSA database for folding pipelines |
| **Slurm** | `cuda` | The same, plus a cluster. `--wrap` the runner — see [Running under Slurm](#running-under-slurm) |
| **Connected compute** | `proto` or `modal` | Credentials only. No local GPU, no weights, no databases |

### Connected compute

proto-tools accepts two remote device strings in place of `cuda`, so the same
config runs without any local accelerator:

```yaml
device: proto     # Proto's hosted service
device: modal     # your own Modal deployment
```

`modal` needs `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` (or `~/.modal.toml`);
without them dispatch fails immediately with a credentials error rather than
falling back to CPU. `af3_msa_device` takes the same values, so the MSA search
can be dispatched independently of folding.

Two limits worth knowing before you rely on this:

* **`acrnet_device` must stay local** (`cpu` / `cuda`). ESM-1b is loaded with
  torch directly rather than dispatched as a proto tool, so a remote value
  there raises with an explanation instead of being silently routed nowhere.
  AcrNET's other two inputs — RaptorX and PSI-BLAST — are local binaries and
  are unaffected.
* **`model_local_path` and remote devices are mutually exclusive.** A local
  weights directory does not exist on a hosted worker, and proto-tools says
  so. Sample a custom checkpoint on hardware that can see the weights.

### Running under Slurm

There is no Slurm-specific code and no template to maintain: the runner is a
plain script, so `--wrap` it. What is worth keeping is the resource sizing,
which is not obvious from the outside:

| job | CPUs | GPU | memory | wall time |
| --- | --- | --- | --- | --- |
| `acr_sample`, `t2ta_sample` (fold) | 32 | 1 | 256 GB | 14 h |
| `gene_completion`, `operon_completion` | 16 | 1 | 128 GB | 2 h |
| `build_blastdb.sh` | 32 | – | 256 GB | 8 h |

```bash
sbatch --job-name=acr --partition=YOUR_PARTITION \
       --cpus-per-task=32 --gpus=1 --mem=256G --time=14:00:00 \
       --output=logs/%x_%j.out --error=logs/%x_%j.err \
       --export=ALL,PROTO_HOME=$HOME/proto_home,PYTHON=$(which python) \
       --wrap="proto_pipelines/scripts/run_pipeline.sh acr_sample \
               proto_pipelines/configs/acr_sample.yaml"
```

`mkdir -p logs` first, since Slurm will not create the log directory. On a
cluster with mixed GPU sizes, exclude the small-memory nodes — Evo 2 7B with
a long prompt will not fit in 20 GB. `build_blastdb.sh` picks up
`SLURM_CPUS_PER_TASK` automatically.

## Custom Evo 2 checkpoints

Any pipeline can sample from local Evo 2 weights instead of the released
ones — a fine-tuned, ablated or watermarked checkpoint — by setting one key:

```yaml
generator: evo2
model_name: evo2_7b                  # the ARCHITECTURE the weights load into
model_local_path: /path/to/weights   # a DIRECTORY, not a checkpoint file
```

`model_local_path` replaces the HuggingFace download; `model_name` still
selects the architecture, so it must name the variant your checkpoint was
derived from. Everything downstream — ORF calling, QC, HMMs, folding, the
Acr callers — is generator-agnostic and needs no change, so a watermarked
model can be run through the identical filter stack and compared against the
stock model run from the same prompts.

Two misuses raise instead of degrading quietly, both covered by a parity
check:

* **Evo 1 + `model_local_path`** raises. `Evo1GeneratorConfig` has no
  `local_path` field, so the path would be dropped and you would sample the
  stock model without any indication.
* **A checkpoint file rather than a directory** raises before the tool
  environment is reached, where the error would otherwise be opaque.

Leave it empty (the default) to use the released weights. Scoring uses the
same checkpoint as sampling, so `evo_score` in the output is the custom
model's likelihood, not the stock model's.

## Calibration

The thresholds and models shipped in `data/models/` are measured, not
guesses. The workflows that fit them -- labelled sequence sets, per-caller
scoring, AUROC analysis -- are **not** in this repository; it carries only
what is needed to run a sampling pass and get results.

They live in the research tree this was exported from. Nothing here loads
them: the fitted artefacts are read directly from `data/models/`.

## Layout

Nothing at the top level is a script: those fourteen modules are the library,
none has a `__main__`, and every one is imported by something below. The only
things you run are the four pipelines, the scripts in `scripts/`, and the
tests.

```
proto_pipelines/
│
│   ── library (imported, never run directly) ──
├── __init__.py           Runs bootstrap.activate() before anything else imports
├── bootstrap.py          Resolves proto_language/proto_tools before first import
├── prompts.py            Prompt-CSV loading, metadata columns retained
├── qc.py                 Prodigal ORF calling + the paper's protein QC predicates
├── hmm.py                Profile-HMM filter over QC survivors
├── af3.py                AlphaFold 3 monomer screen, complex scoring, pDockQ v1
├── cofold.py             Pair enumeration + AF3 cofold constraint + novelty filter
├── identity.py           The three MAFFT identity definitions
├── acr.py                Five-caller anti-CRISPR evidence + Aca co-occurrence
├── acrnet.py             AcrNET feature extraction (RaptorX, PSI-BLAST, ESM-1b)
├── acrnet_model.py       AcrNET architecture, verbatim from the published repo
├── runner.py             One Program per prompt; proposal history -> records
├── reporting.py          Stage CSVs, filter_summary, fold scores, FASTA
├── config.py             YAML loading; rejects retired and unknown keys
│
│   ── entry points (python -m ...) ──
├── pipelines/
│   ├── t2ta_sample.py        generation -> QC -> HMM -> monomer -> cofold
│   ├── acr_sample.py         generation -> QC -> prescreen -> fold -> Acr evidence
│   ├── gene_completion.py    generation -> QC -> MAFFT identity
│   └── operon_completion.py  generation -> QC -> MAFFT identity
├── tests/test_parity.py  18 CPU checks; the install check
│
│   ── data and settings ──
├── configs/              One YAML per pipeline, plus configs/smoke/ for fast runs
├── data/
│   ├── prompts/              bundled prompt CSVs
│   ├── reference/            reference sequences for the identity pipelines
│   └── models/               the fitted models and profiles the pipeline loads
├── scripts/
│   ├── run_pipeline.sh       run any pipeline: local, Slurm, or remote
│   ├── run_smoke.sh          run every smoke config, in cost order
│   └── build_blastdb.sh      build UniRef30 for AcrNET's PSSM (no cluster needed)
└── docs/
    └── ACR_PIPELINE.md   the anti-CRISPR stack in detail
```

The import graph is a DAG, four modules deep at most:

```
bootstrap -> (everything, via __init__)
prompts   -> runner -> config, reporting
identity  -> cofold -> af3
acrnet    -> acr -> af3
qc, hmm, af3, cofold, acr, config, reporting, runner -> pipelines/*
```

## Setup

Verified end to end on 2026-09-24 against `evo-design/proto-language` `main`
(`1a56693e`) and its pinned `proto-tools` submodule (`55339880`), in a clean
environment with no other packages on the path.

### 1. Environment

```bash
conda create -y -p /path/to/envs/proto python=3.11
export PYTHONNOUSERSITE=1          # keep ~/.local out of a clean env
pip install "git+https://github.com/evo-design/proto-language.git"
```

If that clone fails with a TLS error (`GnuTLS recv error (-110)`), clone and
install from the working copy instead — the submodule is fetched either way:

```bash
git clone --recurse-submodules git@github.com:evo-design/proto-language.git /tmp/pl
pip install /tmp/pl/proto-tools /tmp/pl
```

### 2. Anti-CRISPR extra

The Acr callers need four packages that **do not** arrive with proto-language
or proto-tools. Each caller is optional at runtime (an empty `acrnet_home` or
`acr_acranker_model` disables it), so they are an extra:

Run from the directory *containing* `proto_pipelines/`:

```bash
pip install -e "./proto_pipelines[acr]"
```

**Install `torch` from the wheel index matching your driver.** An unpinned
`torch>=2.0` resolves to a CUDA 13 build, which dies at GPU init on a 12.x
driver with *"The NVIDIA driver on your system is too old"*:

```bash
nvidia-smi --query-gpu=driver_version --format=csv,noheader
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Asset roots

`PROTO_HOME` alone is not enough. Weights resolve
`PROTO_ALPHAFOLD3_WEIGHTS_DIR` → `PROTO_MODEL_CACHE` → `PROTO_HOME/…`, and
databases resolve `PROTO_DATABASES_DIR` → `PROTO_MODEL_CACHE/databases` →
`PROTO_HOME/…`. Each override outranks `PROTO_HOME`, shell profiles commonly
export them, and SLURM's `--export=ALL` carries them into a job — so a run can
point at the wrong asset root even with `PROTO_HOME` set correctly. Derive
them together:

```bash
export PROTO_HOME=/path/to/proto_home
export PROTO_MODEL_CACHE=$PROTO_HOME/proto_model_cache
export PROTO_DATABASES_DIR=$PROTO_MODEL_CACHE/databases
export PROTO_ALPHAFOLD3_WEIGHTS_DIR=$PROTO_MODEL_CACHE/alphafold3
```

Use a `PROTO_HOME` you own. proto-tools rebuilds a tool env whenever its setup
files change, so sharing one with other work can delete an env that work
depends on. To share the large read-mostly weights without sharing tool envs,
symlink just the cache:

```bash
mkdir -p $PROTO_HOME/proto_tool_envs
ln -s /shared/proto_model_cache $PROTO_HOME/proto_model_cache
```

proto-tools provisions Prodigal, segmasker, MAFFT, MMseqs2, Foldseek and
AlphaFold 3 into isolated envs on first use, so there are no `*_path` keys in
the configs. First run pays that build cost; later runs do not.

AlphaFold 3 weights are **licensed for non-commercial use by non-commercial
organisations and may not be redistributed** — check the terms before use.

### 4. Optional: AcrNET's PSSM

AcrNET runs without PSI-BLAST, but a zeroed PSSM inflates its score and puts
the protein in a weaker calibration regime (see
[docs/ACR_PIPELINE.md](docs/ACR_PIPELINE.md)), so supply it if you can.

```bash
scripts/build_blastdb.sh        # no cluster needed; CPU-only, ~16 GB, a few hours
```

Then set `acrnet_psiblast` and `acrnet_blast_db` in the config.
It picks up `SLURM_CPUS_PER_TASK` when run under a scheduler.

De novo generated ORFs often have no UniRef30 homologs and land in `no_pssm`
even with the database present — that is expected, not a misconfiguration.

### 5. Verify

```bash
python proto_pipelines/tests/test_parity.py      # 18 checks, expect 0 failures
```

The parity suite is the install check: it exercises config parsing, every
filter's accounting, the Acr callers against their calibration, the shipped
model feature order, and the custom-checkpoint guards. It needs no GPU and
no AF3 weights.

### Working from a checkout

If several proto-language checkouts exist on one machine, or a stale editable
install points at the wrong one, `bootstrap.py` takes an override:

```bash
export PROTO_LANGUAGE_ROOT=/path/to/proto-language
```

Leave it unset to use the installed package. Note that a personal branch can
be internally inconsistent in ways a matched pair of `main`s is not — a
checkout whose `proto_language` expects a `proto_tools` module its own
submodule lacks will fail at import.

### Known proto-tools gaps

These scripts work around two things absent from `proto-tools` `main`:

* **Foldseek TM-scores.** `FoldseekHit` parses a plain 12-column M8 row and
  carries no TM field. The Acr caller's model feature is a TM-score, so
  `acr.py` invokes the provisioned `foldseek` binary directly with
  `--format-output …,qtmscore,alntmscore`. Ranking on E-value instead is not
  viable: as a feature it is worth nothing (CV AUROC 0.9115 vs 0.9106 for
  dropping the structural term entirely, against 0.9237 with a real TM).
* **Evo1 output shape.** proto-tools moved per-sequence scores from
  `Evo1SampleOutput.scores` to `.results[i].metrics`. `runner._evo_score`
  reads either, so both shapes work.

### Repository size

A clone is ~17 MB. `outputs/` is gitignored; the fitted models the
pipelines load are tracked, in `data/models/`.

Profile HMMs (`*.hmm`) and model checkpoints (`*.ckpt`, `*.pt`, `*.pth`) are
stored in **Git LFS** — `data/models/ta_families.hmm` is 12 MB on its
own and is needed at run time. Git keeps a ~130-byte pointer per file
instead of the blob, so history stays small even as these are regenerated.

```bash
git lfs install      # once per machine
git lfs pull         # once per clone, if files came down as pointers
```

**If you skip this, failures are confusing rather than obvious.** The working
tree gets pointer text where the data should be, so pyhmmer reports a
malformed profile and torch a corrupt checkpoint — a parse error, not a
missing file. `git lfs env` tells you whether LFS is active; a `.hmm` of
about 130 bytes tells you it is not.

Patterns rather than paths are tracked, so an asset regenerated under a new
name is captured automatically instead of silently landing in git proper.

## Running the workflows

```bash
python -m proto_pipelines.pipelines.t2ta_sample      --config proto_pipelines/configs/t2ta_sample.yaml
python -m proto_pipelines.pipelines.acr_sample       --config proto_pipelines/configs/acr_sample.yaml
python -m proto_pipelines.pipelines.gene_completion  --config proto_pipelines/configs/gene_completion.yaml
python -m proto_pipelines.pipelines.operon_completion --config proto_pipelines/configs/operon_completion.yaml
```

`t2ta_sample` cofolds its own candidate pairs, so it is a single command; set
`run_cofold: false` in the config to stop after the monomer screen. The
completion workflows are evaluations and run no structure prediction at all —
the published versions never used ESMFold either.

Run the CPU parity checks with:

```bash
python proto_pipelines/tests/test_parity.py
```

## Cost

AlphaFold 3 is far slower than ESMFold, and the screens are where the budget
goes. Two levers:

* Monomer triage runs **single-sequence** (`af3_use_msa: false`) and caps folds
  per generation (`af3_max_proteins_per_proposal`). Lower pLDDT than MSA mode
  is expected; this is triage, and the cutoffs must be set against what this
  mode actually produces.
* The complex screen runs **with a taxonomy-paired MSA**, because that is where
  the interface estimate comes from. That needs the `uniref30-2302` MMseqs2
  database provisioned locally (~365 GB); set `af3_use_msa: false` in the
  cofold config to run single-sequence instead, at a real cost to interface
  confidence. `max_pairs` caps how many pairs are
  folded; with `max_pairs: 0` an unbounded pair list is easy to under-budget,
  since pair count grows quadratically in proteins per generation.

Filters are ordered cheapest-first and the optimizer short-circuits, so a
generation that fails QC never reaches AlphaFold 3. `filter_summary.csv` shows
how many proposals each filter saw.

`run_prompts` holds a single `ToolPool` open across every prompt. Without it,
each per-prompt `Program.run()` opens and closes its own pool, tearing down the
persistent tool workers and reloading Evo's 7B weights onto the GPU once per
prompt — which dominates wall time at paper scale.

## Citation

```
@article{merchant2025semantic,
    author = {Merchant, Aditi T and King, Samuel H and Nguyen, Eric and Hie, Brian L},
    title = {Semantic design of functional de novo genes from a genomic language model},
    year = {2025},
    doi = {10.1038/s41586-025-09749-7},
    URL = {https://www.nature.com/articles/s41586-025-09749-7},
    journal = {Nature}
}
```

AlphaFold 3: Abramson et al., *Nature* 630, 493–500 (2024).
pDockQ: Bryant, Pozzati & Elofsson, *Nat Commun* 13, 1265 (2022).
pDockQ2: Zhu, Shenoy, Kundrotas & Elofsson, *Bioinformatics* 39, btad424 (2023).
