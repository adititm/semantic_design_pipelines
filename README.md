# Semantic design pipelines

A reimplementation of the pipelines from [*Semantic design of functional de
novo genes from a genomic language
model*](https://www.nature.com/articles/s41586-025-09749-7).

| workflow | what it does |
| --- | --- |
| `t2ta_sample` | Type II toxin–antitoxin pairs: generate → QC → TA profile-HMM → monomer fold → pair → cofold + interface scoring |
| `acr_sample` | Anti-CRISPR candidates: generate → QC → sequence prescreen → monomer fold → five-caller evidence → ranked candidates |
| `gene_completion` | How closely a truncated gene is completed, by MAFFT identity to a reference |
| `operon_completion` | Whether the downstream operon genes are produced |

## Two deviations from the published workflow

**PaCRISPR is no longer available.** The paper called anti-CRISPRs with
PaCRISPR, a web server with no offline release that is not part of the
published code, so that step cannot be reproduced as written. It is replaced
by five callers that run locally — a profile HMM over Acr/Aca families,
AcRanker, AcrNET, Foldseek against known Acr chains, and AlphaFold 3 pLDDT.

**Structure prediction uses AlphaFold 3** rather than ESMFold, via
proto-tools' managed environment.

## Custom checkpoints

Any pipeline can sample from local Evo 2 weights instead of the released
ones. Everything downstream is generator-agnostic, so a modified checkpoint
runs through the identical filter stack:

```yaml
generator: evo2
model_name: evo2_7b                  # architecture the weights load into
model_local_path: /path/to/weights   # a directory; empty = released weights
```

## Start here

```bash
git lfs install && git lfs pull
pip install "git+https://github.com/evo-design/proto-language.git"
pip install -e "./proto_pipelines[acr]"
python proto_pipelines/tests/test_parity.py      # 20 checks, no GPU needed

export PROTO_HOME=/path/you/own/proto_home
proto_pipelines/utils/run_pipeline.sh acr_sample \
    proto_pipelines/configs/smoke/acr_sample_smoke.yaml
```

* **[proto_pipelines/README.md](proto_pipelines/README.md)** — setup, the four
  workflows, layout, and cost.
* **[proto_pipelines/docs/SETTINGS.md](proto_pipelines/docs/SETTINGS.md)** —
  every stage, its defaults, and what changing them costs.
* **[proto_pipelines/docs/ACR_PIPELINE.md](proto_pipelines/docs/ACR_PIPELINE.md)**
  — the anti-CRISPR caller stack and how to read its output.

`run_pipeline.sh` works on a workstation, under Slurm, or against connected
compute — the only difference is the `device` key. **No GPU?** Set
`device: modal` (or `proto`) and no local accelerator, AlphaFold 3 weights or
MSA database is needed.
