# semantic_design_watermarking

Semantic-design pipelines for evaluating **custom Evo 2 checkpoints** —
watermarked, fine-tuned or ablated — against the same filter stack used for
the released weights.

The pipelines themselves are generator-agnostic: generation is one
replaceable stage, and everything after it (ORF calling, protein QC,
profile-HMM screening, AlphaFold 3 structure prediction, interface scoring
and the five-caller anti-CRISPR evidence stack) is unchanged. That is what
makes a controlled comparison possible — run the same prompts through a
custom checkpoint and through the stock model, and every downstream number
is measured the same way.

```yaml
generator: evo2
model_name: evo2_7b                  # architecture the weights load into
model_local_path: /path/to/weights   # a directory; empty = released weights
```

## Start here

* **[proto_pipelines/README.md](proto_pipelines/README.md)** — quickstart,
  setup, the four workflows, and the custom-checkpoint hook in detail.
* **[proto_pipelines/docs/ACR_PIPELINE.md](proto_pipelines/docs/ACR_PIPELINE.md)**
  — the anti-CRISPR caller stack.

```bash
git lfs install && git lfs pull
pip install "git+https://github.com/evo-design/proto-language.git"
pip install -e "./proto_pipelines[acr]"
python proto_pipelines/tests/test_parity.py      # 18 checks, no GPU needed

export PROTO_HOME=/path/you/own/proto_home
proto_pipelines/scripts/run_pipeline.sh acr_sample \
    proto_pipelines/configs/smoke/acr_sample_smoke.yaml
```

`run_pipeline.sh` works on a workstation, under Slurm, or against connected
compute — the only difference is the `device` key in the config. **No GPU?**
Set `device: modal` (or `proto`) and no local accelerator, AlphaFold 3
weights or MSA database is needed. Slurm is optional throughout; nothing in
the pipelines requires it.

## Scope

This repository carries **only what runs a sampling pass and produces
results**. The workflows that fit the shipped models -- labelled sequence
sets, per-caller scoring, AUROC analysis and the provenance archive -- are
deliberately absent; the fitted artefacts they produce are in
`proto_pipelines/data/models/` and are loaded directly. Re-deriving them
belongs in the research tree, not here.


The upstream work this builds on is [*Semantic design of functional de novo
genes from a genomic language
model*](https://www.nature.com/articles/s41586-025-09749-7). Watermark
detection itself is **not** implemented here — this repository provides the
generation and evaluation substrate a detector would be measured against.

Candidates these pipelines emit are structure- and similarity-screened, not
validated. A high score means "worth testing", never "is functional".
