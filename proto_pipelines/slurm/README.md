# SLURM templates

**These are examples, not requirements.** Nothing in the pipelines needs
SLURM — `scripts/run_pipeline.sh` runs any workflow on a workstation or
against connected compute. Use these if you have a Slurm cluster and want
the resource requests pre-filled.

## Overriding the cluster-specific parts

The `#SBATCH` directives in these files were written for one cluster
(partitions named `gpu` / `gpu_high_mem` / `preemptible`, and an
`--exclude` for a node with small GPUs). **`sbatch` command-line flags
override in-file directives**, so you do not need to edit them:

```bash
sbatch --partition=your_gpu_partition --exclude= --time=8:00:00 \
       --export=ALL,PROTO_HOME=$HOME/proto_home,PYTHON=$(which python) \
       proto_pipelines/slurm/acr_e2e.sbatch
```

Set `--exclude=` (empty) to clear the node exclusion. Partition names are
the one thing that is never portable — check `sinfo -s`.

## What each template does

| template | runs |
| --- | --- |
| `acr_e2e.sbatch` | Anti-CRISPR pipeline, parity checks first |
| `t2ta_e2e.sbatch` | Toxin–antitoxin pipeline, parity checks first |
| `t2ta_evo1.sbatch` / `t2ta_evo2.sbatch` | The same, pinned to one generator family |
| `t2ta_hmm_gate.sbatch` | TA run with the profile-HMM gate enabled |
| `completion_only.sbatch` | Gene + operon completion. No folding, so no AF3 weights or MSA database needed |
| `smoke_test.sbatch` | Fast end-to-end check of the TA path |
| `acrnet_rescore.sbatch` | Regenerate an AcrNET calibration table (worked example of driving `calibration/`) |
| `build_blastdb.sbatch` | Build the UniRef30 BLAST database for AcrNET's PSSM |

One-off jobs that produced the shipped calibration assets are in
`../archive/slurm/`.

## Required environment

Every template needs `PROTO_HOME` and `PYTHON` passed through
`--export=ALL,...`; the folding ones also check for AlphaFold 3 weights and
refuse to start without them, rather than failing an hour in.
