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
       proto_pipelines/slurm/run_pipeline.sbatch \
       acr_sample proto_pipelines/configs/smoke/acr_sample_smoke.yaml
```

Set `--exclude=` (empty) to clear the node exclusion. Partition names are
the one thing that is never portable — check `sinfo -s`.

## What each template does

Three templates, because everything else was one file per pipeline-config
pair and `scripts/run_pipeline.sh` already covers that.

| template | runs |
| --- | --- |
| `run_pipeline.sbatch` | **Any** pipeline. Takes the pipeline name and config as arguments and wraps `scripts/run_pipeline.sh`. |
| `acrnet_rescore.sbatch` | Regenerate an AcrNET calibration table — the worked example of driving `calibration/`. |
| `build_blastdb.sbatch` | Wrapper around `scripts/build_blastdb.sh` for the UniRef30 BLAST database. |

```bash
sbatch --export=ALL,PROTO_HOME=$HOME/proto_home,PYTHON=$(which python) \
       proto_pipelines/slurm/run_pipeline.sbatch \
       acr_sample proto_pipelines/configs/smoke/acr_sample_smoke.yaml
```

The per-pipeline templates that used to live here are in
`../archive/slurm/`, along with the one-off jobs that produced the shipped
calibration assets. They are superseded, not deleted: each was a pipeline
plus a config, which is now two arguments.

To run every smoke config in one go, on a cluster or not:

```bash
scripts/run_smoke.sh            # all of them
scripts/run_smoke.sh nofold     # skip the two folding pipelines
```

## Required environment

Every template needs `PROTO_HOME` and `PYTHON` passed through
`--export=ALL,...`; the folding ones also check for AlphaFold 3 weights and
refuse to start without them, rather than failing an hour in.
