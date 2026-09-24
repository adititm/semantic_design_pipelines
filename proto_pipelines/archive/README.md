# Archive

Nothing here is imported or executed by the pipelines. It is kept so results
stay traceable to the code that produced them.

| path | what it is |
| --- | --- |
| `OPTIMISATION_NOTES.md` | Findings from the Sept 2026 round: the discarded AcrNET calibration, the PSSM confound, the padding investigation, and the clean-environment defects. Current behaviour is documented in `../docs/`, not here. |
| `data/acr_ref_db_189chains/` | The original Foldseek reference database, 189 PDB Acr chains. Superseded by `calibration/data/acr/ref_db_ext` (220 chains: the same 189 plus 31 AlphaFold 3 models), which is what the configs point at. |
| `slurm/` | Per-pipeline Slurm templates, each of which was one pipeline plus one config — now two arguments to `slurm/run_pipeline.sbatch`. Also the one-off jobs that built calibration assets — fold arrays, per-caller scoring runs, the DefenseFinder screen, an MSA-device experiment. They reference result paths that are gitignored, so they will not run as-is against a fresh clone; they record *how* an asset was produced. |
| `scripts/mmseqs_pssm.py` | MMseqs2-derived PSSMs for AcrNET. **Rejected**: scored worse than zeroing the PSSM block entirely (0.80), because the profile differs from PSI-BLAST in content and not merely in scale. Kept so the approach is not retried blind. |

To regenerate a calibration asset, prefer the maintained scripts in
`../calibration/` — `slurm/acrnet_rescore.sbatch` is the worked example of
driving one.
