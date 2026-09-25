"""YAML configuration loading.

Config keys are validated against each pipeline's known set, so a typo fails
loudly instead of being silently ignored. A few keys from earlier versions of
these workflows are rejected with an explanation rather than quietly
reinterpreted -- most importantly the ESMFold-era ``plddt_threshold``, whose
0-1 scale is not comparable to AlphaFold 3's 0-100.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from proto_pipelines.af3 import AlphaFold3RunConfig
from proto_pipelines.runner import GenerationSettings

RETIRED_KEYS = {
    "segmasker_path": "proto-tools provisions segmasker; remove this key.",
    "mafft_path": "proto-tools provisions MAFFT; remove this key.",
    "trf_path": "Tandem Repeat Finder has no proto-tools wrapper; see README.",
    "rc_truth": "Prodigal meta mode calls genes on both strands; remove this key.",
    "return_both": "Prodigal meta mode calls genes on both strands; remove this key.",
    "run_esm_fold": "ESMFold is replaced by AlphaFold 3; use 'run_af3'.",
    "run_esmfold": "ESMFold is replaced by AlphaFold 3; use 'run_af3'.",
    "batched": "Evo1Generator batches internally; use 'batch_size'.",
    "ptm_threshold": "Renamed to 'af3_ptm_threshold' to mark the predictor change.",
    "plddt_threshold": (
        "Renamed to 'af3_plddt_threshold' AND rescaled: AlphaFold 3 reports "
        "pLDDT on 0-100, not the 0-1 scale the ESMFold configs used. The old "
        "value is not a valid AlphaFold 3 cutoff."
    ),
    "pdockq_threshold": (
        "Renamed to 'pdockq2_threshold'. The gate is pDockQ2 (PAE-based, "
        "applicable to AlphaFold 3); pDockQ v1 is reported but not used."
    ),
}


def load_yaml(path: str | Path, *, allowed_keys: set[str]) -> dict[str, Any]:
    """Load a pipeline config and reject unknown or retired keys.

    Args:
        path: YAML file containing a top-level mapping.
        allowed_keys: Keys this pipeline understands.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file is not a mapping, or contains a retired or
            unrecognised key.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    retired = sorted(set(data) & set(RETIRED_KEYS))
    if retired:
        details = "\n".join(f"  {key}: {RETIRED_KEYS[key]}" for key in retired)
        raise ValueError(f"{path} uses keys that are no longer supported:\n{details}")

    unknown = sorted(set(data) - allowed_keys)
    if unknown:
        raise ValueError(
            f"{path} has unrecognised key(s) {unknown}; allowed: {sorted(allowed_keys)}"
        )
    return data


#: Env var overriding where every run writes, without editing any config.
OUTPUT_ROOT_ENV = "PROTO_PIPELINES_OUTPUT_ROOT"


def resolve_output_dir(data: dict[str, Any], *, cli_root: str | None = None) -> Path:
    """Resolve a run's output directory under a configurable root.

    ``output_dir`` in a config is a *name* (``t2ta_sample``), not a path, so
    the same config works wherever the repo is checked out and whatever the
    working directory is. It is joined onto the first root that is set:

    1. ``--output-root`` on the command line,
    2. the ``PROTO_PIPELINES_OUTPUT_ROOT`` environment variable,
    3. ``output_root`` in the config,
    4. ``<repo>/outputs`` -- resolved from this file's location, not the CWD.

    An absolute ``output_dir`` bypasses all of it and is used as given.

    Args:
        data: Parsed config.
        cli_root: Value of ``--output-root``, when the caller passed one.

    Returns:
        The directory the run should write to. Not created here.
    """
    name = Path(str(data.get("output_dir", "run")))
    if name.is_absolute():
        return name
    root = (
        cli_root
        or os.environ.get(OUTPUT_ROOT_ENV)
        or data.get("output_root")
        # proto_pipelines/config.py -> proto_pipelines -> repo root
        or Path(__file__).resolve().parent.parent / "outputs"
    )
    return Path(root).expanduser() / name


def generation_settings(
    data: dict[str, Any], *, prepend_prompt: bool
) -> GenerationSettings:
    """Build :class:`GenerationSettings` from a parsed config mapping.

    Args:
        data: Parsed config.
        prepend_prompt: Whether the pipeline needs prompt + generation as one
            sequence (the completion pipelines) or the generation alone.

    Returns:
        Populated sampling settings.
    """
    return GenerationSettings(
        generator=data.get("generator", "evo2"),
        model_checkpoint=data.get("model_name", "evo2_7b"),
        model_local_path=str(data.get("model_local_path", "") or ""),
        n_tokens=int(data.get("n_tokens", 1000)),
        temperature=float(data.get("temperature", 0.8)),
        top_k=int(data.get("top_k", 4)),
        n_sample_per_prompt=int(data.get("n_sample_per_prompt", 5)),
        batch_size=int(data.get("batch_size", 10)),
        prepend_prompt=prepend_prompt,
        device=data.get("device", "cuda"),
        seed=data.get("seed"),
        verbose=bool(data.get("verbose", False)),
    )


def af3_run_config(
    data: dict[str, Any], *, output_dir: Path, subdirectory: str, use_msa: bool
) -> AlphaFold3RunConfig:
    """Build :class:`AlphaFold3RunConfig` from a parsed config mapping.

    Args:
        data: Parsed config.
        output_dir: The pipeline's output directory.
        subdirectory: Folder under ``output_dir`` for AlphaFold 3 results.
        use_msa: Default MSA setting for this stage; overridden by the
            config's ``af3_use_msa`` when present.

    Returns:
        Populated AlphaFold 3 execution settings.
    """
    return AlphaFold3RunConfig(
        use_msa=bool(data.get("af3_use_msa", use_msa)),
        pair_heterocomplex_msas=bool(data.get("af3_pair_heterocomplex_msas", True)),
        msa_search_mode=data.get("af3_msa_search_mode", "local"),
        # Without this the tool default ("cpu") runs a 365 GB uniref30
        # search on CPU while the GPU idles; measured 543 s vs ~1200 s.
        msa_device=data.get("af3_msa_device", "cuda"),
        num_recycles=int(data.get("af3_num_recycles", 10)),
        num_diffusion_samples=int(data.get("af3_num_diffusion_samples", 5)),
        seeds=list(data.get("af3_seeds", [0])),
        output_dir=(
            str(output_dir / subdirectory)
            if data.get("af3_save_structures", True)
            else None
        ),
        device=data.get("device", "cuda"),
        verbose=bool(data.get("verbose", False)),
    )
