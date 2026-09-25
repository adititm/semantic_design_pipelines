"""Per-prompt Program construction and result collection.

``Evo1Generator`` requires every prompt in a config to be the same length and
to number either one or exactly ``len(segment.proposal_sequences)``. The
original pipelines worked around the first constraint by grouping prompts by
length; this implementation instead runs **one Program per prompt**, which is always
valid, keeps result attribution trivial, and costs nothing because Evo batches
internally via ``batch_size``. All prompts run in a single process so the model
stays resident.

Each ``Program.run()`` would otherwise open and close its own ``ToolPool``,
tearing down the persistent tool workers between prompts and reloading Evo's
7B weights onto the GPU every time. ``run_prompts`` therefore holds one pool
open across the whole sweep; ``Program._enter_compute`` is a no-op while a
pool is already active, so each per-prompt Program reuses it and the model
loads once.

A ``RejectionSamplingOptimizer`` only admits proposals that clear every filter,
so ``segment.result_sequences`` is the survivor set. Rejected proposals are
recovered from the optimizer's proposal history (``track_proposals=True``,
``tracking_interval=1``), which records the rejecting filter and every
constraint metadata dict gathered before the proposal was dropped. A run that
filters everything out should be readable, not silent.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from proto_language.core import Constraint, Construct, Program, Segment
from proto_language.generator import (
    Evo1Generator,
    Evo1GeneratorConfig,
    Evo2Generator,
    Evo2GeneratorConfig,
)
from proto_language.optimizer import (
    RejectionSamplingOptimizer,
    RejectionSamplingOptimizerConfig,
)

from proto_pipelines.prompts import Prompt

logger = logging.getLogger(__name__)

ConstraintBuilder = Callable[[Segment, Prompt], list[Constraint]]


@dataclass
class GenerationSettings:
    """Evo sampling settings shared by every pipeline.

    Attributes:
        generator: Which Evo family to sample from -- ``"evo2"`` (default) or
            ``"evo1"``. The checkpoint must belong to the chosen family.
        model_checkpoint: Checkpoint name. Defaults to ``evo2_7b``. The paper
            used Evo 1.5 (``evo-1.5-8k-base`` with ``generator: evo1``), so
            reproducing its numbers requires switching both.
        model_local_path: Directory of local Evo 2 weights, replacing the
            HuggingFace download. Use this to sample from a fine-tuned or
            modified checkpoint. ``model_checkpoint`` still selects the
            *architecture* the weights are loaded into, so it must name the
            variant the checkpoint was derived from. Evo 2 only -- Evo 1 has
            no such hook, and setting both raises.
        n_tokens: New tokens to generate per sample.
        temperature: Sampling temperature.
        top_k: Top-k sampling cutoff.
        n_sample_per_prompt: Samples drawn per prompt.
        batch_size: Sequences Evo folds into one GPU batch.
        prepend_prompt: Emit prompt + generation as the segment sequence. The
            completion pipelines need this (they call ORFs across the join);
            the TA and Acr pipelines do not, matching ``make_fasta`` versus
            ``make_gene_completion_fasta`` in the published workflow.
        device: Device string for Evo.
        seed: Optimizer seed; also seeds Evo sampling. ``None`` leaves it
            unseeded.
        verbose: Forward Evo and optimizer progress output.
    """

    generator: str = "evo2"
    model_checkpoint: str = "evo2_7b"
    model_local_path: str = ""
    n_tokens: int = 1000
    temperature: float = 0.8
    top_k: int = 4
    n_sample_per_prompt: int = 5
    batch_size: int = 10
    prepend_prompt: bool = False
    device: str = "cuda"
    seed: int | None = None
    verbose: bool = False


@dataclass
class ProposalRecord:
    """One Evo proposal and what happened to it.

    Attributes:
        prompt_id: ``Prompt.prompt_id`` of the prompt that produced it.
        proposal_index: Index within this prompt's run.
        outcome: ``"accepted"``, or the label of the filter that rejected it.
        dna: The generated DNA (prompt included when ``prepend_prompt``).
        energy: Aggregated constraint energy, or ``None`` when the proposal
            was rejected before scoring completed.
        constraint_data: ``{constraint label: metadata dict}`` for every
            constraint that ran before the proposal was dropped.
    """

    prompt_id: str
    proposal_index: int
    outcome: str
    dna: str
    energy: float | None = None
    constraint_data: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        """Whether this proposal cleared every filter."""
        return self.outcome == "accepted"

    def data(self, constraint_label: str) -> dict[str, Any]:
        """Return one constraint's metadata, or an empty dict if it never ran."""
        return self.constraint_data.get(constraint_label, {})


def _build_generator(prompt: Prompt, settings: GenerationSettings) -> Any:
    """Construct the Evo generator named by ``settings.generator``.

    Evo 1 and Evo 2 are separate proto generators with different config
    shapes, so the family is a config choice rather than only a checkpoint
    name. Both honour ``prepend_prompt`` the same way -- the generator derives
    ``max_new_tokens`` from the segment length, subtracting the prompt only
    when it is going to prepend it -- so the flag must be passed to whichever
    family is selected. Stitching the prompt on afterwards instead would make
    the generator emit ``len(prompt)`` extra novel tokens and return a
    sequence one prompt longer than the segment.

    Args:
        prompt: The prompt to sample from.
        settings: Sampling settings, including which family to use.

    Returns:
        An assigned-ready generator instance.

    Raises:
        ValueError: If ``settings.generator`` is not "evo1" or "evo2", or if
            ``model_local_path`` is set for a non-Evo-2 generator.
        NotADirectoryError: If ``model_local_path`` is not a directory.
    """
    family = settings.generator.lower()
    if settings.model_local_path and family != "evo2":
        raise ValueError(
            f"model_local_path is set but generator is {settings.generator!r}. "
            "Only Evo 2 can load local weights -- Evo1GeneratorConfig has no "
            "local_path field, so the path would be silently ignored and you "
            "would sample from the stock checkpoint instead."
        )
    if settings.model_local_path:
        weights = Path(settings.model_local_path)
        if not weights.is_dir():
            raise NotADirectoryError(
                f"model_local_path {weights} is not a directory. Evo 2 expects "
                "a weights *directory*, not a checkpoint file."
            )
    if family == "evo1":
        return Evo1Generator(
            Evo1GeneratorConfig(
                prompts=[prompt.sequence],
                model_checkpoint=settings.model_checkpoint,
                top_k=settings.top_k,
                temperature=settings.temperature,
                prepend_prompt=settings.prepend_prompt,
                batch_size=settings.batch_size,
                device=settings.device,
                verbose=settings.verbose,
            )
        )
    if family == "evo2":
        return Evo2Generator(
            Evo2GeneratorConfig(
                prompts=[prompt.sequence],
                model_checkpoint=settings.model_checkpoint,
                # None, not "", or proto-tools treats the empty string as a
                # path and fails inside the tool env rather than here.
                local_path=settings.model_local_path or None,
                top_k=settings.top_k,
                temperature=settings.temperature,
                prepend_prompt=settings.prepend_prompt,
                batch_size=settings.batch_size,
                device=settings.device,
                verbose=settings.verbose,
            )
        )
    raise ValueError(f"generator must be 'evo1' or 'evo2', got {settings.generator!r}")


def _records_from_history(optimizer: Any, prompt_id: str) -> list[ProposalRecord]:
    """Turn the optimizer's proposal history into ``ProposalRecord`` objects.

    With ``track_proposals=True`` and ``tracking_interval=1`` the optimizer
    appends one history entry per proposal, carrying the proposal's sequence,
    accept/reject outcome, and the metadata of every constraint that ran
    before it was dropped.

    Args:
        optimizer: The optimizer that has finished running.
        prompt_id: Prompt identifier stamped onto each record.

    Returns:
        One record per proposal, ordered by proposal index.
    """
    records: dict[int, ProposalRecord] = {}
    for entry in optimizer.history:
        for proposal in entry.get("proposal_results") or []:
            index = int(proposal.get("proposal_idx", len(records)))
            constructs = proposal.get("constructs") or []
            segments = constructs[0].get("segments") if constructs else None
            segment_data = segments[0] if segments else {}
            constraints = segment_data.get("constraints") or {}
            records[index] = ProposalRecord(
                prompt_id=prompt_id,
                proposal_index=index,
                outcome=(
                    "accepted"
                    if proposal.get("accepted")
                    else str(proposal.get("rejected_by") or "rejected")
                ),
                dna=segment_data.get("sequence") or "",
                energy=proposal.get("energy_score"),
                constraint_data={
                    label: (entry_data or {}).get("data", {})
                    for label, entry_data in constraints.items()
                },
            )
    return [records[index] for index in sorted(records)]


def run_prompt(
    prompt: Prompt,
    settings: GenerationSettings,
    build_constraints: ConstraintBuilder,
    segment_label: str = "generation",
) -> list[ProposalRecord]:
    """Generate and screen ``n_sample_per_prompt`` sequences for one prompt.

    Args:
        prompt: The prompt to sample from.
        settings: Evo sampling settings.
        build_constraints: Callback receiving the newly created segment and
            the prompt, and returning the ordered filter chain. Order matters:
            the optimizer evaluates filters in sequence and skips the rest
            once one rejects, so cheap checks must come before AlphaFold 3.
            The prompt is passed because some filters are prompt-specific (the
            completion pipelines require each ORF to span the prompt).
        segment_label: Label recorded on the segment.

    Returns:
        One ``ProposalRecord`` per proposal, accepted and rejected alike, in
        proposal order.
    """
    segment_length = (
        len(prompt.sequence) + settings.n_tokens
        if settings.prepend_prompt
        else settings.n_tokens
    )
    segment = Segment(length=segment_length, sequence_type="dna", label=segment_label)
    construct = Construct([segment], label=f"{segment_label}_construct")

    generator = _build_generator(prompt, settings)
    generator.assign(segment)

    constraints = build_constraints(segment, prompt)
    if any(constraint.threshold is None for constraint in constraints):
        raise ValueError(
            "run_prompt expects every constraint to carry a threshold so the "
            "optimizer treats it as an ordered filter and short-circuits."
        )

    optimizer = RejectionSamplingOptimizer(
        constructs=[construct],
        generators=[generator],
        constraints=constraints,
        config=RejectionSamplingOptimizerConfig(
            num_samples=settings.n_sample_per_prompt,
            num_results=settings.n_sample_per_prompt,
            seed=settings.seed,
            verbose=settings.verbose,
            track_proposals=True,
            tracking_interval=1,
        ),
    )

    program = Program(
        optimizers=[optimizer],
        num_results=settings.n_sample_per_prompt,
        verbose=settings.verbose,
    )
    program.run()

    records = _records_from_history(optimizer, prompt.prompt_id)
    if not records:
        print(
            f"WARNING: prompt {prompt.prompt_id}: the optimizer recorded no proposal "
            f"history; nothing to report for this prompt."
        )
    return records


def _shared_compute() -> AbstractContextManager[Any]:
    """Return one compute context for the whole prompt sweep.

    Mirrors ``Program._resolve_compute``: a local ``ToolPool`` when tools run
    locally, and nothing when they do not. Holding a single pool open across
    every prompt keeps the persistent tool workers alive, so Evo's weights are
    loaded once instead of once per prompt.

    Returns:
        A ``ToolPool`` to enter, or ``nullcontext()`` when a pool is already
        active (the caller supplied one) or a remote dispatch backend is
        configured, since pools cannot nest and remote routing needs none.
    """
    from proto_tools.tools.tool_registry import ToolRegistry
    from proto_tools.utils.tool_pool import ToolPool, get_active_pool

    if get_active_pool() is not None:
        return nullcontext()
    if ToolRegistry.dispatch_backend_configured():
        logger.debug("Remote dispatch backend active; not opening a local ToolPool.")
        return nullcontext()
    return ToolPool()


def _append_checkpoint(path: Path | None, records: list[ProposalRecord]) -> None:
    """Append one prompt's proposals to the checkpoint file as JSON lines.

    JSONL because it is append-only: a kill mid-write costs the last line, not
    the file. Checkpointing must never take the run down with it, so a failure
    here is reported and the sweep continues -- the records are still in
    memory and will be written normally if the run finishes.

    Args:
        path: Destination JSONL, or ``None`` to disable checkpointing.
        records: The proposals produced by the prompt that just finished.
    """
    if path is None or not records:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "prompt_id": record.prompt_id,
                            "proposal_index": record.proposal_index,
                            "outcome": record.outcome,
                            "accepted": record.accepted,
                            "energy": record.energy,
                            "dna": record.dna,
                            "constraints": record.constraint_data,
                        },
                        default=str,
                    )
                    + "\n"
                )
    except OSError as error:
        print(f"WARNING: could not append to checkpoint {path}: {error}")


def run_prompts(
    prompts: list[Prompt],
    settings: GenerationSettings,
    build_constraints: ConstraintBuilder,
    segment_label: str = "generation",
    checkpoint_path: Path | None = None,
) -> list[ProposalRecord]:
    """Run :func:`run_prompt` over every prompt and concatenate the records.

    Args:
        prompts: Prompts to sample from, in file order.
        settings: Evo sampling settings.
        build_constraints: Filter-chain factory, called once per prompt.
        segment_label: Label recorded on each segment.
        checkpoint_path: JSONL file appended after each prompt. Every result
            table is written only once the whole sweep returns, so without
            this a job killed part-way -- a timeout, or a preemption on a
            requeue partition -- loses every proposal it had already screened.
            With it, at most the prompt in flight is lost.

    Returns:
        Every proposal from every prompt, in prompt order.
    """
    records: list[ProposalRecord] = []
    with _shared_compute():
        for position, prompt in enumerate(prompts, start=1):
            # Printed rather than logged: something in the tool stack stops
            # INFO records reaching the handler after the first Program.run(),
            # which silently hid every prompt after the first. A sweep over
            # dozens of prompts is long enough that losing progress output
            # matters, so this line does not depend on logging configuration.
            print(
                f"Prompt {position}/{len(prompts)} ({prompt.prompt_id}): "
                f"sampling {settings.n_sample_per_prompt} sequence(s)",
                flush=True,
            )
            fresh = run_prompt(prompt, settings, build_constraints, segment_label)
            records.extend(fresh)
            _append_checkpoint(checkpoint_path, fresh)

    accepted = sum(1 for record in records if record.accepted)
    print(
        f"Screened {len(records)} proposal(s); {accepted} passed every filter",
        flush=True,
    )
    if records and accepted == 0:
        print(
            "WARNING: every proposal was rejected. See the filter diagnostics "
            "table for the stage that removed them."
        )
    return records
