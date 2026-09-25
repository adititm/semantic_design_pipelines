"""Parity checks against the published semantic-design code.

These run on CPU and do not touch AlphaFold 3 or Evo. They exist to catch the
failure mode that matters most when a filter is re-expressed: one that still
runs but no longer means the same thing. Each check compares this
implementation's output against the published function, and each includes at
least one case this implementation is expected to *reject*, so a check that
can only ever pass would be visible.
"""

from __future__ import annotations

import itertools
import sys
import tempfile
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from proto_pipelines.stages.af3 import pdockq_v1
from proto_pipelines.stages.qc import (
    has_underrepresented_amino_acids,
    is_highly_repetitive,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Reference implementations, copied verbatim from the published workflow
# ---------------------------------------------------------------------------


def original_is_highly_repetitive(seq, min_repeat_length=3, threshold=0.3):
    """Verbatim from published semantic_design.filter_protein_fasta."""
    seq_len = len(seq)
    seq_array = np.array(list(seq))
    for k in range(min_repeat_length, min_repeat_length + 7):
        kmers = np.lib.stride_tricks.sliding_window_view(seq_array, k)
        kmer_strings = ["".join(kmer) for kmer in kmers]
        if (
            kmer_strings
            and max(Counter(kmer_strings).values()) * k > seq_len * threshold
        ):
            return True
    return False


def original_is_underrepresented(seq):
    """Verbatim from published semantic_design.filter_protein_fasta."""
    aa_counts = Counter(seq)
    total_unique = len(aa_counts)
    sorted_counts = sorted(aa_counts.values(), reverse=True)
    num_bottom = max(1, int(0.3 * total_unique))
    return all(count < 2 for count in sorted_counts[-num_bottom:])


def original_pdockq(pdb: str):
    """Verbatim from the published extract_pdockq_scores."""

    def parse_atm_record(line):
        record: dict[str, Any] = defaultdict()
        record["atm_name"] = line[12:16].strip()
        record["res_name"] = line[17:20].strip()
        record["chain"] = line[21]
        record["res_no"] = int(line[22:26])
        record["coords"] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        record["B"] = float(line[60:66])
        return record

    chain_coords: dict[str, list[list[float]]] = defaultdict(list)
    plddt_dict: OrderedDict[str, list[float]] = OrderedDict()
    for line in pdb.splitlines():
        if not line.startswith("ATOM"):
            continue
        rec = parse_atm_record(line)
        if rec["atm_name"] == "CB" or (
            rec["atm_name"] == "CA" and rec["res_name"] == "GLY"
        ):
            chain_coords[rec["chain"]].append(list(rec["coords"]))
            plddt_dict.setdefault(f"{rec['chain']}{rec['res_no']}", []).append(rec["B"])
    plddt = np.array([np.mean(v) for v in plddt_dict.values()])

    chains = list(chain_coords.keys())
    if len(chains) < 2 or plddt.size == 0:
        return 0.0
    coords1 = np.array(chain_coords[chains[0]])
    coords2 = np.array(chain_coords[chains[1]])
    mat = np.append(coords1, coords2, axis=0)
    diffs = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    l1 = len(coords1)
    contacts = np.argwhere(dists[:l1, l1:] <= 8)
    if contacts.size == 0:
        return 0.0
    avg_if_plddt = float(
        np.average(
            np.concatenate(
                [plddt[np.unique(contacts[:, 0])], plddt[np.unique(contacts[:, 1])]]
            )
        )
    )
    x = avg_if_plddt * np.log10(contacts.shape[0] + 1)
    return float(0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Two positives, two negatives for each predicate: a check built only from
# cases it is expected to pass cannot fail informatively.
REPETITIVE_CASES = [
    ("AAAAAAAAAAAAAAAAAAAA", True),
    ("MKTAYIAKQRQISFVKSHFSMKTAYIAKQRQISFVKSHFS", True),
    ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV", False),
    ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQ", False),
]

# Expected values were derived by hand from the predicate's definition
# (the rarest ceil-ish 30% of observed residue types all occurring < 2 times),
# not read off this implementation's own output.
UNDERREPRESENTED_CASES = [
    ("ACDEFGHIKLMNPQRSTVWY", True),  # 20 types, all singletons
    ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRV", True),  # 16 types, 4 rarest singletons
    ("AAAACCCCGGGGTTTT", False),  # 4 types, rarest has count 4
    (  # ubiquitin: 18 types, two of the 5 rarest occur twice
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        False,
    ),
]


def _atom(serial, name, res_name, chain, res_no, xyz, bfactor):
    """Format one PDB ATOM record in fixed-column PDB layout.

    Columns follow the PDB spec exactly (13-16 atom name, 17 altLoc, 18-20
    residue name, 22 chain, 23-26 residue number, 31-54 coordinates, 55-60
    occupancy, 61-66 B-factor) because both this implementation and the published workflow parse
    this record by slicing those columns.
    """
    x, y, z = xyz
    return (
        f"ATOM  {serial:>5} {name:<4}"
        f" {res_name:>3} {chain}{res_no:>4}    "
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.00:>6.2f}{bfactor:>6.2f}"
    )


def make_two_chain_pdb(separation: float, bfactor: float) -> str:
    """Build a two-chain PDB with a controllable inter-chain distance.

    Args:
        separation: Distance in angstroms between the two chains' CB atoms.
            Below 8 the chains form interface contacts; above 8 they do not.
        bfactor: Per-residue pLDDT written into the B-factor column.

    Returns:
        PDB text with 6 CB atoms per chain.
    """
    lines = []
    serial = 1
    for index in range(6):
        lines.append(
            _atom(serial, "CB", "ALA", "A", index + 1, (index * 3.8, 0.0, 0.0), bfactor)
        )
        serial += 1
    for index in range(6):
        lines.append(
            _atom(
                serial,
                "CB",
                "ALA",
                "B",
                index + 1,
                (index * 3.8, separation, 0.0),
                bfactor,
            )
        )
        serial += 1
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_repetitiveness() -> list[str]:
    """The repetitiveness predicate must match the published workflow on every case."""
    failures = []
    for sequence, expected in REPETITIVE_CASES:
        here = is_highly_repetitive(sequence)
        published = original_is_highly_repetitive(sequence)
        if here != published:
            failures.append(
                f"repetitive: here={here} published={published} for {sequence[:20]}"
            )
        if here != expected:
            failures.append(
                f"repetitive: got {here}, expected {expected} for {sequence[:20]}"
            )
    return failures


def check_underrepresented() -> list[str]:
    """The underrepresented-AA predicate must match the published workflow."""
    failures = []
    for sequence, expected in UNDERREPRESENTED_CASES:
        here = has_underrepresented_amino_acids(sequence)
        published = original_is_underrepresented(sequence)
        if here != published:
            failures.append(
                f"underrepresented: here={here} published={published} for {sequence[:20]}"
            )
        if here != expected:
            failures.append(
                f"underrepresented: got {here}, expected {expected} for {sequence[:20]}"
            )
    return failures


def check_pdockq() -> list[str]:
    """The pDockQ v1 must reproduce the published workflow's value bit for bit.

    The negative control is a pair of chains held 20 A apart: no interface
    contacts, so both implementations must return exactly 0.0. Without it a
    scorer that returned a constant would pass the positive case.
    """
    failures = []
    cases = [
        ("contacting", 6.0, 90.0),
        ("separated", 20.0, 90.0),
        ("low-confidence", 6.0, 30.0),
    ]
    for name, separation, bfactor in cases:
        pdb = make_two_chain_pdb(separation, bfactor)
        here = pdockq_v1(pdb)["pdockq_v1"]
        published = original_pdockq(pdb)
        if abs(here - published) > 1e-9:
            failures.append(f"pdockq[{name}]: here={here!r} published={published!r}")
        if name == "separated" and here != 0.0:
            failures.append(
                f"pdockq[separated]: expected 0.0 for a non-interface, got {here!r}"
            )
        if name == "contacting" and here <= 0.0:
            failures.append(
                f"pdockq[contacting]: expected a positive score, got {here!r}"
            )
    return failures


def check_configs() -> list[str]:
    """Every shipped config must parse, and retired keys must be rejected."""
    from proto_pipelines.core.config import load_yaml
    from proto_pipelines.pipelines import (
        acr_sample,
        gene_completion,
        operon_completion,
        t2ta_sample,
    )

    failures = []
    modules = {
        "t2ta_sample": t2ta_sample,
        "acr_sample": acr_sample,
        "gene_completion": gene_completion,
        "operon_completion": operon_completion,
    }
    config_dir = REPO_ROOT / "proto_pipelines" / "configs"
    for name, module in modules.items():
        path = config_dir / f"{name}.yaml"
        try:
            load_yaml(path, allowed_keys=module.ALLOWED_KEYS)
        except Exception as error:
            failures.append(f"config[{name}]: {error}")

    # Negative control: a config using the old ESMFold-scale key must fail.
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write("output_dir: /tmp/x\nplddt_threshold: 0.6\n")
        retired_path = Path(handle.name)
    try:
        load_yaml(retired_path, allowed_keys=t2ta_sample.ALLOWED_KEYS)
        failures.append(
            "config: a retired 'plddt_threshold' key was accepted; it must be rejected"
        )
    except ValueError:
        pass
    finally:
        retired_path.unlink()
    return failures


def check_accepted_proteins() -> list[str]:
    """Survivors must be recoverable whether or not the pipeline folded.

    The completion pipelines have a QC stage but no AlphaFold 3 stage, so a
    collector that reads only the AF3 metadata returns nothing for them and
    their identity analysis silently scores zero sequences. The second case
    below is that control; the first keeps the folding path honest.
    """
    from proto_pipelines.core.reporting import accepted_proteins
    from proto_pipelines.core.runner import ProposalRecord

    protein = {"protein_id": "gene_1", "sequence": "MKTAYIAK", "length": 8}

    with_af3 = ProposalRecord(
        "p0",
        0,
        "accepted",
        "ACGT",
        constraint_data={
            "protein_qc": {"qc_proteins": [protein]},
            # The AF3 screen rejected this one, so it must not be reported even
            # though QC passed it: the last filtering stage decides.
            "af3_monomer_screen": {"af3_proteins": []},
        },
    )
    qc_only = ProposalRecord(
        "p0",
        1,
        "accepted",
        "ACGT",
        constraint_data={"protein_qc": {"qc_proteins": [protein]}},
    )
    rejected = ProposalRecord(
        "p0",
        2,
        "protein_qc",
        "ACGT",
        constraint_data={"protein_qc": {"qc_proteins": [protein]}},
    )

    failures = []
    if accepted_proteins([with_af3]):
        failures.append("accepted_proteins reported a protein the AF3 screen rejected")
    got = accepted_proteins([qc_only])
    if len(got) != 1:
        failures.append(
            f"accepted_proteins returned {len(got)} protein(s) for a QC-only pipeline, want 1 "
            "(the completion pipelines do not fold)"
        )
    elif got[0]["protein_uid"] != "p0_1_gene_1":
        failures.append(f"unexpected protein_uid {got[0]['protein_uid']!r}")
    if accepted_proteins([rejected]):
        failures.append("accepted_proteins reported a protein from a rejected proposal")
    return failures


def check_auroc() -> list[str]:
    """AUROC must be right at the extremes, and must handle ties as 0.5.

    A scorer comparison is only meaningful if the statistic behind it is
    correct. The tie case is the one most often wrong: when a score saturates
    (pDockQ2 pins near zero for poor interfaces), a naive `>` comparison
    silently reports 0.0 instead of 0.5 and makes a useless score look
    perfectly anti-correlated.
    """
    import numpy as np

    # Carried here rather than imported: the suite must stay runnable without
    # the calibration tooling. Tie handling is the point of the check -- a
    # naive (pos > neg).mean() reports 0.0 for an all-tied score instead of
    # 0.5, which makes a useless score look perfectly anti-correlated.
    def auroc(positives, negatives):
        if positives.size == 0 or negatives.size == 0:
            return float("nan")
        comparisons = positives[:, None] - negatives[None, :]
        wins = (comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()
        return float(wins / (positives.size * negatives.size))

    def youden_threshold(positives, negatives):
        best = {
            "threshold": float("nan"),
            "sensitivity": 0.0,
            "specificity": 0.0,
            "youden_j": -1.0,
        }
        if positives.size == 0 or negatives.size == 0:
            return best
        for cutoff in np.unique(np.concatenate([positives, negatives])):
            sensitivity = float((positives >= cutoff).mean())
            specificity = float((negatives < cutoff).mean())
            j = sensitivity + specificity - 1.0
            if j > best["youden_j"]:
                best = {
                    "threshold": float(cutoff),
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "youden_j": j,
                }
        return best

    cases = [
        ("perfect separation", np.array([1.0, 2, 3]), np.array([0.0, 0.1, 0.2]), 1.0),
        ("perfect inversion", np.array([0.0, 0.1]), np.array([1.0, 2]), 0.0),
        ("all tied", np.array([1.0, 1, 1]), np.array([1.0, 1, 1]), 0.5),
        ("half overlap", np.array([1.0, 3]), np.array([0.0, 2]), 0.75),
    ]
    failures = []
    for name, pos, neg, want in cases:
        got = auroc(pos, neg)
        if abs(got - want) > 1e-9:
            failures.append(f"auroc[{name}]: got {got}, want {want}")
    if not np.isnan(auroc(np.array([]), np.array([1.0]))):
        failures.append("auroc with an empty class must be nan, not a number")

    best = youden_threshold(np.array([1.0, 2, 3]), np.array([0.0, 0.1, 0.2]))
    if best["sensitivity"] != 1.0 or best["specificity"] != 1.0:
        failures.append(f"youden_threshold failed on separable data: {best}")
    empty = youden_threshold(np.array([]), np.array([1.0]))
    if not np.isnan(empty["threshold"]):
        failures.append("youden_threshold with an empty class must return nan")
    return failures


def check_reporting() -> list[str]:
    """Filter-summary counts must add up and attribute rejections correctly.

    The diagnostics table is how a user finds out why a screen yielded
    nothing, so a wrong one is worse than none. The fixture below mixes an
    acceptance, a rejection at the first filter (which therefore never reaches
    the second), and a rejection at the second -- the last case is the control
    that a naive "evaluated == total" implementation would get wrong.
    """
    import tempfile

    from proto_pipelines.core.reporting import (
        write_filter_summary,
        write_fold_scores,
        write_proposal_table,
    )
    from proto_pipelines.core.runner import ProposalRecord

    records = [
        ProposalRecord(
            "p0", 0, "accepted", "ACGT", constraint_data={"qc": {}, "af3": {}}
        ),
        ProposalRecord("p0", 1, "qc", "ACGT", constraint_data={"qc": {}}),
        ProposalRecord("p0", 2, "af3", "ACGT", constraint_data={"qc": {}, "af3": {}}),
        # Passed every filter but lost the top-k cut: must not be blamed on a filter.
        ProposalRecord(
            "p0",
            3,
            "did_not_enter_top_k",
            "ACGT",
            constraint_data={"qc": {}, "af3": {}},
        ),
    ]
    # A proposal the AF3 screen rejected: its folds must still be reported, or
    # the "fold first, then choose cutoffs" workflow is impossible.
    fold_records = [
        ProposalRecord(
            "p0",
            0,
            "af3_monomer_screen",
            "ACGT",
            constraint_data={
                "af3_monomer_screen": {
                    "af3_proteins": None,
                    "af3_folded_proteins": [
                        {
                            "protein_id": "g1",
                            "avg_plddt": 81.0,
                            "ptm": 0.6,
                            "passed_af3_screen": True,
                        },
                        {
                            "protein_id": "g2",
                            "avg_plddt": 42.0,
                            "ptm": 0.2,
                            "passed_af3_screen": False,
                        },
                    ],
                }
            },
        )
    ]
    expected = {
        "qc": {"evaluated": 4, "rejected": 1, "passed": 3},
        "af3": {"evaluated": 3, "rejected": 1, "passed": 2},
        "(non-filter) did_not_enter_top_k": {
            "evaluated": 1,
            "rejected": 1,
            "passed": 0,
        },
        "TOTAL": {"evaluated": 4, "rejected": 3, "passed": 1},
    }

    failures = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        summary = write_filter_summary(
            records, ["qc", "af3"], root / "filter_summary.csv"
        )
        for _, row in summary.iterrows():
            want = expected[row["filter"]]
            for column, value in want.items():
                if int(row[column]) != value:
                    failures.append(
                        f"filter_summary[{row['filter']}].{column}: got {row[column]}, want {value}"
                    )

        filter_rows = summary[summary["filter"] != "TOTAL"]
        total_rejected = int(
            summary.loc[summary["filter"] == "TOTAL", "rejected"].iloc[0]
        )
        if int(filter_rows["rejected"].sum()) != total_rejected:
            failures.append(
                f"per-row rejections sum to {int(filter_rows['rejected'].sum())}, "
                f"TOTAL says {total_rejected}"
            )

        table = write_proposal_table(records, root / "generated_sequences.csv")
        if len(table) != len(records):
            failures.append(
                f"proposal table has {len(table)} rows, want {len(records)}"
            )
        if int(table["accepted"].sum()) != 1:
            failures.append(
                f"proposal table marks {int(table['accepted'].sum())} accepted, want 1"
            )
        if not (root / "generated_sequences.csv").exists():
            failures.append("write_proposal_table did not write its CSV")

        folds = write_fold_scores(fold_records, root / "af3_fold_scores.csv")
        if len(folds) != 2:
            failures.append(
                f"fold scores table has {len(folds)} rows, want 2 (pass + fail)"
            )
        elif int(folds["passed_af3_screen"].sum()) != 1:
            failures.append("fold scores table lost the pass/fail distinction")
        elif not (folds["avg_plddt"].iloc[0] >= folds["avg_plddt"].iloc[1]):
            failures.append("fold scores table is not sorted by pLDDT descending")
        empty = write_fold_scores([], root / "empty_folds.csv")
        if not empty.empty or not (root / "empty_folds.csv").exists():
            failures.append(
                "write_fold_scores must still write a file when nothing folded"
            )
    return failures


def check_filter_labels() -> list[str]:
    """Every filter in the chain must own its rejections in filter_summary.

    ``write_filter_summary`` is told the filter order by the pipeline, and a
    filter the pipeline forgets to list is not reported as absent -- its
    rejections are silently re-attributed to a ``(non-filter)`` row, which
    reads as an optimizer eviction rather than a gate that fired. The cofold
    stage was shipped that way once. The control below omits the label on the
    same records and asserts the misreport comes back, so a check that could
    only ever pass would be visible.
    """
    import tempfile

    from proto_pipelines.core.reporting import write_filter_summary
    from proto_pipelines.core.runner import ProposalRecord

    failures: list[str] = []

    def record(index: int, outcome: str, evaluated: list[str]) -> ProposalRecord:
        return ProposalRecord(
            prompt_id="p0",
            proposal_index=index,
            outcome=outcome,
            dna="A",
            constraint_data={label: {} for label in evaluated},
        )

    chain = ["protein_qc", "profile_hmm", "af3_monomer_screen", "ta_cofold"]
    records = [
        record(0, "accepted", chain),
        record(1, "protein_qc", ["protein_qc"]),
        record(2, "ta_cofold", chain),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        summary = write_filter_summary(records, chain, Path(tmp) / "summary.csv")
        row = summary[summary["filter"] == "ta_cofold"]
        if row.empty:
            failures.append("no ta_cofold row in filter_summary")
        elif int(row["rejected"].iloc[0]) != 1:
            failures.append(f"ta_cofold rejected={row['rejected'].iloc[0]}, expected 1")
        if summary["filter"].str.contains("non-filter").any():
            failures.append("a cofold rejection leaked into a (non-filter) row")

        # Control: the same records with the label dropped must misreport.
        control = write_filter_summary(records, chain[:-1], Path(tmp) / "control.csv")
        if not control["filter"].str.contains("non-filter").any():
            failures.append("control did not reproduce the bug; this check cannot fail")

    return failures


def check_generator_prepend_parity() -> list[str]:
    """Swapping the generator family must not change how much is generated.

    Both families derive ``max_new_tokens`` from the segment length and
    subtract the prompt only when they are going to prepend it, so the flag
    has to reach whichever config is built. It once did not reach Evo 2, and
    the prompt was stitched on afterwards instead -- which left Evo 2
    generating ``len(prompt)`` extra novel tokens and returning a sequence one
    prompt longer than the segment, silently, only in gene_completion.

    Builds the generators but never samples, so this stays a CPU check.
    """
    from proto_language.core import Segment

    from proto_pipelines.core.prompts import Prompt
    from proto_pipelines.core.runner import GenerationSettings, _build_generator

    failures: list[str] = []
    prompt = Prompt(index=0, sequence="ATG" * 40)
    prompt_length, n_tokens = len(prompt.sequence), 1000
    segment_length = prompt_length + n_tokens  # what run_prompt builds

    for family, checkpoint in (("evo1", "evo-1.5-8k-base"), ("evo2", "evo2_7b")):
        settings = GenerationSettings(
            generator=family,
            model_checkpoint=checkpoint,
            n_tokens=n_tokens,
            prepend_prompt=True,
        )
        try:
            generator = _build_generator(prompt, settings)
            generator.assign(
                Segment(length=segment_length, sequence_type="dna", label="s")
            )
        except Exception as error:
            failures.append(f"{family}: could not build generator: {error}")
            continue

        if not generator.prepend_prompt:
            failures.append(
                f"{family}: prepend_prompt did not reach the generator config"
            )
            continue
        new_tokens = generator._compute_max_new_tokens(prompt_length, True)
        if new_tokens != n_tokens:
            failures.append(
                f"{family}: generates {new_tokens} novel tokens, expected {n_tokens}"
            )
        if prompt_length + new_tokens != segment_length:
            failures.append(
                f"{family}: final length {prompt_length + new_tokens}, "
                f"segment is {segment_length}"
            )

    return failures


def check_fold_cap_keeps_hmm_hits() -> list[str]:
    """The per-proposal fold cap must not discard the protein that qualified.

    A generation reaches the structure screen because the profile-HMM filter
    recognised something in it, and that something is often the shorter ORF.
    Capping on length alone folded the two longest and dropped the only
    protein with family signal -- observed in a real run, where ``gene_2``
    carried the sole hit and was the one discarded. The controls below cover
    the no-HMM fallback and reproduce the original length-only behaviour.
    """
    from proto_pipelines.stages.af3 import (
        AlphaFold3MonomerScreenConfig,
        _hmm_qualifying_ids,
    )

    class _Sequence:
        def __init__(self, metadata: dict[str, Any]) -> None:
            self._constraints_metadata = metadata

    failures: list[str] = []
    proteins = [
        {"protein_id": "gene_2", "length": 120},
        {"protein_id": "gene_3", "length": 200},
        {"protein_id": "gene_4", "length": 250},
    ]
    metadata = {
        "profile_hmm": {
            "data": {
                "hmm_hits": [
                    {"protein_id": "gene_2", "qualifies": True},
                    {"protein_id": "gene_3", "qualifies": False},
                    {"protein_id": "gene_4", "qualifies": False},
                ]
            }
        }
    }
    config = AlphaFold3MonomerScreenConfig(max_proteins_per_proposal=2)

    def keep(sequence: _Sequence) -> list[str]:
        qualifying = _hmm_qualifying_ids(sequence, config.hmm_constraint_label)
        ordered = sorted(
            proteins,
            key=lambda entry: (entry["protein_id"] not in qualifying, -entry["length"]),
        )
        return [
            entry["protein_id"] for entry in ordered[: config.max_proteins_per_proposal]
        ]

    kept = keep(_Sequence(metadata))
    if "gene_2" not in kept:
        failures.append(f"fold cap dropped the HMM-qualifying protein; kept {kept}")

    # Control: with no HMM stage the ordering must stay longest-first.
    fallback = keep(_Sequence({}))
    if fallback != ["gene_4", "gene_3"]:
        failures.append(f"no-HMM fallback changed; kept {fallback}")

    # Control: the original length-only cap must still reproduce the bug.
    length_only = [
        entry["protein_id"]
        for entry in sorted(proteins, key=lambda e: e["length"], reverse=True)[:2]
    ]
    if "gene_2" in length_only:
        failures.append("control did not reproduce the bug; this check cannot fail")

    return failures


def check_shipped_configs_build_generators() -> list[str]:
    """Every shipped config must construct the generator it names.

    ``generator`` and ``model_name`` are independent keys, so a config can
    validate cleanly and still pair an Evo 2 generator with an Evo 1
    checkpoint -- which is exactly what happened when the default flipped to
    ``evo2`` and the Evo 1.5 configs, having no ``generator`` key, inherited
    it. Loading the config could not catch that; only building the generator
    can. No model is loaded, just the config object.

    The control asserts the mismatch is still rejected, so this cannot pass
    vacuously.
    """
    import glob
    import importlib

    from proto_pipelines.core.config import generation_settings, load_yaml
    from proto_pipelines.core.prompts import Prompt
    from proto_pipelines.core.runner import _build_generator

    failures: list[str] = []
    modules = {
        stem: importlib.import_module(f"proto_pipelines.pipelines.{stem}")
        for stem in (
            "t2ta_sample",
            "acr_sample",
            "gene_completion",
            "operon_completion",
        )
    }
    prompt = Prompt(index=0, sequence="ATG" * 30)

    configs = sorted(glob.glob("proto_pipelines/configs/**/*.yaml", recursive=True))
    if not configs:
        return ["no configs found; check the working directory"]

    for config_path in configs:
        # Match the config to its pipeline by which ALLOWED_KEYS accepts it,
        # rather than by parsing the filename: variant configs are named
        # freely (t2ta_evo2_smoke, t2ta_hmm_gate_smoke) and a name-based table
        # silently stops covering whatever it fails to recognise.
        parsed = None
        for module in modules.values():
            try:
                parsed = load_yaml(config_path, allowed_keys=module.ALLOWED_KEYS)
                break
            except ValueError:
                continue
        if parsed is None:
            failures.append(
                f"{Path(config_path).name}: no pipeline accepts this config's keys"
            )
            continue
        settings = generation_settings(parsed, prepend_prompt=False)
        try:
            _build_generator(prompt, settings)
        except Exception as error:
            failures.append(
                f"{Path(config_path).name}: generator={settings.generator} "
                f"model_name={settings.model_checkpoint} -> {type(error).__name__}"
            )

    # Control: a deliberately mismatched pair must still be rejected.
    from proto_pipelines.core.runner import GenerationSettings

    try:
        _build_generator(
            prompt,
            GenerationSettings(generator="evo2", model_checkpoint="evo-1.5-8k-base"),
        )
        failures.append("control did not reject an evo2/evo1-checkpoint mismatch")
    except Exception:
        pass

    return failures


def check_acr_assets() -> list[str]:
    """The shipped Acr callers must reproduce their calibrated behaviour.

    The combined model is a plain logistic fit whose coefficients were
    calibrated against a specific feature order; if the HMM set, the AcRanker
    feature construction or the model file drift apart the scores silently
    become meaningless rather than erroring. This pins all three.

    Controls: a random-DNA ORF must score below a known Acr, and the AcRanker
    feature vector must match the calibration implementation exactly.
    """
    import json
    from itertools import product as _product

    # AcRanker's published 412-feature construction, carried here rather than
    # imported, so the suite stays a self-contained check of the shipped
    # models. Verbatim from server2.prot_feats_seq: 20 L2-normalised amino
    # acid fractions, then 2-mer and 3-mer counts over a 7-group reduced
    # alphabet, each divided by len(seq)-1 and L2-normalised.
    import numpy as _np
    import numpy as np

    from proto_pipelines.callers.acr import (
        AcrEvidenceConfig,
        _acranker_features,
        score_proteins,
    )

    _AA = "ACDEFGHIKLMNPQRSTVWY"
    _GROUPS = {
        "A": "1",
        "V": "1",
        "G": "1",
        "I": "2",
        "L": "2",
        "F": "2",
        "P": "2",
        "Y": "3",
        "M": "3",
        "T": "3",
        "S": "3",
        "H": "4",
        "N": "4",
        "Q": "4",
        "W": "4",
        "R": "5",
        "K": "5",
        "D": "6",
        "E": "6",
        "C": "7",
    }

    def _l2(vector):
        norm = _np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _kmer_block(sequence, k):
        index = {"".join(p): i for i, p in enumerate(_product("1234567", repeat=k))}
        counts = _np.zeros(7**k)
        for start in range(len(sequence) - k + 1):
            kmer = sequence[start : start + k]
            if any(residue not in _GROUPS for residue in kmer):
                continue
            counts[index["".join(_GROUPS[r] for r in kmer)]] += 1
        return counts / max(len(sequence) - 1, 1)

    def ref_features(sequence):
        length = max(len(sequence), 1)
        composition = _np.array([sequence.count(a) / length for a in _AA])
        return _np.concatenate(
            [
                _l2(composition),
                _l2(_kmer_block(sequence, 2)),
                _l2(_kmer_block(sequence, 3)),
            ]
        )

    failures: list[str] = []
    data = Path("proto_pipelines/data/models/acr")
    model_path = data / "acr_combined_model.json"
    for asset in (
        "acr_families_default.hmm",
        "acranker_booster.json",
        "acr_combined_model.json",
        "acr_divergent_model.json",
    ):
        if not (data / asset).exists():
            failures.append(f"missing shipped Acr asset: {asset}")
    if failures:
        return failures

    # Both tiers are plain logistic fits whose coefficients were calibrated
    # against a specific feature order; a silent reorder makes the scores
    # meaningless rather than erroring.
    # acrnet_pctile, NOT acrnet_score: the raw probability is inflated toward
    # 1.0 whenever PSI-BLAST builds no PSSM (67% of negatives here vs 2% of
    # positives), so a model fit on it partly reads profilability rather than
    # Acr-likeness -- a has_pssm flag alone scored AUROC 0.825.
    expected = {
        "acr_combined_model.json": [
            "hmm_score",
            "acranker_raw",
            "acranker_z",
            "best_tmscore",
            "avg_plddt",
            "acrnet_pctile",
        ],
        "acr_divergent_model.json": ["acranker_raw", "best_tmscore", "acrnet_pctile"],
        "acr_prescreen_model.json": ["acranker_raw", "acrnet_pctile"],
    }
    for name, want in expected.items():
        got = json.loads((data / name).read_text())["features"]
        if got != want:
            failures.append(f"{name} feature order changed: {got}")

    # AcRanker features must match the calibration code exactly, or the
    # booster is being fed a different vector than it was scored with.
    sequence = "MKFIKYLSTAHLNYMNIAVYENGSKIKARVENVVNGKSVGARDFDSTEQLESWFYGLPGSGLG"
    drift = float(
        np.abs(np.array(_acranker_features(sequence)) - ref_features(sequence)).max()
    )
    if drift > 1e-9:
        failures.append(
            f"AcRanker features drifted from calibration: max|diff|={drift:.2e}"
        )

    # Control: a known Acr must outscore a random-DNA ORF of equal length.
    config = AcrEvidenceConfig(
        hmm_path=str(data / "acr_families_default.hmm"),
        acranker_model=str(data / "acranker_booster.json"),
        acranker_shuffles=5,
        combined_model=str(model_path),
        divergent_model=str(data / "acr_divergent_model.json"),
    )
    # A two-sequence fixture, not the calibration set: one known Acr and one
    # random ORF, which is all this control needs.
    table = pd.read_csv(
        Path(__file__).resolve().parent / "fixtures/acr_control_pair.csv"
    )
    acr = table[table.seq_class == "acr"].iloc[0]
    junk = table[table.seq_class == "random_orf"].iloc[0]
    scored = score_proteins(
        [
            {"protein_id": "acr", "sequence": acr.sequence, "avg_plddt": 60.0},
            {"protein_id": "junk", "sequence": junk.sequence, "avg_plddt": 60.0},
        ],
        config,
    )
    by_id = {r["protein_id"]: r for r in scored}
    # Both must route to the divergent tier: neither has a Pfam hit, and the
    # full model would penalise that absence rather than ignore it.
    for name in ("acr", "junk"):
        if by_id[name].get("acr_tier") != "divergent":
            failures.append(
                f"{name} routed to {by_id[name].get('acr_tier')!r}, expected 'divergent'"
            )
    if by_id["acr"]["acr_locus_score"] <= by_id["junk"]["acr_locus_score"]:
        failures.append(
            f"control failed: known Acr scored {by_id['acr']['acr_locus_score']:.3f} "
            f"<= random ORF {by_id['junk']['acr_locus_score']:.3f}"
        )
    return failures


def check_acrnet_batch_of_one() -> list[str]:
    """AcrNET must score a single protein identically to one in a batch.

    AcrNET's own ``forward`` calls an unconditional ``.squeeze()`` and then
    reads ``size(2)``, so a batch of one loses its batch dimension and raises
    IndexError. A proposal often yields exactly one protein, so this is the
    common case in-pipeline rather than an edge case -- it crashed the first
    end-to-end run. The scorer duplicates single-item batches and discards
    the duplicate; this pins both that it works and that it changes nothing.
    """
    from pathlib import Path as _Path

    import numpy as np

    from proto_pipelines.callers.acrnet import score

    checkpoint = _Path("proto_pipelines/data/models/acr/acrnet/model.ckpt")
    if not checkpoint.exists():
        return []  # AcrNET assets are optional

    failures: list[str] = []
    rng = np.random.default_rng(0)
    length = 60
    features = {
        f"p{i}": {
            "ss3": "C" * length,
            "ss8": "L" * length,
            "acc": "E" * length,
            "sequence": "A" * length,
            "pssm": np.zeros(1110, dtype="float32"),
            "embedding": rng.standard_normal(1280).astype("float32"),
        }
        for i in range(3)
    }
    reference: float | None = None
    for size in (1, 2, 3):
        subset = {k: features[k] for k in list(features)[:size]}
        try:
            scored = score(subset, str(checkpoint))
        except Exception as error:
            failures.append(f"batch of {size} raised {type(error).__name__}: {error}")
            continue
        if len(scored) != size:
            failures.append(f"batch of {size} returned {len(scored)} scores")
        if reference is None:
            reference = scored["p0"]
        elif abs(scored["p0"] - reference) > 1e-6:
            failures.append(
                f"batch of {size} changed p0: {scored['p0']:.6f} vs {reference:.6f}"
            )
    return failures


def check_acr_model_features_resolve() -> list[str]:
    """Every model feature must resolve to a field on the evidence record.

    The calibration CSVs and the pipeline records name some columns
    differently (``best_tmscore`` vs ``foldseek_tmscore``). The original
    lookup used ``record.get(name)``, so a mismatch silently contributed 0.0
    and still produced a plausible score -- the Foldseek term was zeroed in
    every pipeline run, costing an 18x difference on a real candidate
    (0.667 -> 0.037). The control below asserts an unresolvable feature now
    raises rather than defaulting.
    """
    import json as _json
    import tempfile as _tempfile

    from proto_pipelines.callers.acr import AcrEvidenceConfig, score_proteins

    data = Path("proto_pipelines/data/models/acr")
    if not (data / "acr_divergent_model.json").exists():
        return []

    failures: list[str] = []
    config = AcrEvidenceConfig(
        hmm_path=str(data / "acr_families_default.hmm"),
        acranker_model=str(data / "acranker_booster.json"),
        acranker_shuffles=3,
        combined_model=str(data / "acr_combined_model.json"),
        divergent_model=str(data / "acr_divergent_model.json"),
    )
    proteins = [{"protein_id": "g1", "sequence": "M" * 60, "avg_plddt": 60.0}]
    record = score_proteins(proteins, config, {"g1": 0.9})[0]
    if record.get("acr_locus_score") is None:
        failures.append("combined model produced no acr_locus_score")
    if record.get("acrnet_score") != 0.9:
        failures.append(
            "acrnet_score must be attached before the model runs; "
            f"got {record.get('acrnet_score')!r}"
        )

    # Control: a feature that matches no record field must raise.
    broken = _json.loads((data / "acr_divergent_model.json").read_text())
    broken["features"] = ["acranker_raw", "no_such_feature", "acrnet_score"]
    with _tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump(broken, fh)
        path = fh.name
    bad = config.model_copy(update={"combined_model": path, "divergent_model": path})
    try:
        score_proteins(proteins, bad, {"g1": 0.9})
        failures.append("an unresolvable model feature was silently accepted")
    except KeyError:
        pass
    return failures


def check_prescreen_fold_gating() -> list[str]:
    """Prescreen gating must skip weak ORFs but never lose an HMM hit.

    Three properties, each with a way to fail: ORFs above the threshold are
    folded, an HMM-qualifying ORF is folded even when its prescreen score is
    far below it (that signal had zero false positives across 191 negatives
    and must not be outranked), and a proposal where nothing clears the bar
    still folds its best ORF -- returning no structures would look like a
    pipeline failure rather than a weak generation.
    """
    from proto_pipelines.stages.af3 import (
        AlphaFold3MonomerScreenConfig,
        _hmm_qualifying_ids,
        _prescreen_scores,
    )

    class _Sequence:
        def __init__(self, metadata: dict[str, Any]) -> None:
            self._constraints_metadata = metadata

    failures: list[str] = []
    scores = [0.85, 0.42, 0.05, 0.02, 0.01]
    evidence = [
        {"protein_id": f"gene_{i}", "acr_locus_score": s}
        for i, s in enumerate(scores, start=1)
    ]
    sequence = _Sequence(
        {
            "acr_prescreen": {"data": {"acr_evidence": evidence}},
            # gene_5 scores 0.01 but is an HMM hit, so it must survive.
            "profile_hmm": {
                "data": {"hmm_hits": [{"protein_id": "gene_5", "qualifies": True}]}
            },
        }
    )
    config = AlphaFold3MonomerScreenConfig(
        prescreen_constraint_label="acr_prescreen", prescreen_min_score=0.20
    )
    proteins = [{"protein_id": f"gene_{i}", "length": 100} for i in range(1, 6)]
    prescore = _prescreen_scores(sequence, config.prescreen_constraint_label)
    qualifying = _hmm_qualifying_ids(sequence, config.hmm_constraint_label)
    kept = {
        p["protein_id"]
        for p in proteins
        if prescore.get(p["protein_id"], 1.0) >= config.prescreen_min_score
        or p["protein_id"] in qualifying
    }
    if kept != {"gene_1", "gene_2", "gene_5"}:
        failures.append(f"prescreen gating kept {sorted(kept)}, expected gene_1/2/5")
    if "gene_5" not in kept:
        failures.append("an HMM-qualifying ORF was dropped by the prescreen")

    # Control: everything below threshold must still yield one fold.
    weak = _Sequence(
        {
            "acr_prescreen": {
                "data": {
                    "acr_evidence": [
                        {"protein_id": f"g{i}", "acr_locus_score": 0.01}
                        for i in range(3)
                    ]
                }
            }
        }
    )
    weak_scores = _prescreen_scores(weak, "acr_prescreen")
    weak_proteins = [{"protein_id": f"g{i}"} for i in range(3)]
    survivors = [
        p for p in weak_proteins if weak_scores.get(p["protein_id"], 1.0) >= 0.20
    ] or weak_proteins[:1]
    if len(survivors) != 1:
        failures.append(
            f"a proposal with no ORF above threshold folded {len(survivors)}, expected 1"
        )
    return failures


def check_acrnet_tiers() -> list[str]:
    """AcrNET tiers must be per-PSSM-regime, not one global table.

    A missing PSI-BLAST PSSM inflates AcrNET toward 1.0 regardless of class:
    zeroing the PSSM on the same 63 phage proteins moved them from 19% to 78%
    above 0.99. Reading one global table therefore rewards every ORF that
    PSI-BLAST cannot profile -- which is exactly the divergent candidates the
    pipeline exists to find.

    The control is the last assertion: one score must land in DIFFERENT tiers
    in the two regimes. A reversion to a single shared table would tier it
    identically and fail here.
    """
    import json

    failures: list[str] = []
    path = (
        Path(__file__).resolve().parents[1]
        / "data/models/acr/acrnet_operating_points.json"
    )
    if not path.exists():
        return [f"missing operating points: {path}"]
    ops = json.loads(path.read_text())

    if set(ops.get("regimes", {})) != {"has_pssm", "no_pssm"}:
        return [
            f"expected regimes has_pssm/no_pssm, got {sorted(ops.get('regimes', {}))}"
        ]

    def tier_of(regime: str, score: float) -> str | None:
        for entry in ops["regimes"][regime]["tiers"]:
            if entry["lower"] <= score < entry["upper"]:
                return entry["name"]
        return None

    for name, regime in ops["regimes"].items():
        edges = sorted((t["lower"], t["upper"]) for t in regime["tiers"])
        if edges[0][0] > 0.0 or edges[-1][1] < 1.0:
            failures.append(f"{name}: tiers do not span [0, 1]: {edges}")
        for (_, upper), (lower, _) in itertools.pairwise(edges):
            if upper != lower:
                failures.append(f"{name}: gap between tiers at {upper} -> {lower}")

        ratios = [
            t["likelihood_ratio"]
            for t in sorted(regime["tiers"], key=lambda t: t["lower"])
            if t["likelihood_ratio"] is not None
        ]
        if ratios != sorted(ratios):
            failures.append(f"{name}: likelihood ratios not monotone: {ratios}")
        if ratios and ratios[-1] < 2.0:
            failures.append(f"{name}: top tier LR {ratios[-1]} too weak to be a tier")
        if not regime.get("negative_quantiles"):
            failures.append(f"{name}: no negative_quantiles for the percentile")

    # CONTROL: the regimes must actually disagree, or the split is cosmetic.
    probe = 0.9995
    hit, miss = tier_of("has_pssm", probe), tier_of("no_pssm", probe)
    if hit == miss:
        failures.append(
            f"score {probe} tiered {hit!r} in BOTH regimes; the per-regime "
            "split is doing nothing"
        )
    return failures


def check_acrnet_batch_invariance() -> list[str]:
    """A protein's AcrNET score must not depend on what it was batched with.

    AcrNET pads with index 0, which one-hots to a *valid* residue
    ('A'/'C'/'L'/'E') rather than zeros, so padded positions enter the
    convolution and its max-pool. Scoring in chunks therefore leaked batch
    composition into the score. The control is the second assertion: two
    genuinely different proteins must still score differently, so a `score`
    that returned a constant could not pass this check.
    """
    import numpy as np

    failures: list[str] = []
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "data/models/acr/acrnet/model.ckpt"
    if not checkpoint.exists():
        return [f"missing AcrNET checkpoint: {checkpoint}"]

    from proto_pipelines.callers.acrnet import score

    def synthetic(seed: int, length: int) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        return {
            "sequence": "".join(rng.choice(list("AVGILFPYMTSHNQWRKDEC"), length)),
            "ss3": "".join(rng.choice(list("CEH"), length)),
            "ss8": "".join(rng.choice(list("LHTESGBI"), length)),
            "acc": "".join(rng.choice(list("EBM"), length)),
            "pssm": np.zeros(1110, dtype=np.float32),
            "embedding": rng.normal(0, 0.1, 1280).astype(np.float32),
        }

    query = synthetic(1, 65)
    observed = []
    for companions in (
        [],
        [synthetic(2, 70)],
        [synthetic(2, 800)],
        [synthetic(j, 80 + j * 37) for j in range(5)],
    ):
        features = {"QUERY": query}
        features.update({f"c{j}": c for j, c in enumerate(companions)})
        observed.append(score(features, str(checkpoint))["QUERY"])
    spread = max(observed) - min(observed)
    if spread > 1e-12:
        failures.append(
            f"score moved {spread:.3e} across batch compositions; padding is "
            "leaking batch membership into the result"
        )

    # CONTROL: the check above is only meaningful if score() still separates
    # different inputs at all.
    one = score({"X": synthetic(3, 65)}, str(checkpoint))["X"]
    two = score({"X": synthetic(4, 140)}, str(checkpoint))["X"]
    if abs(one - two) < 1e-9:
        failures.append(
            f"two different proteins both scored {one!r}; the invariance "
            "check above is vacuous"
        )
    return failures


def check_custom_checkpoint() -> list[str]:
    """A custom Evo 2 checkpoint must be loadable, and misuse must raise.

    ``model_local_path`` replaces the HuggingFace download so a fine-tuned or
    modified checkpoint can be sampled. Two ways to get this silently
    wrong, both guarded here: pointing it at an Evo 1 run, where the field
    does not exist and would be dropped (you would sample the stock model and
    never know), and pointing it at a checkpoint *file* instead of the
    weights directory Evo 2 expects.

    The control is the last assertion: with no local path set, the generator
    config must carry ``local_path=None`` rather than an empty string, since
    proto-tools treats "" as a path and fails inside the tool env.
    """
    from proto_pipelines.core.prompts import Prompt
    from proto_pipelines.core.runner import GenerationSettings, _build_generator

    failures: list[str] = []
    prompt = Prompt(index=0, sequence="ACGT" * 8)

    # Evo 1 has no local_path field; setting one must raise, not be ignored.
    try:
        _build_generator(
            prompt,
            GenerationSettings(
                generator="evo1",
                model_checkpoint="evo-1.5-8k-base",
                model_local_path="/nonexistent/weights",
            ),
        )
        failures.append(
            "evo1 + model_local_path was accepted; the path would "
            "be silently dropped and stock weights sampled"
        )
    except ValueError:
        pass
    except Exception as error:
        failures.append(
            f"evo1 + model_local_path raised {type(error).__name__}, "
            "expected ValueError"
        )

    # A file, not a directory, must raise before the tool env is reached.
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        try:
            _build_generator(
                prompt,
                GenerationSettings(generator="evo2", model_local_path=handle.name),
            )
            failures.append("model_local_path pointing at a file was accepted")
        except NotADirectoryError:
            pass
        except Exception as error:
            failures.append(
                f"file path raised {type(error).__name__}, "
                "expected NotADirectoryError"
            )

    # CONTROL: unset must mean None, not "".
    settings = GenerationSettings(generator="evo2")
    if settings.model_local_path != "":
        failures.append("default model_local_path is not empty")
    if (settings.model_local_path or None) is not None:
        failures.append(
            "an unset model_local_path does not normalise to None; "
            "proto-tools would treat '' as a path"
        )
    return failures


def check_af3_gate_governs_candidacy() -> list[str]:
    """A protein that failed the AlphaFold 3 gate must not be a candidate.

    The gate is applied when folding, but the Acr stage deliberately scores
    *every* folded ORF so an Aca partner still contributes locus context
    even when its own fold is poor. Without this, a low-pLDDT protein would
    be scored and then counted as something to test.

    Controls: a protein that passed the gate with the same score must stay a
    candidate (or the check would pass for a function that rejects
    everything), and a protein with no AF3 verdict at all -- the
    sequence-only prescreen stage -- must not be treated as a failure.
    """
    from proto_pipelines.callers.acr import AcrEvidenceConfig, score_proteins

    failures: list[str] = []
    seq = "MKIAELLNRYSDGAALTQEEQAFLDGYFEQLDAQNEALSAEIAALRAQLAGKDA"
    proteins = [
        {
            "protein_id": "failed",
            "sequence": seq,
            "avg_plddt": 18.0,
            "ptm": 0.09,
            "passed_af3_screen": False,
        },
        {
            "protein_id": "passed",
            "sequence": seq,
            "avg_plddt": 88.0,
            "ptm": 0.81,
            "passed_af3_screen": True,
        },
        {"protein_id": "unfolded", "sequence": seq},
    ]
    config = AcrEvidenceConfig(min_score=0.0, require_af3_pass=True)
    records = {r["protein_id"]: r for r in score_proteins(proteins, config)}

    if records["failed"].get("is_candidate") is not False:
        failures.append("a protein that failed the AF3 gate was still a candidate")
    # CONTROL: identical score, gate passed -> must remain a candidate.
    if records["passed"].get("is_candidate") is not True:
        failures.append("a protein that passed the AF3 gate was not a candidate")
    # CONTROL: no verdict is not a failure.
    if records["unfolded"].get("is_candidate") is not True:
        failures.append(
            "a protein with no AF3 verdict was treated as a gate failure; the "
            "sequence-only prescreen stage would reject everything"
        )
    # The failure must still be scored, so locus context survives.
    if records["failed"].get("acr_locus_score") is None and config.divergent_model:
        failures.append("gate failures are not being scored at all")

    off = AcrEvidenceConfig(min_score=0.0, require_af3_pass=False)
    relaxed = {r["protein_id"]: r for r in score_proteins(proteins, off)}
    if relaxed["failed"].get("is_candidate") is not True:
        failures.append("require_af3_pass=False did not re-admit the gate failure")
    return failures


def check_vendored_sources_unmodified() -> list[str]:
    """Vendored third-party files must stay byte-identical to upstream.

    ``vendor/acrnet_model.py`` is the published AcrNET architecture. It has
    to match the shipped checkpoint exactly, and the docs claim it is
    verbatim -- a claim that is only worth making if it is checked. It is
    also excluded from black and ruff, because a formatter silently breaks
    exactly this property: an earlier `ruff --fix` rewrote `super(AcrNET,
    self)` to `super()` and black rewrapped the Conv2d call, leaving the
    behaviour identical and the claim false.

    Hashes are pinned so this runs offline. If you deliberately update a
    vendored file, re-pin the hash in the same commit.
    """
    import hashlib

    failures: list[str] = []
    pinned = {
        # banma12956/AcrNET model.py, fetched 2026-09-24
        "acrnet_model.py": "a560991acf195b1b65136815db1292f0b6f198e505f31cc8e40d16a4fec94132",
    }
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    for name, want in pinned.items():
        path = vendor / name
        if not path.exists():
            failures.append(f"vendored file missing: {path}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            failures.append(
                f"{name} no longer matches upstream (sha256 {got[:16]}..., "
                f"expected {want[:16]}...). A formatter or edit has touched a "
                "verbatim third-party file; restore it or re-pin deliberately."
            )
    return failures


def check_configs_document_every_key() -> list[str]:
    """Every accepted config key must appear in the shipped YAML.

    Two failure directions, both silent. A key in the YAML that the pipeline
    does not accept is rejected at startup -- loud, but only once someone
    runs it. A key the pipeline *does* accept but that appears nowhere in
    the config is invisible: the setting exists, changes behaviour, and no
    user will ever find it.

    A key documented as a commented example counts as documented; that is
    how optional settings are surfaced without turning them on.
    """
    import importlib
    import re

    import yaml

    failures: list[str] = []
    root = Path(__file__).resolve().parents[2]
    pipelines = {
        "acr_sample": "configs/acr_sample.yaml",
        "t2ta_sample": "configs/t2ta_sample.yaml",
        "gene_completion": "configs/gene_completion.yaml",
        "operon_completion": "configs/operon_completion.yaml",
    }
    for pipe, rel in pipelines.items():
        module = importlib.import_module(f"proto_pipelines.pipelines.{pipe}")
        allowed = next(
            (
                getattr(module, name)
                for name in dir(module)
                if isinstance(getattr(module, name), (set, frozenset))
                and name.isupper()
            ),
            None,
        )
        if allowed is None:
            failures.append(f"{pipe}: no ALLOWED_KEYS set found")
            continue
        path = root / "proto_pipelines" / rel
        if not path.exists():
            failures.append(f"missing config: {rel}")
            continue
        text = path.read_text()
        active = set((yaml.safe_load(text) or {}).keys())
        commented = {m.group(1) for m in re.finditer(r"^#\s*([a-z0-9_]+):", text, re.M)}

        rejected = sorted(active - allowed)
        if rejected:
            failures.append(
                f"{rel} sets {rejected}, which {pipe} does not accept; the run "
                "would fail at startup"
            )
        undocumented = sorted(allowed - (active | commented))
        if undocumented:
            failures.append(
                f"{pipe} accepts {undocumented} but {rel} never mentions them, "
                "so the setting is invisible to users"
            )
    return failures


def main() -> int:
    """Run every check and report the results."""
    checks = {
        "repetitiveness vs published": check_repetitiveness,
        "underrepresented AAs vs published": check_underrepresented,
        "pDockQ v1 vs published": check_pdockq,
        "config parsing + retired-key rejection": check_configs,
        "configs document every accepted key": check_configs_document_every_key,
        "survivor collection with and without folding": check_accepted_proteins,
        "filter diagnostics accounting": check_reporting,
        "every chain filter owns its rejections": check_filter_labels,
        "generator swap preserves generation length": check_generator_prepend_parity,
        "fold cap keeps HMM-qualifying proteins": check_fold_cap_keeps_hmm_hits,
        "shipped configs build their generators": check_shipped_configs_build_generators,
        "Acr callers reproduce calibration": check_acr_assets,
        "vendored sources unmodified": check_vendored_sources_unmodified,
        "AcrNET handles a batch of one": check_acrnet_batch_of_one,
        "AcrNET score is batch-invariant": check_acrnet_batch_invariance,
        "Acr model features resolve": check_acr_model_features_resolve,
        "prescreen fold gating": check_prescreen_fold_gating,
        "AUROC statistic incl. ties": check_auroc,
        "custom Evo 2 checkpoint hook": check_custom_checkpoint,
        "AF3 gate governs candidacy": check_af3_gate_governs_candidacy,
        "AcrNET tiers from real distribution": check_acrnet_tiers,
    }
    total_failures = 0
    for name, check in checks.items():
        failures = check()
        total_failures += len(failures)
        status = "PASS" if not failures else "FAIL"
        print(f"[{status}] {name}")
        for failure in failures:
            print(f"        {failure}")
    print(f"\n{len(checks)} check(s) run, {total_failures} failure(s)")
    return 1 if total_failures else 0


if __name__ == "__main__":
    sys.exit(main())
