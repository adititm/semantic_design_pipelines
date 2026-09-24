"""Prompt-CSV loading.

A prompt file is a CSV whose first column holds the DNA prompt and whose
remaining columns carry metadata the pipeline may need -- ``Protein_Label``
for gene completion, ``Expected_Response`` for operon completion. The whole
table is kept so those columns stay available downstream.

Prompts are not grouped by length: ``Evo1Generator`` requires equal-length
prompts per call, and the runner sidesteps that by sampling one prompt at a
time while Evo batches internally.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

VALID_DNA = set("ACGTacgt")


@dataclass(frozen=True)
class Prompt:
    """One row of a prompt CSV.

    Attributes:
        index: Zero-based row index, used to build stable result IDs.
        sequence: DNA prompt from column 0, upper-cased.
        fields: Every column of the row keyed by header name.
    """

    index: int
    sequence: str
    fields: dict[str, str] = field(default_factory=dict)

    @property
    def prompt_id(self) -> str:
        """Stable identifier for this prompt: ``p<row index>``."""
        return f"p{self.index:04d}"

    def get(self, column: str, default: str = "") -> str:
        """Return a metadata column, or ``default`` when absent."""
        return self.fields.get(column, default)


def read_prompts(path: str | Path) -> list[Prompt]:
    """Read a prompt CSV into ``Prompt`` records.

    Args:
        path: CSV whose first column holds the DNA prompt and whose first row
            is a header. Remaining columns are retained as metadata.

    Returns:
        One ``Prompt`` per data row, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file has no header, no data rows, or a prompt
            contains non-ACGT characters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Prompt CSV has no header row: {path}")
        sequence_column = reader.fieldnames[0]

        prompts: list[Prompt] = []
        for index, row in enumerate(reader):
            raw = (row.get(sequence_column) or "").strip()
            if not raw:
                print(f"WARNING: {path}: row {index} has an empty prompt; skipping.")
                continue
            sequence = raw.upper()
            invalid = sorted(set(sequence) - VALID_DNA - {"N"})
            if invalid:
                raise ValueError(
                    f"{path}: row {index} prompt contains non-DNA characters {invalid}"
                )
            prompts.append(
                Prompt(
                    index=index,
                    sequence=sequence,
                    fields={key: (value or "").strip() for key, value in row.items() if key},
                )
            )

    if not prompts:
        raise ValueError(f"Prompt CSV contained no usable rows: {path}")
    return prompts
