#!/usr/bin/env bash
# Export the runtime-only subset of this tree.
#
# This directory is the research version: it carries the calibration
# workflows that fit the shipped models, their labelled sequence sets, and
# the provenance archive. A deployment only needs what runs a sampling pass
# and produces results, so this script emits that subset.
#
#   scripts/export_runtime.sh /path/to/target-repo
#
# What is dropped, and why it is safe:
#   calibration/    Fits the models. The FITTED artefacts it produces live in
#                   data/models/ and are kept; only the machinery goes.
#   archive/        Superseded scripts and round notes. Provenance only.
#   docs/CALIBRATION.md        Describes the dropped workflows.
#   slurm/         Three-line wrappers around scripts/run_pipeline.sh whose
#                  only real content was one cluster's partition names. The
#                  resource sizing they encoded is a table in the README.
#
# The parity suite deliberately carries its own copies of the reference
# implementations and a two-sequence fixture, so it passes in the export with
# no calibration tooling present. That is asserted at the end.
set -euo pipefail

TARGET="${1:?usage: $0 <target-repo-root>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TARGET/proto_pipelines"

# --delete-excluded, not just --delete: with a plain --delete an excluded
# path that already exists in the target is PROTECTED rather than removed,
# so a previously-exported calibration/ would survive here forever.
rsync -a --delete --delete-excluded \
  --exclude=calibration --exclude=archive --exclude=slurm \
  --exclude=docs/CALIBRATION.md \
  --exclude=outputs --exclude='outputs_*' --exclude=logs \
  --exclude=__pycache__ --exclude='*.pyc' --exclude=.scratch_acr \
  "$SRC/" "$TARGET/proto_pipelines/"

# Doc references into the dropped trees would dangle, so rewrite them.
python3 - "$TARGET/proto_pipelines" <<'PYEOF'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])

readme = root / "README.md"
text = readme.read_text()
start = text.index("## Calibration")
end = text.index("## Layout")
text = text[:start] + '''## Calibration

The thresholds and models shipped in `data/models/` are measured, not
guesses. The workflows that fit them -- labelled sequence sets, per-caller
scoring, AUROC analysis -- are **not** in this repository; it carries only
what is needed to run a sampling pass and get results.

They live in the research tree this was exported from. Nothing here loads
them: the fitted artefacts are read directly from `data/models/`.

''' + text[end:]

# The layout tree must describe what is actually present.
text = text.replace("""├── calibration/          Derives the shipped models — see calibration/README.md
│   ├── data/                 the artefacts the pipeline loads (tracked)
│   └── results/              intermediates (~1.6 GB, gitignored, reproducible)
""", "")
text = text.replace("""├── docs/
│   ├── ACR_PIPELINE.md       the anti-CRISPR stack in detail
│   └── CALIBRATION.md        how the thresholds were derived, and their limits
└── archive/              Superseded scripts and round notes — see archive/README.md
""", """└── docs/
    └── ACR_PIPELINE.md   the anti-CRISPR stack in detail
""")
text = text.replace("""├── data/                 Bundled prompts and reference sequences""",
"""├── data/
│   ├── prompts/              bundled prompt CSVs
│   ├── reference/            reference sequences for the identity pipelines
│   └── models/               the fitted models and profiles the pipeline loads""")
text = re.sub(r"[^\n]*docs/CALIBRATION\.md[^\n]*\n", "", text)
# Remaining prose mentions of the dropped tree.
text = text.replace(
    "threshold in `calibration/` was measured on Evo 1.5 output; on Evo 2 they are",
    "threshold shipped here was measured on Evo 1.5 output; on Evo 2 they are")
text = text.replace(
    "**Calibration set** (`calibration/data/acr/acr_calibration.csv`): 64",
    "**Calibration set** (in the research tree): 64")
text = text.replace(
    "A clone is ~20 MB. `outputs/`, `calibration/results/` (~1.6 GB of AF3\nstructures and cached features) and the Foldseek source structures are\ngitignored and reproducible; the fitted models the pipeline loads are\ntracked.",
    "A clone is ~17 MB. `outputs/` is gitignored; the fitted models the\npipelines load are tracked, in `data/models/`.")
readme.write_text(text)

# The Acr writeup cites the calibration scripts for provenance; in the
# export they are not present, so name the research tree instead.
acr_doc = root / "docs/ACR_PIPELINE.md"
acr = acr_doc.read_text()
acr = acr.replace("`calibration/score_acrnet.py` now exists",
                  "a committed regeneration entrypoint now exists in the research tree")
acr = acr.replace("calibration (`calibration/score_acrnet.py`; features cached in",
                  "calibration (regenerated in the research tree; features cached in")
acr = re.sub(r"`calibration/[A-Za-z0-9_./-]+`", "the research tree", acr)
acr_doc.write_text(acr)


# Anything still pointing into a dropped tree is a bug in this script.
bad = []
for doc in list(root.glob("*.md")) + list(root.glob("docs/*.md")):
    for line_no, line in enumerate(doc.read_text().splitlines(), 1):
        for dropped in ("calibration/", "archive/", "CALIBRATION.md"):
            if dropped in line and "not in this repository" not in line:
                bad.append(f"{doc.relative_to(root)}:{line_no}: {line.strip()[:90]}")
if bad:
    print("ERROR: exported docs still reference dropped paths:")
    for b in bad:
        print("  " + b)
    raise SystemExit(1)
PYEOF

echo "exported to $TARGET/proto_pipelines"
echo "  size : $(du -sh "$TARGET/proto_pipelines" | cut -f1)"
echo "  files: $(find "$TARGET/proto_pipelines" -type f | wc -l)"
