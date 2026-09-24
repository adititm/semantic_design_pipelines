#!/usr/bin/env bash
# Run every pipeline's smoke config, in increasing cost order.
#
# The fastest full check that an install works end to end. Completion first
# (no folding, minutes), then the two folding pipelines (AlphaFold 3, ~1 hour
# each on one GPU).
#
#   scripts/run_smoke.sh              # everything
#   scripts/run_smoke.sh nofold       # skip the folding pipelines
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG=proto_pipelines/configs/smoke

RUN=("gene_completion   $CFG/gene_completion_smoke.yaml"
     "operon_completion $CFG/operon_completion_smoke.yaml")
if [ "${1:-}" != "nofold" ]; then
    RUN+=("t2ta_sample   $CFG/t2ta_sample_smoke.yaml"
          "t2ta_sample   $CFG/t2ta_evo2_smoke.yaml"
          "t2ta_sample   $CFG/t2ta_hmm_gate_smoke.yaml"
          "acr_sample    $CFG/acr_sample_smoke.yaml")
fi

FAILED=()
for entry in "${RUN[@]}"; do
    read -r pipeline config <<<"$entry"
    echo; echo "############ $pipeline  <-  $(basename "$config")"
    # Parity runs once, on the first pipeline only.
    if SKIP_CHECKS="${SEEN:-0}" "$HERE/run_pipeline.sh" "$pipeline" "$config"; then
        SEEN=1
    else
        FAILED+=("$pipeline/$(basename "$config")")
        SEEN=1
    fi
done

echo
if [ "${#FAILED[@]}" -eq 0 ]; then
    echo "ALL SMOKE RUNS PASSED (${#RUN[@]} run(s))"
else
    echo "FAILED (${#FAILED[@]} of ${#RUN[@]}):"
    printf '  %s\n' "${FAILED[@]}"
    exit 1
fi
