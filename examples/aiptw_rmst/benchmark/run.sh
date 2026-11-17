#!/usr/bin/env bash
# Mini benchmark: the tutorial analysis in R (survreg) and in survjax.
# Run from the repository root. Needs R with survival; prints CSV rows.
set -euo pipefail

CORES=${CORES:-12}
OUT=${OUT:-$(mktemp -d)}
export JAX_ENABLE_X64=1 PYTHONPATH=.

# survjax simulates every cohort; the 500-patient one is the committed fixture.
cp tests/fixtures/aiptw_rmst/cohort.csv "$OUT/cohort_500.csv"
for n in 5000 50000; do
  uv run --group examples python -m examples.aiptw_rmst.export_cohort "$OUT/cohort_$n.csv" --n "$n"
done

# (cohort size, bootstrap sizes) pairs, smaller for the larger cohorts; the
# bootstrap sizes divide evenly over 12 devices.
grid=("500 120,1200,12000" "5000 120,1200" "50000 120")
for cell in "${grid[@]}"; do
  read -r n boots <<<"$cell"
  Rscript --vanilla examples/aiptw_rmst/benchmark/benchmark.R "$CORES" "$boots" "$OUT/cohort_$n.csv"
  for devices in 1 "$CORES"; do
    XLA_FLAGS=--xla_force_host_platform_device_count=$devices \
      uv run --group examples python examples/aiptw_rmst/benchmark/benchmark.py \
      "$OUT/cohort_$n.csv" --n-boot ${boots//,/ }
  done
done
