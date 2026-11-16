# DOCS.BAYES.PORTABILITY.9 — Documentation handoff

Independent review: round 1/3

## Read first

1. PLAN.md and preceding BAYES.PORTABILITY.9 PLAN/REVIEW/DEVIATIONS records.
2. docs/use-cases/bayesian-rmst.md Reproduce section.
3. examples/bayesian_rmst/README.md and notebook Running instructions.
4. Existing check_bayesian_run.py for the exact-trial and outside-checkout
   evidence established by the preceding code step.

## Changes and invariants

The notebook needs installed dependencies only, while the command-line runner
and the tutorial's full-model import snippet use a checkout. Its local draw
uses the shared key/calibrated limit and is checked for exact agreement.
The notebook enables float64; the shared runner helper enforces it. Neither
is a general float64 requirement of the model/factor. These instructions now
agree across the page, README and notebook. The checked dependency versions
are explicit (addresses previous review N-2), without claiming a fresh install
or broader version matrix. Notebook code and saved outputs are unchanged.
No API, dependencies, tests, spec or protected paths changed.

## Validation

- `scripts/build_docs.sh`: initial sandbox run exited 2 on the existing
  Weibull-simulation page: sysctl -n machdep.cpu.brand_string was denied.
  Kernel shutdown also logged sandbox psutil sysctl errors. Re-ran the exact
  script with approved escalation: exit 0, both strict builds succeeded,
  ending `Verified both dtypes`. Bayesian page execution: 5.50s x32, 5.59s x64.
  Kernel launches printed the environment's TCP transport notices.
- Rendered HTML text inspected in both dtypes: standalone use, explicit
  versions and shared-runner distinction present; old insufficient-download
  claim absent. Both downloadable notebooks are byte-identical to source.
- `scripts/check.sh --fast`: exit 0, all gates passed with the unchanged
  committed feature selection. Ruff: 307 files formatted, all checks passed.
  Mypy: no issues in 25 source files. Implementation x64: 1219 passed/8 skipped;
  x32: 1221 passed/6 skipped. Doctests: 4 passed per dtype. Spec x64: 4654
  passed/46 skipped; x32: 4597 passed/93 skipped. Two existing NumPy log-zero
  warnings per spec dtype. Tier 2 not run for this docs step.
- Installed metadata verified: survjax 0.0.1a3, NumPyro 0.22.0, Matplotlib
  3.11.2, JupyterLab 4.6.4, ipykernel 7.4.0, JAX 0.11.2, NumPy 2.5.3.
- Parsed notebook JSON and compared all code cells, metadata and saved outputs
  with the starting commit: unchanged. No fresh standalone environment install
  or repeat NUTS run needed for these prose-only changes.
- git diff --check: clean.

## Remaining work and limits

Covers the preceding code step's Docs to-do; no further required issue #9
sub-steps. Previous code-review N-1 (unused tmp_path argument) is a nonblocking
nit outside this documentation step. The approved local-draw deviation remains
open for the maintainer to resolve, unchanged here. Integration and issue
closure belong to the maintainer. No publishing or remote writes performed.
The previous diff hook's nvim.log was preserved at
/tmp/survjax-bayesian-portability-nvim.log before review.
