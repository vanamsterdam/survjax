# DOCS.BAYES.9 — Bayesian RMST guidance handoff

Independent review: round 1/3

## Read first

1. [PLAN.md](PLAN.md), issue #9 and spec 11.3 with its Bayesian use-case notes.
2. docs/use-cases/bayesian-rmst.md: factor recipe, posterior axes, optional
   deterministic route and reproduction instructions.
3. examples/bayesian_rmst/bayesian-rmst.ipynb (Markdown-only diff) and README.md.
4. Existing model.py, run.py and tests/unit/check_bayesian_run.py for evidence.

## Changes and invariants

- Each observation contributes one likelihood term, `(N,)` inside the plate.
  Array broadcasting already computes the vector; the plate records observation
  structure and applies subsample scaling. A scalar sum inside the plate
  broadcasts to N copies. A sum outside is equivalent for full-data inference.
- Subsampling explicitly indexes both the observations and covariates and
  builds a `(B,)` likelihood. NumPyro supplies N/B, with full-data NUTS retained.
- Posterior batches are `(draw, arm)`, with paired control/treatment columns;
  grids add time/horizon first, while Predictive adds draw first. Quantiles
  reduce draws. The optional deterministic route now reuses the existing model
  on the page, keeping posterior-as-batch central.
- Float64 reproduces this committed trial and numerical checks, rather than
  being a universal model requirement. The notebook's NUTS key differs from
  the runner. Full-checkout notebook imports preserve the shared generator;
  the standalone download is explicitly insufficient.
- No library code, model code, notebook code cells, dependencies, protected
  paths, committed posterior or existing notebook outputs changed. No feature
  gate selection changed. No spec deviation was needed for this docs step.

## Validation and requirement evidence

- `scripts/build_docs.sh`: exit 0, both strict executed builds succeeded,
  ending `Verified both dtypes`. Bayesian page execution took 8.59s (x32) and
  6.71s (x64). Read rendered HTML text for the factor, deterministic route,
  shapes and reproduction instructions; checked all new guidance appears in
  both HTML outputs. Spec 3.3, 5, 8.7 and issue #9 documentation acceptance.
- `scripts/check.sh --fast`: exit 0, `check: all gates passed` for baseline=all
  plus observation.censoring, observation.truncation, observation.traced_metadata,
  family.lognormal and family.loglogistic. Ruff: 302 files formatted, all checks
  passed. Mypy: no issues in 25 source files. Implementation x64: 1219 passed,
  8 skipped; x32: 1221 passed, 6 skipped. Doctests: 4 passed per dtype.
  Spec x64: 4654 passed, 46 skipped; x32: 4597 passed, 93 skipped. Both spec
  runs emitted the two existing NumPy log-zero warnings in
  test_a11_ph_direct_methods.py. Tier 2 full gate not run for this docs step.
- `uv run --group examples --group test pytest -q
  tests/unit/check_bayesian_run.py`: 1 passed in 10.96s. The existing test
  checks runner acceptance (factor wiring, RMST route agreement, convergence
  and truth intervals) and all committed arrays bit for bit. Spec 11.3.
- Fresh-kernel execution via nbclient, timeout 180s, working directory
  examples/bayesian_rmst: exit 0, every code cell executed, with no new outputs
  written to source. Supports the documented notebook portability assessment.
  IPython used a temporary directory because ~/.ipython was not writable;
  kernel shutdown logged a psutil sysctl PermissionError under the sandbox
  after execution. These did not fail notebook execution. Kernel launches
  also printed the environment's TCP transport notice during docs execution.
- NumPyro 0.22.0 source inspection and a one-off fixed-factor probe: vector
  `[-1,-2,-3]` -> -6; sum inside size-three plate -> -18; sum outside -> -6.
  Fixed minibatch indices [0,2] gave `(2,)`, scale 1.5 and scaled total -6.
  Verified against official plate documentation, linked on page and notebook.
- Notebook code-cell equality against HEAD passed, including metadata and
  existing outputs. An auxiliary assertion that outputs were empty failed:
  nine cells already had outputs in the starting commit. All remain unchanged.
- `git diff --check`: clean.

## Limits and remaining work

No runtime changes or new tests were necessary. No standalone notebook bundle
or duplicated generator was introduced. No comparison with NumPyro censored
wrappers is claimed, so no unverified tail/version claim was added. NumPyro's
Weibull positional convention is explained locally using explicit keywords.
No remaining sub-steps for issue #9; the optional adapter in #7 is separate.
Integration and roadmap bookkeeping belong to the maintainer; issue #9 can be
closed after integration. No publishing or remote changes were performed.
