# AIPTW.2: runnable notebook for the AIPTW tutorial (plan and record)

- **Step ID:** AIPTW.2, at the maintainer's request: "all that's left is a
  runnable notebook variant". It is the docs to-do of AIPTW.0.
- **Feature:** docs and examples only. No library change, no new dependency
  and no `spec-features.txt` change.
- **Branch:** `aiptw`. **Preceding steps:** AIPTW.0 (`work/AIPTW.0/`) and
  AIPTW.1 (withdrawn, `work/AIPTW.1/`).
  **Step starting commit:** `e5575db0fecc07472636ac26e58b334c4f7ff083`.

## Approach

- **Same layout as the other notebooks.** It follows the Bayesian and MNIST
  notebooks: an executed `examples/aiptw_rmst/aiptw-rmst.ipynb` with outputs,
  a download link from the docs page, a README pointer, the ruff notebook
  entries in `pyproject.toml`, and an optional `tests/unit/check_*` test.
- **Generated from the page.** The notebook was produced once by a
  conversion script, which is not committed:
  - markdown and code cells are carried over as they are;
  - each `literalinclude` of `analysis.py` becomes a code cell with that
    definition, so readers can edit a function and rerun;
  - `propensity_scores` is added, because the page calls it without showing
    it;
  - the page's hidden setup cell becomes a visible setup cell, which imports
    the analysis dependencies instead of `analysis.py`;
  - `{download}` roles become relative links.
- **Kept in step with `analysis.py`.** The notebook duplicates the analysis
  functions, so `check_aiptw_notebook.py` checks that every `analysis.py`
  definition appears unchanged in a notebook cell, and that the notebook
  runs top to bottom in a fresh kernel.
- **One page change.** `zip(..., strict=True)` in the two plotting cells,
  for ruff on the notebook, so the page and the notebook stay identical.

## Docs to-do

- None.
