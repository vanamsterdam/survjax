# AIPTW.2 review notes

## Read first, in order

1. `examples/aiptw_rmst/aiptw-rmst.ipynb`: compare it with
   `docs/advanced/aiptw-rmst.md`.
2. `tests/unit/check_aiptw_notebook.py`.
3. The links in `examples/aiptw_rmst/README.md`, the page intro and
   `pyproject.toml`.

## Invariants

- **The notebook's function cells equal `analysis.py`.** Each
  `analysis.py` definition must appear in a notebook cell, compared exactly
  on its source.
- **The notebook runs from the `examples/aiptw_rmst` directory** and finds
  the repository root through `pyproject.toml`, as the Bayesian notebook
  does. It sets float64 itself.

## Checks run

- `ruff format --check .` and `ruff check .` pass; the notebook is linted,
  with E402 ignored as for the other notebooks.
- `tests/unit/check_aiptw_notebook.py` (docs and examples groups): 2 passed.
- `tests/unit/check_aiptw.py` in float64: 7 passed.
- `scripts/build_docs.sh` succeeded in both dtypes, and the page's download
  link points to the copied notebook.
- `scripts/check.sh --fast`: 4633 passed, 93 skipped, all gates passed.
- **Not run:** the full `scripts/check.sh`, and the `independent-review`
  skill, which is unavailable in this session.

## Unsure about

- **The committed outputs include a wall time** ("1000 bootstrap analyses
  in 4.8 s"), which varies by machine.
- **The notebook is 180 kB,** mostly the four PNG figures.
