# Bayesian survival demo

A Bayesian Weibull proportional hazards model of a synthetic two-arm trial,
fitted with NumPyro, with the treatment effect reported as a difference in
restricted mean survival time. All data are synthetic. Read the tutorial on
the [docs page](../../docs/use-cases/bayesian-rmst.md), or run it cell by cell
in the [Jupyter notebook](bayesian-rmst.ipynb), which samples the posterior
itself:

```sh
uv run --group docs --group examples jupyter lab examples/bayesian_rmst/bayesian-rmst.ipynb
```

- `model.py`: the simulated trial (`make_trial`) and the NumPyro model.
- `run.py`: samples the posterior and saves the draws the docs page uses.
- `results/`: the committed run, `posterior.npz` (trial and chain-grouped
  draws) and `metadata.json` (priors, seed, versions, diagnostics, truth).

To sample again, from the repository root:

```sh
JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.run /tmp/survjax-bayes
```

The output directory must be new. The run checks the likelihood factor, the
agreement of the two RMST routes, the convergence diagnostics and that the true
hazard ratio and ΔRMST(5) lie inside their 95% credible intervals; it saves its
results before checking. survjax itself does not depend on the `examples`
group.

Use float64 to reproduce the committed trial and its numerical checks; it is
not a general requirement of the Weibull model or the NumPyro factor. The
shared `make_trial()` helper raises `RuntimeError` if float64 is disabled to
protect that reproduction. The notebook uses a different NUTS key, so use the
runner for the committed posterior.

The factor takes the `(N,)` per-patient likelihood inside the observation plate.
The array calculation is already vectorized. The plate records independence
and shapes and supplies scaling when subsampling. Summing inside that plate
broadcasts the total back to `N` entries and counts the likelihood `N` times;
a summed factor outside the plate is equivalent for full-data inference.
For minibatches, explicitly index both `data[idx]` and `arm[idx]` before
evaluating the likelihood; NumPyro supplies `N / batch_size` scaling.
The NUTS runner uses full data.

The notebook needs a full checkout, including `model.py` and the shared
`examples/weibull_simulation/study.py` generator. It finds the repository root
from either the root or its own directory; the notebook download alone is not
standalone. Keeping this import ensures both trials use the same generator.
