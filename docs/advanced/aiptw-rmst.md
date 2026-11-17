---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
mystnb:
  render_markdown_format: myst
---

# Doubly robust causal survival analysis

Does adjuvant chemotherapy after surgery for stage III colon cancer buy
patients time, and how much? In a randomized trial you would compare the
arms directly. In routine-care data you cannot: older and frailer patients
are less often given chemotherapy *and* die sooner, so a naive comparison
credits the treatment with the better health of the patients who got it.

This tutorial estimates the causal effect from such observational data, as
the difference in **restricted mean survival time at 5 years**: the expected
number of years, out of the first five, that patients live with
chemotherapy versus with surgery alone. We use **augmented inverse
probability of treatment weighting** (AIPTW), a *doubly robust* estimator
built from two models:

- a **propensity score** model for who gets chemotherapy: logistic
  regression with NumPyro's Bernoulli likelihood;
- an **outcome** model for survival: a Weibull proportional hazards model
  with a survjax likelihood.

Administrative censoring is a **custom survjax distribution** of a few
lines. We will write the whole analysis as one pure function of the data,
so we can run the entire analysis on 1,000 bootstrap resamples in seconds
with a single `jax.vmap`.
The data are simulated, so we know the true answer. The same tutorial is
available as a
{download}`runnable Jupyter notebook <../../examples/aiptw_rmst/aiptw-rmst.ipynb>`,
which defines every analysis function in its own cell.

```{code-cell} ipython3
:tags: [remove-input]
import sys
import warnings
from pathlib import Path

import jax

warnings.filterwarnings("ignore", message="IProgress not found")

jax.config.update("jax_enable_x64", True)

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from examples.aiptw_rmst.analysis import (
    analyse, bootstrap, counterfactual_rmst, fit_propensity, fit_weibull_ph,
    outcome_design_matrix, propensity_scores, weibull_ph,
)
from examples.aiptw_rmst.cohort import TAU, simulate_cohort, true_death_model, true_rmst
from examples.aiptw_rmst.reporting import markdown_table, standardized_difference

ARMS = ("Surgery alone", "Chemotherapy")
COLORS = ("C0", "C1")
```

## The cohort

We simulate 500 patients who had surgery for stage III colon cancer. Each has
three baseline covariates:

| Covariate | Affects chemotherapy? | Affects survival? |
| --- | --- | --- |
| **Age** | yes: older patients get it less often | yes: older patients die sooner |
| **Frailty** (performance status ≥ 2 or severe comorbidity) | yes, strongly | yes |
| **Stromal radiomic score (SRS)** | no | yes, mildly, and it modifies the chemotherapy effect |

Age and frailty are **confounders**. The SRS is a fictional imaging
biomarker. It is an AI score of how stroma-rich the tumour looks on the
preoperative CT. It was computed afterwards on archived scans, so clinicians
never saw it when choosing treatment. Stroma-rich tumours carry a somewhat
worse prognosis and gain a little less from chemotherapy. In the true model
the hazard ratio for chemotherapy is 0.70 at an average SRS and about 0.81
one standard deviation above it.

```{code-cell} ipython3
key = jax.random.PRNGKey(1)
data_key, bootstrap_key = jax.random.split(key)
data = simulate_cohort(data_key, n=500)
data.survival
```

Follow-up is purely administrative: patients entered the study over 4 years
and the analysis is at 8 years. So each patient's potential follow-up is
between 4 and 8 years, and anyone still alive then is censored. Treated and
untreated patients differ before any treatment effect:

```{code-cell} ipython3
:tags: [remove-input]
age_years = 66 + 10 * data.covariates[:, 0]
baseline = {"Age (years)": age_years, "Frail (%)": 100 * data.covariates[:, 1],
            "SRS": data.covariates[:, 2]}
rows = []
for name, x in baseline.items():
    rows.append([name, *(f"{x[data.treated == a].mean():.1f}" for a in (0.0, 1.0)),
                 f"{standardized_difference(x, data.treated, jnp.ones(500)):+.2f}"])
rows.append(["Patients", *(f"{int((data.treated == a).sum())}" for a in (0.0, 1.0)), ""])
display(markdown_table(["", *ARMS, "Standardized difference"], rows))
```

The chemotherapy patients are younger and less often frail. The SRS
played no part in the decision, so any difference between the arms is
chance alone.

## Censoring as a custom survival distribution

With administrative censoring, the follow-up window comes from the study
dates, so we do not need to estimate it. Potential follow-up is uniform
between 4 and 8 years, with censoring survival $G(t)=(8-t)/4$ inside that
window. survjax has no uniform family, so we wrote one,
`UniformFollowUp`, in
{download}`follow_up.py <../../examples/aiptw_rmst/follow_up.py>`. It
takes a few short methods (see [custom distributions](../distributions.md)
for how). It is then a survjax distribution like any other, with batching,
`jit` and every derived method:

```{code-cell} ipython3
from examples.aiptw_rmst.follow_up import UniformFollowUp

FOLLOW_UP = UniformFollowUp(start=4.0, end=8.0)
t = jnp.array([3.0, 5.0, 7.0])
FOLLOW_UP.survival(t), FOLLOW_UP.hazard(t)
```

## Who gets chemotherapy: the propensity score

The propensity score $e(x)=P(A=1\mid X=x)$ is a logistic regression of
treatment on an intercept and all three covariates. The analyst does not
know that the SRS is irrelevant to treatment, and including it costs little.
Both models in this tutorial are fitted by maximum likelihood with one small
helper around optimistix's BFGS:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: minimise
```

NumPyro supplies the Bernoulli log likelihood of the logistic regression:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: fit_propensity
```

Weighting only works if every kind of patient could have received either
treatment (*positivity*). The fitted scores overlap well, with no patient
close to 0 or 1:

```{code-cell} ipython3
ps_theta, _ = fit_propensity(data)
propensity = propensity_scores(ps_theta, data.covariates)
print(f"propensity scores from {propensity.min():.2f} to {propensity.max():.2f}")
```

```{code-cell} ipython3
:tags: [remove-input]
fig, ax = plt.subplots(figsize=(6, 3))
bins = np.linspace(0, 1, 26)
for a, (label, color) in enumerate(zip(ARMS, COLORS, strict=True)):
    ax.hist(propensity[data.treated == a], bins=bins, alpha=0.6, color=color, label=label)
ax.set(xlabel="estimated probability of chemotherapy", ylabel="patients")
ax.legend(frameon=False)
fig.tight_layout()
```

Weighting each patient by $1/\hat e$ (chemotherapy) or $1/(1-\hat e)$
(surgery alone) removes the imbalance in the confounders:

```{code-cell} ipython3
:tags: [remove-input]
ipt_weight = jnp.where(data.treated == 1, 1 / propensity, 1 / (1 - propensity))
rows = [[name, f"{standardized_difference(x, data.treated, jnp.ones(500)):+.2f}",
         f"{standardized_difference(x, data.treated, ipt_weight):+.2f}"]
        for name, x in baseline.items()]
display(markdown_table(["Standardized difference", "Unweighted", "Weighted"], rows))
```

## What survival to expect: the outcome model

The outcome model is a Weibull proportional hazards model with treatment,
the three covariates and a prespecified treatment × SRS interaction, because
the marker is a candidate predictor of benefit. Its design matrix has one
row per patient and five columns:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: outcome_design_matrix
```

The parameter vector `theta` holds the log shape, the log scale and the five
coefficients. Multiplying the design matrix by the coefficients gives every
patient's log hazard ratio, so one `ProportionalHazards` distribution holds
one survival model per patient:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: weibull_ph
```

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: fit_weibull_ph
```

We fit the model on the treatments the patients actually received:

```{code-cell} ipython3
observed_X = outcome_design_matrix(data.treated, data.covariates)  # (500, 5)
outcome_theta, _ = fit_weibull_ph(data.survival, observed_X)
```

From the fitted model we predict, for **every** patient, the 5-year RMST
they would have had under each treatment. We rebuild the design matrix with
the treatment column set to 0 for everyone, and then to 1 for everyone.
The covariates stay as observed. That is one closed-form `rmst` call on a
batch of 500 distributions per treatment:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: counterfactual_rmst
```

```{code-cell} ipython3
rmst_0 = counterfactual_rmst(outcome_theta, data.covariates, 0.0)  # (500,) surgery alone
rmst_1 = counterfactual_rmst(outcome_theta, data.covariates, 1.0)  # (500,) chemotherapy
rmst_0[:3], rmst_1[:3]
```

Averaging these over the cohort is the *g-formula* estimate. It is unbiased
only if the outcome model is right.

## The doubly robust estimator

Let $Y_i=\min(T_i, 5)$ be the restricted survival time, $\hat m_a(X_i)$ the
model RMST above and $\hat e_a$ the probability of the treatment $a$. AIPTW
starts from the outcome model and corrects it with the weighted residuals
of the patients who actually got treatment $a$:

$$
\hat\psi_a=\frac1n\sum_{i=1}^n\left[\hat m_a(X_i)
+\frac{\mathbb 1\{A_i=a\}\,\Delta_i}{\hat e_a(X_i)\,G(Y_i)}\bigl(Y_i-\hat m_a(X_i)\bigr)\right].
$$

Censoring adds one wrinkle. $Y_i$ is known only if the patient died before 5
years or was still followed at 5 years ($\Delta_i=1$). Those patients stand
in for the censored ones, through the inverse probability of censoring
weight $1/G(Y_i)$ from our custom distribution:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: censoring_weights
```

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: aiptw
```

$\hat\psi_a$ is consistent if *either* the propensity model *or* the outcome
model is correct. That is the double robustness.

## The whole analysis as one function

`analyse` takes the data and nothing else. It fits both models, predicts and
returns the per-arm RMST from four estimators. Besides the g-formula and
AIPTW, these are a naive comparison, a Weibull model with treatment only,
and IPTW, the weighted mean without the outcome model:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: naive_rmst
```

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: iptw
```

The results go into a named tuple, so that they stay a JAX pytree:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: Estimates
```

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: analyse
```

```{code-cell} ipython3
estimates = jax.jit(analyse)(data)
for name in ("naive", "g_formula", "iptw", "aiptw"):
    rmst = getattr(estimates, name)
    print(f"{name:>9}: ΔRMST(5) = {rmst[1] - rmst[0]:.3f} years")
```

## Bootstrap confidence intervals in one `vmap`

Because `analyse` is a pure JAX function, the nonparametric bootstrap needs
no loop. We draw a table of patient indices, one row per replicate.
`resample` (from `cohort.py`) picks those rows out of every array in the
data. `jax.vmap` turns the analysis of one resample into the analysis of all
rows of the table at once, and `jax.jit` compiles that into a single
program. Each replicate refits both models from scratch:

```{literalinclude} ../../examples/aiptw_rmst/analysis.py
:pyobject: bootstrap
```

```{code-cell} ipython3
import time

start = time.perf_counter()
replicates = bootstrap(bootstrap_key, data, n_boot=1000)
replicates.aiptw.block_until_ready()
print(f"1000 bootstrap analyses in {time.perf_counter() - start:.1f} s, "
      f"{int(replicates.converged.sum())} with all fits converged")
```

## Results

The true values come from the simulation model, averaged over a million
simulated patients. The naive comparison has its own target, the difference
between the patients who happened to get each treatment:

```{code-cell} ipython3
:tags: [remove-input]
truth = true_rmst(jax.random.PRNGKey(0), n_draws=1_000_000)
true_delta = float(truth["causal"][1] - truth["causal"][0])
naive_target = float(truth["naive"][1] - truth["naive"][0])

labels = {"naive": "Naive (unadjusted)", "g_formula": "g-formula",
          "iptw": "IPTW", "aiptw": "AIPTW"}
rows = []
for name, label in labels.items():
    point = getattr(estimates, name)
    draws = getattr(replicates, name)
    delta = draws[:, 1] - draws[:, 0]
    low, high = jnp.quantile(delta, jnp.array([0.025, 0.975]))
    rows.append([label, f"{point[0]:.2f}", f"{point[1]:.2f}",
                 f"{point[1] - point[0]:.3f}", f"[{low:.3f}, {high:.3f}]"])
rows.append(["**Truth**", f"{truth['causal'][0]:.2f}", f"{truth['causal'][1]:.2f}",
             f"**{true_delta:.3f}**", ""])
rows.append(["Target of the naive comparison", f"{truth['naive'][0]:.2f}",
             f"{truth['naive'][1]:.2f}", f"{naive_target:.3f}", ""])
display(markdown_table(["Estimator", "RMST(5), surgery", "RMST(5), chemo",
                        "ΔRMST(5) (years)", "95% bootstrap CI"], rows))
```

In truth, chemotherapy adds about a quarter of a year, roughly 3 months,
of life in the first five years. The naive comparison more than doubles
that, because it also credits chemotherapy with the younger, fitter
patients who received it. Its interval misses the truth. The intervals of
the adjusted estimators all contain it, and AIPTW gets there while needing
only one of its two models to be right.

```{code-cell} ipython3
:tags: [remove-input]
fig, ax = plt.subplots(figsize=(6, 3))
for name, color in (("naive", "0.6"), ("aiptw", "C2")):
    draws = getattr(replicates, name)
    style = {"histtype": "step", "linewidth": 1.5} if name == "naive" else {"alpha": 0.7}
    ax.hist(draws[:, 1] - draws[:, 0], bins=40, color=color, label=labels[name], **style)
ax.axvline(true_delta, color="black", linestyle="--", label="Truth")
ax.axvline(0, color="0.5", linewidth=0.8)
ax.set(xlabel="ΔRMST(5), chemotherapy minus surgery alone (years)", ylabel="bootstrap replicates")
ax.legend(frameon=False)
fig.tight_layout()
```

The outcome model also gives *standardized* survival curves: each
patient's predicted curve under one treatment, averaged over the whole
cohort. The area under each curve up to the dotted line at 5 years is the
g-formula RMST:

```{code-cell} ipython3
:tags: [remove-input]
times = jnp.linspace(0.0, 8.0, 161)
fig, ax = plt.subplots(figsize=(6, 3.5))
for a, (label, color) in enumerate(zip(ARMS, COLORS, strict=True)):
    fitted = weibull_ph(outcome_theta, outcome_design_matrix(float(a), data.covariates))
    true_curve = true_death_model(data.covariates, float(a)).survival_grid(times).mean(axis=1)
    curve = fitted.survival_grid(times).mean(axis=1)  # (161, 500) -> (161,)
    ax.plot(times, curve, color=color, label=label)
    ax.plot(times, true_curve, color=color, linestyle="--", linewidth=1)
ax.plot([], [], color="0.3", linestyle="--", linewidth=1, label="Truth")
ax.plot(times, FOLLOW_UP.survival(times), color="0.5", linewidth=0.8, label="Censoring G(t)")
ax.axvline(TAU, color="0.5", linestyle=":")
ax.set(xlabel="years since surgery", ylabel="survival probability", ylim=(0, 1.02))
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
```

## Who benefits most?

The interaction in the outcome model lets us ask how the benefit varies with
the SRS. Here it is for a patient aged 66 who is not frail. Every bootstrap
replicate carries its own fitted outcome model, so the band costs one more
`vmap`:

```{code-cell} ipython3
srs = jnp.linspace(-2.0, 2.0, 41)
reference = jnp.column_stack([jnp.zeros_like(srs), jnp.zeros_like(srs), srs])  # age 66, not frail


def benefit_by_srs(theta):
    """(41,) model-based ΔRMST(5) along the SRS grid, for one fitted outcome model."""
    return (counterfactual_rmst(theta, reference, 1.0)
            - counterfactual_rmst(theta, reference, 0.0))


band = jax.vmap(benefit_by_srs)(replicates.outcome_theta)  # (1000, 41)
```

```{code-cell} ipython3
:tags: [remove-input]
true_benefit = (true_death_model(reference, 1.0).rmst(TAU)
                - true_death_model(reference, 0.0).rmst(TAU))
low, high = jnp.quantile(band, jnp.array([0.025, 0.975]), axis=0)
fig, ax = plt.subplots(figsize=(6, 3))
ax.fill_between(srs, low, high, color="C2", alpha=0.2, label="95% bootstrap band")
ax.plot(srs, benefit_by_srs(outcome_theta), color="C2", label="Estimate")
ax.plot(srs, true_benefit, color="black", linestyle="--", label="Truth")
ax.axhline(0, color="0.5", linewidth=0.8)
ax.set(xlabel="stromal radiomic score (SD)", ylabel="ΔRMST(5) (years)")
ax.legend(frameon=False)
fig.tight_layout()
```

In truth the benefit shrinks as the SRS rises. With 500 patients, though, a
treatment × marker interaction is poorly determined. In this cohort the
estimate even slopes the wrong way, and the band is wide enough to hold
either direction. Read such a curve as a description, not as evidence of
who benefits; that needs far more patients.

## Takeaways

- A causal survival analysis is a handful of small, separate functions: a
  propensity model, an outcome model, a censoring distribution and an
  estimator. survjax supplies the censored likelihood, the per-patient
  RMST and the custom distribution. NumPyro and optimistix do the rest.
- Writing the analysis as one pure function of the data makes the
  bootstrap a `vmap`. A thousand complete reanalyses take seconds.
- Every analysis function is shown above;
  {download}`analysis.py <../../examples/aiptw_rmst/analysis.py>` has them
  in one file with their imports. The simulation and the true values are in
  {download}`cohort.py <../../examples/aiptw_rmst/cohort.py>`, and the
  censoring distribution is in
  {download}`follow_up.py <../../examples/aiptw_rmst/follow_up.py>`.

This analysis assumes no unmeasured confounding and positivity. The
simulation guarantees both, but real data do not. The AIPTW estimator here
weights the censoring with a known distribution. The fully efficient
version adds a censoring augmentation term: patients censored before 5 years
then count through their model-predicted RMST given survival to their
censoring time, which narrows the interval (Tsiatis, *Semiparametric Theory
and Missing Data*, Springer 2006, chapters 9 and 10).
