# Mini benchmark: the AIPTW analysis in survjax and in R

This benchmark times the analysis of the tutorial page in two ways:
`analysis.py` (NumPyro, optimistix BFGS and survjax) and its line-by-line R
port `../analysis.R` (`glm` and `survreg`). Both run on cohorts that survjax
simulates. The two give the same estimates, to a relative 1e-9, on the
committed 500-patient cohort (see `tests/fixtures/aiptw_rmst/`). The
bootstrap intervals differ only through their random resamples.

```sh
examples/aiptw_rmst/benchmark/run.sh   # from the repository root, about 10 minutes
```

## Results

Measured on 2026-11-17 on an Apple M4 Max (16 cores), on CPU only. The
software was R 4.4.2 with survival 3.7-0, and JAX 0.11.2 with optimistix 0.1.0
in float64.

- **One analysis** is the median of 30 runs for both engines.
- **Bootstrap** times are wall time for all the replicates. The JAX
  bootstrap times are the median of 3 runs, and R's are a single run.
- **Compilation** takes 1.4 to 1.7 s for each bootstrap size and is not
  included in the JAX times. Every replicate converged.

| Patients | Task | R, 1 core | R, 12 cores | JAX, 1 device | JAX, 12 devices |
| --- | --- | --- | --- | --- | --- |
| 500 | one analysis | 3.0 ms | | 5.8 ms | |
| 500 | 120 replicates | 0.39 s | 0.09 s | 0.40 s | 0.23 s |
| 500 | 1,200 replicates | 3.9 s | 0.41 s | 3.9 s | 2.2 s |
| 500 | 12,000 replicates | 40 s | 3.8 s | 44 s | 25 s |
| 5,000 | one analysis | 18 ms | | 65 ms | |
| 5,000 | 120 replicates | 2.0 s | 0.24 s | 4.3 s | 2.3 s |
| 5,000 | 1,200 replicates | 20 s | 2.1 s | 41 s | 24 s |
| 50,000 | one analysis | 0.17 s | | 0.34 s | |
| 50,000 | 120 replicates | 23 s | 2.6 s | 45 s | 27 s |

## Reading the results

**On CPU, JAX does not pay off for this analysis.**

- **At 500 patients,** a vmapped bootstrap is as fast as R's loop, about
  3.3 ms per replicate.
- **For larger cohorts, R is about twice as fast.**
- **With 12 cores,** R's forked processes speed up about 10 times. Sharding the
  JAX bootstrap over 12 CPU devices speeds it up less than 2 times.

There are three reasons:

1. **The optimiser does about ten times more work.** BFGS takes 45 to 105
   steps per fit, and each step evaluates the likelihood and its gradient at
   least once. survreg's Newton–Raphson converges in a handful of
   iterations with an analytic Hessian. The gap is the algorithm, not the
   language. Plain Newton in optimistix diverges from the tutorial's starting
   point, because it has no step halving like survreg's.
2. **`vmap` over an iterative fit runs in lockstep.** The batched while loop
   runs until the slowest replicate has converged. Under sharding, every
   device waits for that global stopping test, which is why 12 devices give
   less than a 2-fold speed-up.
3. **CPU work has little to vectorise.** A batch of 1,200 × 500 patients is
   about as fast as 1,200 separate fits. There is no wide parallel hardware to
   fill.

JAX would plausibly pay off on a GPU, where the 1,200 × 500 batch is one
small kernel per step. It would also pay off with a damped Newton solver, or
when gradients of the whole analysis are needed, which R cannot give. None of
these cases was measured here.
