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

| Patients | Task | R, 1 core | R, 12 cores | JAX, 1 device (about 6 cores busy) | JAX, 12 devices |
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

**"1 device" is not one core.** A single JAX process keeps about 5.8 cores
busy: 1,200 replicates of 500 patients take 4.0 s of wall time and 23 s of
CPU time. R takes 4.0 s of CPU time for the same work on one core. Setting
`--xla_cpu_multi_thread_eigen=false` or `--xla_cpu_use_thunk_runtime=false`
does not change this, and JAX 0.11 has no setting that limits its CPU threads.

## Other ways to spread the replicates

These were measured in scratch scripts on 1,200 replicates of 500 patients,
unless a row says otherwise. The times are warm, with compilation excluded.

| Strategy | Time |
| --- | --- |
| `vmap` over all 1,200 replicates (the table above) | 4.0 s |
| `lax.map(..., batch_size=100)`: vmap 100 at a time, the batches in series | 4.3 s |
| `lax.map(..., batch_size=10)` | 12.0 s |
| `lax.map(..., batch_size=1)`: fully serial | 13.2 s |
| 5,000 patients, 240 replicates: full `vmap` / batch 10 / batch 1 | 8.5 / 11.0 / 24.8 s |
| `pmap` or `shard_map` over 12 CPU devices | fails, see below |
| 12 processes, each vmapping 100 replicates | 2.0 s, plus 8.3 s to start and compile |
| 12 processes, 1,000 replicates each (12,000 in total) | 23 s, against R's 3.8 s on 12 cores |

- **Batching helps.** Larger `vmap` batches are never slower, and a fully
  serial map is about 3 times slower than one `vmap`. So the lockstep while
  loop costs less than the per-step overhead it saves.
- **`pmap` and `shard_map` fail inside optimistix.** In JAX 0.11's
  manual-sharding mode, lineax builds the BFGS identity operator and
  compares its input and output structures. The output carries a sharding
  annotation, so the comparison fails ("Symmetric/Hermitian matrices must
  have matching input and output structures"). The sharded `jit` above
  avoids this, but runs every device until the global stopping test.
- **Processes do not scale either.** Since each JAX process already takes
  about 6 cores, adding processes mostly divides the same cores between
  them. With 100 replicates per process, the time grows from 0.39 s with 1
  process to 0.43, 0.69, 1.42 and 2.11 s with 2, 4, 8 and 12 processes.

## Reading the results

**On CPU, JAX does not pay off for this analysis.**

- **At 500 patients,** a vmapped bootstrap matches single-core R in wall
  time, about 3.3 ms per replicate, but it uses about 6 cores to do so.
- **For larger cohorts, R is about twice as fast** in wall time.
- **With 12 cores,** R's forked processes speed up about 10 times. No JAX
  strategy tried here got better than a 2-fold speed-up.

There are three reasons:

1. **The optimiser does about ten times more work.** BFGS takes 45 to 105
   steps per fit, and each step evaluates the likelihood and its gradient at
   least once. survreg's Newton–Raphson converges in a handful of
   iterations with an analytic Hessian. The gap is the algorithm, not the
   language. Plain Newton in optimistix diverges from the tutorial's starting
   point, because it has no step halving like survreg's.
2. **JAX already spends the spare cores.** XLA's CPU runtime spreads one
   process over about 6 cores, so there is little headroom left for more
   devices or processes.
3. **CPU work has little to vectorise.** A batch of 1,200 × 500 patients
   costs about as much work as 1,200 separate fits. There is no wide parallel
   hardware to fill.

JAX would plausibly pay off on a GPU, where the 1,200 × 500 batch is one
small kernel per step. It would also pay off with a damped Newton solver, or
when gradients of the whole analysis are needed, which R cannot give. None of
these cases was measured here.
