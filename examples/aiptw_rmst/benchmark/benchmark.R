#!/usr/bin/env Rscript
# Time the R analysis: one cohort, then bootstraps of growing size.
#
# Usage: Rscript --vanilla benchmark.R <cores> <n_boot,n_boot,...> <cohort.csv>...
# Prints one CSV row per timing, matching benchmark.py. Run from the
# repository root; see run.sh.

source("examples/aiptw_rmst/analysis.R")

args <- commandArgs(trailingOnly = TRUE)
cores <- as.integer(args[[1]])
n_boots <- as.integer(strsplit(args[[2]], ",")[[1]])
paths <- args[-(1:2)]
RNGkind("L'Ecuyer-CMRG")  # reproducible replicates also when forked

median_seconds <- function(run, repeats) {
  # Median elapsed time of run().
  median(replicate(repeats, system.time(run())[["elapsed"]]))
}

cat("engine,n,n_boot,compile_s,run_s,converged,delta_aiptw,ci_low,ci_high\n")
for (path in paths) {
  data <- read.csv(path)
  n <- nrow(data)
  estimates <- analyse(data)
  run_s <- median_seconds(function() analyse(data), 30)
  cat(sprintf("R,%d,1,,%.5f,%d,%.6f,,\n", n, run_s,
              as.integer(estimates$converged), diff(estimates$aiptw)))

  for (n_boot in n_boots) {
    for (n_cores in unique(c(1L, cores))) {
      set.seed(n_boot)
      start <- Sys.time()
      replicates <- bootstrap(data, n_boot, cores = n_cores)
      run_s <- as.numeric(Sys.time() - start, units = "secs")
      draws <- vapply(replicates, function(r) diff(r$aiptw), numeric(1))
      ci <- quantile(draws, c(0.025, 0.975))
      n_converged <- sum(vapply(replicates, function(r) r$converged, logical(1)))
      cat(sprintf("R (%d core%s),%d,%d,,%.4f,%d,,%.4f,%.4f\n", n_cores,
                  if (n_cores == 1) "" else "s", n, n_boot, run_s, n_converged,
                  ci[[1]], ci[[2]]))
    }
  }
}
