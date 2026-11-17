#!/usr/bin/env Rscript
# R point estimates of the AIPTW tutorial analysis on the survjax-simulated
# cohort, for tests/unit/check_aiptw.py. Run from the repository root:
#   Rscript --vanilla scripts/fixtures/generate_aiptw_r_estimates.R
script <- "scripts/fixtures/generate_aiptw_r_estimates.R"
fixture <- "tests/fixtures/aiptw_rmst"
stopifnot(file.exists(script), getRversion() == "4.4.2",
          as.character(utils::packageVersion("survival")) == "3.7.0")
options(OutDec = ".", warn = 2)
source("examples/aiptw_rmst/analysis.R")

data <- read.csv(file.path(fixture, "cohort.csv"))
stopifnot(nrow(data) == 500L)
estimates <- analyse(data)
stopifnot(estimates$converged)

pair <- function(x) sprintf("[%.17g, %.17g]", x[[1]], x[[2]])
lines <- c(
  "{",
  sprintf('  "generator": "%s",', script),
  sprintf('  "R": "%s", "survival": "%s",', getRversion(), utils::packageVersion("survival")),
  sprintf('  "cohort_md5": "%s",', unname(tools::md5sum(file.path(fixture, "cohort.csv")))),
  sprintf('  "naive": %s,', pair(estimates$naive)),
  sprintf('  "g_formula": %s,', pair(estimates$g_formula)),
  sprintf('  "iptw": %s,', pair(estimates$iptw)),
  sprintf('  "aiptw": %s', pair(estimates$aiptw)),
  "}"
)
writeLines(lines, file.path(fixture, "r_estimates.json"))
