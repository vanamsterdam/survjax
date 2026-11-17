# Doubly robust (AIPTW) estimate of the 5-year RMST difference, in R.
#
# The same analysis as analysis.py, function by function, for the benchmark
# in benchmark/. The explanation is on the tutorial page; not library API.

library(survival)

TAU <- 5  # restriction time, years
FOLLOW_UP <- c(start = 4, end = 8)  # administrative follow-up, uniform, years


# Propensity score: logistic regression --------------------------------------

fit_propensity <- function(data) {
  # Logistic regression of treatment on an intercept and the three covariates.
  glm(treated ~ z_age + frail + srs, family = binomial, data = data)
}

propensity_scores <- function(fit, data) {
  # (N,) fitted probability of chemotherapy.
  predict(fit, newdata = data, type = "response")
}


# Outcome model: Weibull proportional hazards --------------------------------

# Columns: chemo, age, frail, SRS and the chemo x SRS interaction.
OUTCOME_FORMULA <- Surv(time, event) ~ treated + z_age + frail + srs + treated:srs

fit_weibull_ph <- function(formula, data) {
  # Maximum likelihood Weibull fit of right-censored survival times. survreg
  # fits the accelerated failure time form, which for the Weibull is the same
  # model as proportional hazards with coefficients -beta / scale.
  survreg(formula, data = data, dist = "weibull")
}

weibull_rmst <- function(fit, newdata) {
  # (N,) RMST(TAU) of the fitted Weibull for every row of newdata.
  shape <- 1 / fit$scale
  scale <- exp(predict(fit, newdata = newdata, type = "lp"))
  # The integral of exp(-(t / scale)^shape) from 0 to TAU, in closed form
  # with the regularised lower incomplete gamma function.
  scale * gamma(1 + 1 / shape) * pgamma((TAU / scale)^shape, shape = 1 / shape)
}

counterfactual_rmst <- function(fit, data, treated) {
  # (N,) model-based RMST(TAU) for every patient, had they all had `treated`.
  # Same covariates, but the treatment column set to `treated` for everyone.
  counterfactual_data <- data
  counterfactual_data$treated <- treated
  weibull_rmst(fit, counterfactual_data)
}


# Censoring: known administrative follow-up ----------------------------------

censoring_weights <- function(data) {
  # (N,) inverse probability of censoring weights, 0 when the status at TAU is unknown.
  known_by_tau <- data$event == 1 | data$time >= TAU
  survived_censoring <- punif(pmin(data$time, TAU), FOLLOW_UP[["start"]],
                              FOLLOW_UP[["end"]], lower.tail = FALSE)  # >= 0.75
  ifelse(known_by_tau, 1 / survived_censoring, 0)
}


# Estimators ------------------------------------------------------------------

aiptw <- function(outcome, ipcw, treated, propensity, rmst_0, rmst_1) {
  # (2,) augmented IPTW mean of min(T, TAU): (surgery alone, chemotherapy).
  weight_0 <- (1 - treated) * ipcw / (1 - propensity)
  weight_1 <- treated * ipcw / propensity
  psi_0 <- mean(rmst_0 + weight_0 * (outcome - rmst_0))
  psi_1 <- mean(rmst_1 + weight_1 * (outcome - rmst_1))
  c(psi_0, psi_1)
}

iptw <- function(outcome, ipcw, treated, propensity) {
  # (2,) weighted (Hajek) mean of min(T, TAU): (surgery alone, chemotherapy).
  weight_0 <- (1 - treated) * ipcw / (1 - propensity)
  weight_1 <- treated * ipcw / propensity
  psi_0 <- sum(weight_0 * outcome) / sum(weight_0)
  psi_1 <- sum(weight_1 * outcome) / sum(weight_1)
  c(psi_0, psi_1)
}

naive_rmst <- function(data) {
  # (2,) RMST per arm from a Weibull PH model with treatment only.
  fit <- fit_weibull_ph(Surv(time, event) ~ treated, data)
  list(rmst = unname(weibull_rmst(fit, data.frame(treated = c(0, 1)))),
       converged = converged(fit))
}

converged <- function(fit) {
  # survreg reports no flag; it stops at maxiter when it fails to converge.
  fit$iter < survreg.control()$maxiter
}

analyse <- function(data) {
  # The whole analysis of one cohort: fit, predict and estimate.
  ps_fit <- fit_propensity(data)
  propensity <- propensity_scores(ps_fit, data)

  outcome_fit <- fit_weibull_ph(OUTCOME_FORMULA, data)
  rmst_0 <- counterfactual_rmst(outcome_fit, data, 0)
  rmst_1 <- counterfactual_rmst(outcome_fit, data, 1)

  outcome <- pmin(data$time, TAU)  # min(T, TAU) where it is known
  ipcw <- censoring_weights(data)
  naive <- naive_rmst(data)

  list(
    naive = naive$rmst,
    g_formula = c(mean(rmst_0), mean(rmst_1)),
    iptw = iptw(outcome, ipcw, data$treated, propensity),
    aiptw = aiptw(outcome, ipcw, data$treated, propensity, rmst_0, rmst_1),
    converged = ps_fit$converged & converged(outcome_fit) & naive$converged
  )
}

bootstrap <- function(data, n_boot, cores = 1) {
  # Analyse n_boot resamples with replacement, one after another (cores = 1)
  # or forked over `cores` processes. Seed with set.seed(); for cores > 1 use
  # RNGkind("L'Ecuyer-CMRG") so that the replicates are reproducible.
  n <- nrow(data)
  analyse_resample <- function(b) analyse(data[sample.int(n, replace = TRUE), ])
  parallel::mclapply(seq_len(n_boot), analyse_resample, mc.cores = cores)
}
