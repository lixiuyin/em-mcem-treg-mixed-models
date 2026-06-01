# General Linear Mixed Model (LMM): comparison of hand-coded EM/MCEM against library fits (lme4 / nlme)
# ---------------------------------------------------------------------------
# Model: y_i = X_i beta + Z_i u_i + epsilon, Z_i = X_i[,1:k]
#        (random intercept + random slope for the first covariate, k=2).
# lme4 mapping: y ~ c1 + c2 + c3 + (1 + c1 | group)
# nlme mapping: lme(y ~ c1+c2+c3, random = ~ 1 + c1 | group, method = "ML"/"REML")
# Hand-coded EM defaults to ML. Setting options(lmm_reml=TRUE) switches the
# variance-component updates and objective to REML, matching lme4(REML=TRUE).
# ---------------------------------------------------------------------------
suppressMessages({library(this.path); library(lme4); library(nlme)})
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

set.seed(2025)
N_SIM <- as.integer(Sys.getenv("N_SIM", "200"))
N_SIM_MCEM <- as.integer(Sys.getenv("N_SIM_MCEM", "30"))
MC_TIMES <- 50

m <- 50
group_sizes <- sample(c(15, 20, 25), m, replace = TRUE)
beta_real <- c(1, 2, -1, 0.5)
G0_real <- matrix(c(4, 1, 1, 2), nrow = 2, byrow = TRUE)
sigma2_real <- 1
beta_0 <- matrix(0, 4, 1); G0_0 <- diag(2); sigma2_0 <- 0.5
out_dir <- this.dir()

# Build a long-format data frame from the internal data object (X columns: intercept + c1 + c2 + c3).
build_frame <- function(data) {
  grp <- rep(1:data$m, times = data$group_sizes)
  data.frame(y = data$y, c1 = data$X[, 2], c2 = data$X[, 3], c3 = data$X[, 4], group = factor(grp))
}

fit_lme4 <- function(fr, reml = FALSE) {
  mfit <- lmer(y ~ c1 + c2 + c3 + (1 + c1 | group), data = fr, REML = reml,
               control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 3e5),
                                     check.conv.singular = .makeCC("ignore", tol = 1e-4)))
  list(beta = as.numeric(fixef(mfit)), G0 = matrix(VarCorr(mfit)$group, 2, 2),
       sigma2 = sigma(mfit)^2, loglik = as.numeric(logLik(mfit)))
}

fit_nlme <- function(fr, method = "ML") {
  method <- match.arg(method, c("ML", "REML"))
  mfit <- tryCatch(lme(y ~ c1 + c2 + c3, random = ~ 1 + c1 | group, data = fr, method = method,
                       control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200)),
                   error = function(e) NULL)
  if (is.null(mfit)) return(NULL)
  vc <- as.matrix(getVarCov(mfit))
  list(beta = as.numeric(fixef(mfit)), G0 = matrix(vc, 2, 2),
       sigma2 = mfit$sigma^2, loglik = as.numeric(logLik(mfit)))
}

run_hand_em <- function(data, reml = FALSE, is_mcem = FALSE, mc_times = MC_TIMES,
                        max_iter = if (is_mcem) 50 else 300, tolerance = if (is_mcem) 1e-6 else 1e-8) {
  old <- getOption("lmm_reml", FALSE)
  on.exit(options(lmm_reml = old), add = TRUE)
  options(lmm_reml = reml)
  run_single_em_or_mcem(data, beta_0, G0_0, sigma2_0, max_iter = max_iter, tolerance = tolerance,
                        is_mcem = is_mcem, mc_times = mc_times)
}

last_loglik <- function(result) result$history[[length(result$history)]]$loglik

diff_line <- function(a, b) {
  sprintf("max|beta|=%.2e max|G0|=%.2e |sigma^2|=%.2e |logLik|=%.2e",
          max(abs(a$final_beta_hat - b$beta)), max(abs(a$final_G0_hat - b$G0)),
          abs(a$final_sigma2_hat - b$sigma2), abs(last_loglik(a) - b$loglik))
}

cat("================= Single-dataset element-wise comparison (ML and REML) =================\n")
data1 <- generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real)
fr1 <- build_frame(data1)

em_ml <- run_hand_em(data1, reml = FALSE, max_iter = 500, tolerance = 1e-10)
em_reml <- run_hand_em(data1, reml = TRUE, max_iter = 1000, tolerance = 1e-10)
l4_ml <- fit_lme4(fr1, reml = FALSE); l4_reml <- fit_lme4(fr1, reml = TRUE)
ln_ml <- fit_nlme(fr1, "ML"); ln_reml <- fit_nlme(fr1, "REML")

cat("True values: beta=(1,2,-1,0.5)  G0 diagonal=(4,2)  sigma^2=1\n")
cat(sprintf("%-14s %-26s %-18s %-10s %-10s\n", "Method", "beta", "G0 diag", "sigma^2", "logLik"))
print_est <- function(label, beta, G0, sigma2, loglik) {
  cat(sprintf("%-14s (%5.3f,%5.3f,%6.3f,%5.3f) (%6.3f,%6.3f)  %-10.4f %-10.4f\n",
              label, beta[1], beta[2], beta[3], beta[4], diag(G0)[1], diag(G0)[2], sigma2, loglik))
}
print_est("hand EM ML", as.numeric(em_ml$final_beta_hat), em_ml$final_G0_hat, em_ml$final_sigma2_hat, last_loglik(em_ml))
print_est("lme4 ML", l4_ml$beta, l4_ml$G0, l4_ml$sigma2, l4_ml$loglik)
if (!is.null(ln_ml)) print_est("nlme ML", ln_ml$beta, ln_ml$G0, ln_ml$sigma2, ln_ml$loglik)
print_est("hand EM REML", as.numeric(em_reml$final_beta_hat), em_reml$final_G0_hat, em_reml$final_sigma2_hat, last_loglik(em_reml))
print_est("lme4 REML", l4_reml$beta, l4_reml$G0, l4_reml$sigma2, l4_reml$loglik)
if (!is.null(ln_reml)) print_est("nlme REML", ln_reml$beta, ln_reml$G0, ln_reml$sigma2, ln_reml$loglik)

cat(sprintf("[diff] hand EM ML   vs lme4 ML  : %s\n", diff_line(em_ml, l4_ml)))
cat(sprintf("[diff] hand EM REML vs lme4 REML: %s\n", diff_line(em_reml, l4_reml)))
if (!is.null(ln_ml)) {
  cat(sprintf("[diff] hand EM ML   vs nlme ML  : max|beta|=%.2e max|G0|=%.2e |sigma^2|=%.2e\n",
              max(abs(em_ml$final_beta_hat - ln_ml$beta)), max(abs(em_ml$final_G0_hat - ln_ml$G0)),
              abs(em_ml$final_sigma2_hat - ln_ml$sigma2)))
}
if (!is.null(ln_reml)) {
  cat(sprintf("[diff] hand EM REML vs nlme REML: max|beta|=%.2e max|G0|=%.2e |sigma^2|=%.2e\n",
              max(abs(em_reml$final_beta_hat - ln_reml$beta)), max(abs(em_reml$final_G0_hat - ln_reml$G0)),
              abs(em_reml$final_sigma2_hat - ln_reml$sigma2)))
}

cat(sprintf("\n================= %d-run simulation aggregate (ML vs REML bias / MSE) =================\n", N_SIM))
abind2 <- function(a, mm) array(c(a, mm), dim = c(2, 2, dim(a)[3] + 1))
new_store <- function() list(b = matrix(0, 4, 0), G = array(0, c(2, 2, 0)), s = numeric(0))
add_store <- function(st, b, G, s2) {
  st$b <- cbind(st$b, as.numeric(b)); st$G <- abind2(st$G, G); st$s <- c(st$s, as.numeric(s2)[1]); st
}
mse_lmm <- function(b, G, s2) {
  n <- ncol(b); v <- numeric(n)
  for (i in 1:n) v[i] <- norm(matrix(b[, i], ncol = 1) - matrix(beta_real, ncol = 1), "2") +
    norm(G[, , i] - G0_real, "F") + abs(s2[i] - sigma2_real)
  mean(v)
}
mdiag <- function(G) rowMeans(apply(G, 3, diag))
print_summary <- function(label, st) {
  cat(sprintf("%-14s (%5.3f,%5.3f,%6.3f,%5.3f) (%6.3f,%6.3f)  %-10.4f %-10.4f\n",
              label, rowMeans(st$b)[1], rowMeans(st$b)[2], rowMeans(st$b)[3], rowMeans(st$b)[4],
              mdiag(st$G)[1], mdiag(st$G)[2], mean(st$s), mse_lmm(st$b, st$G, st$s)))
}

emml <- new_store(); emre <- new_store(); l4ml <- new_store(); l4re <- new_store()
md_ml <- numeric(0); md_re <- numeric(0)
for (s in 1:N_SIM) {
  d <- generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real)
  fr <- build_frame(d)
  e_ml <- run_hand_em(d, reml = FALSE, max_iter = 300, tolerance = 1e-8)
  e_re <- run_hand_em(d, reml = TRUE, max_iter = 500, tolerance = 1e-8)
  l_ml <- fit_lme4(fr, reml = FALSE)
  l_re <- fit_lme4(fr, reml = TRUE)
  emml <- add_store(emml, e_ml$final_beta_hat, e_ml$final_G0_hat, e_ml$final_sigma2_hat)
  emre <- add_store(emre, e_re$final_beta_hat, e_re$final_G0_hat, e_re$final_sigma2_hat)
  l4ml <- add_store(l4ml, l_ml$beta, l_ml$G0, l_ml$sigma2)
  l4re <- add_store(l4re, l_re$beta, l_re$G0, l_re$sigma2)
  md_ml <- c(md_ml, max(abs(as.numeric(e_ml$final_beta_hat) - l_ml$beta), abs(e_ml$final_G0_hat - l_ml$G0),
                        abs(e_ml$final_sigma2_hat - l_ml$sigma2)))
  md_re <- c(md_re, max(abs(as.numeric(e_re$final_beta_hat) - l_re$beta), abs(e_re$final_G0_hat - l_re$G0),
                        abs(e_re$final_sigma2_hat - l_re$sigma2)))
  if (s %% 25 == 0) cat(sprintf("  ...%d/%d\n", s, N_SIM))
}

cat("\nTrue values: beta=(1,2,-1,0.5)  G0 diagonal=(4,2)  sigma^2=1\n")
cat(sprintf("%-14s %-26s %-18s %-10s %-10s\n", "Method", "beta mean", "G0 diag mean", "sigma^2", "MSE_total"))
print_summary("hand EM ML", emml)
print_summary("lme4 ML", l4ml)
print_summary("hand EM REML", emre)
print_summary("lme4 REML", l4re)
cat(sprintf("\n[element-wise agreement] ML  : N=%d, mean max-diff=%.2e, worst=%.2e\n", N_SIM, mean(md_ml), max(md_ml)))
cat(sprintf("[element-wise agreement] REML: N=%d, mean max-diff=%.2e, worst=%.2e\n", N_SIM, mean(md_re), max(md_re)))

cat(sprintf("\n================= MCEM(M=%d) ML vs REML: %d simulation runs =================\n", MC_TIMES, N_SIM_MCEM))
mcem_ml <- new_store(); mcem_re <- new_store()
set.seed(2025)
for (s in 1:N_SIM_MCEM) {
  d <- generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real)
  mc_ml <- run_hand_em(d, reml = FALSE, is_mcem = TRUE, mc_times = MC_TIMES)
  mc_re <- run_hand_em(d, reml = TRUE, is_mcem = TRUE, mc_times = MC_TIMES)
  mcem_ml <- add_store(mcem_ml, mc_ml$final_beta_hat, mc_ml$final_G0_hat, mc_ml$final_sigma2_hat)
  mcem_re <- add_store(mcem_re, mc_re$final_beta_hat, mc_re$final_G0_hat, mc_re$final_sigma2_hat)
}
cat(sprintf("%-14s %-26s %-18s %-10s %-10s\n", "Method", "beta mean", "G0 diag mean", "sigma^2", "MSE_total"))
print_summary("MCEM ML", mcem_ml)
print_summary("MCEM REML", mcem_re)
cat("MCEM includes Monte Carlo noise; use EM rows above for digit-level library validation.\n")

sink(file.path(out_dir, "library_compare_summary.txt"))
cat("LMM: hand-coded EM/MCEM vs lme4 / nlme comparison (seed=2025, m=50)\n\n")
cat(sprintf("Single-dataset hand EM ML   vs lme4 ML  : %s\n", diff_line(em_ml, l4_ml)))
cat(sprintf("Single-dataset hand EM REML vs lme4 REML: %s\n", diff_line(em_reml, l4_reml)))
if (!is.null(ln_ml)) cat(sprintf("Single-dataset hand EM ML   vs nlme ML  : max|beta|=%.2e max|G0|=%.2e |sigma^2|=%.2e\n",
                                  max(abs(em_ml$final_beta_hat - ln_ml$beta)), max(abs(em_ml$final_G0_hat - ln_ml$G0)),
                                  abs(em_ml$final_sigma2_hat - ln_ml$sigma2)))
if (!is.null(ln_reml)) cat(sprintf("Single-dataset hand EM REML vs nlme REML: max|beta|=%.2e max|G0|=%.2e |sigma^2|=%.2e\n",
                                    max(abs(em_reml$final_beta_hat - ln_reml$beta)), max(abs(em_reml$final_G0_hat - ln_reml$G0)),
                                    abs(em_reml$final_sigma2_hat - ln_reml$sigma2)))
cat(sprintf("\n%d simulation runs (true beta=(1,2,-1,0.5) G0 diagonal=(4,2) sigma^2=1):\n", N_SIM))
sink_summary <- function(label, st) {
  cat(sprintf("  %-12s beta=(%.3f,%.3f,%.3f,%.3f) G0 diag=(%.3f,%.3f) sigma^2=%.4f MSE=%.4f\n",
              label, rowMeans(st$b)[1], rowMeans(st$b)[2], rowMeans(st$b)[3], rowMeans(st$b)[4],
              mdiag(st$G)[1], mdiag(st$G)[2], mean(st$s), mse_lmm(st$b, st$G, st$s)))
}
sink_summary("hand EM ML", emml)
sink_summary("lme4 ML", l4ml)
sink_summary("hand EM REML", emre)
sink_summary("lme4 REML", l4re)
cat(sprintf("EM vs lme4 mean max-diff: ML=%.2e, REML=%.2e\n", mean(md_ml), mean(md_re)))
cat(sprintf("\nMCEM(M=%d), %d simulation runs:\n", MC_TIMES, N_SIM_MCEM))
sink_summary("MCEM ML", mcem_ml)
sink_summary("MCEM REML", mcem_re)
cat("\nInterpretation: ML uses divisors m and n, which match lme4/nlme ML but can underestimate variance components. REML propagates fixed-effect uncertainty and should be reported alongside ML when the ML variance estimates are visibly biased.\n")
sink()
cat("\n[done] Summary written to", file.path(out_dir, "library_compare_summary.txt"), "\n")
