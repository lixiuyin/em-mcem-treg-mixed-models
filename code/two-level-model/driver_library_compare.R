# Two-level linear model: comparison and evaluation of hand-coded EM/MCEM vs library (lme4)
# ---------------------------------------------------------------------------
# Purpose: Compare the EM/MCEM estimates from the hand-coded implementation in paper chapter 4
#   against the R ecosystem gold standard lme4::lmer on the **same batch of simulated data**, to
#   (1) verify the correctness of the hand-coded implementation, and
#   (2) quantify the downward bias of ML variance-component estimates and the REML correction.
#
# Model mapping to lme4:
#   Level 1: y_ij = x_ij' beta_j + epsilon,  Level 2: beta_j = W_j gamma + mu_j, mu_j ~ N(0, D)
#   => y_ij = (x_ij' W_j) gamma + x_ij' mu_j + epsilon
#   Fixed design = rows (x_ij' W_j) ((p+1)(q+1) columns); random design = x_ij (p+1 columns,
#   random coefficients, grouped by group indicator).
#   lme4 formula: y ~ 0 + FX1+...+FX_{12} + (0 + R1+R2+R3 | group)
#
# Estimator alignment:
#   - Hand-coded EM(ML)   <-> lmer(REML=FALSE)
#   - Hand-coded EM(REML) <-> lmer(REML=TRUE)    (REML: see README section 4.5.2)
# ---------------------------------------------------------------------------
suppressMessages({library(this.path); library(lme4)})
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

# Append a 3x3 matrix to the third dimension of a 3x3xn array (avoids depending on the abind package)
abind3 <- function(a, m) array(c(a, m), dim = c(3, 3, dim(a)[3] + 1))

set.seed(2025)
N_SIM <- as.integer(Sys.getenv("N_SIM", "200"))   # number of simulations for EM comparison
N_SIM_MCEM <- as.integer(Sys.getenv("N_SIM_MCEM", "30"))  # MCEM is slower, so fewer simulations are used
MC_TIMES <- 50

J <- 20; p <- 2; q <- 3
gamma_real <- matrix(seq(1, (p + 1) * (q + 1)), ncol = 1)
D_real <- matrix(c(5, 1, 3, 1, 6, 2, 3, 2, 7), ncol = 3, byrow = TRUE)
sigma2_real <- 3
gamma_0 <- matrix(rep(0.1, (p + 1) * (q + 1)), ncol = 1)
D_0 <- diag(3); sigma2_0 <- 0.1

out_dir <- this.dir()
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 5e5),
                    check.conv.singular = .makeCC("ignore", tol = 1e-4))
RX_FORM <- as.formula(paste0("y ~ 0 + ", paste0("X", 1:((p + 1) * (q + 1)), collapse = " + "),
                             " + (0 + R1 + R2 + R3 | group)"))

# Build lme4 long-format data frame from the internal data object (fixed design X_j*W_j, random design X_j)
build_lme4_frame <- function(data) {
  J <- data$J; N <- data$sample_size
  pg <- (data$p + 1) * (data$q + 1); k <- data$p + 1
  FX <- matrix(0, N, pg); RX <- matrix(0, N, k); grp <- integer(N); yv <- numeric(N); pos <- 0
  for (j in 1:J) {
    Xj <- get_X_j(data, j); Wj <- get_W_j(data, j); yj <- get_y_j(data, j)
    nj <- nrow(Xj); idx <- (pos + 1):(pos + nj)
    FX[idx, ] <- Xj %*% Wj; RX[idx, ] <- Xj; yv[idx] <- yj; grp[idx] <- j; pos <- pos + nj
  }
  d <- data.frame(y = yv, FX); names(d) <- c("y", paste0("X", 1:pg))
  d$R1 <- RX[, 1]; d$R2 <- RX[, 2]; d$R3 <- RX[, 3]; d$group <- factor(grp)
  d
}

fit_lme4 <- function(frame, reml) {
  m <- lmer(RX_FORM, data = frame, REML = reml, control = ctrl)
  list(gamma = as.numeric(fixef(m)), D = matrix(VarCorr(m)$group, 3, 3),
       sigma2 = sigma(m)^2, loglik = as.numeric(logLik(m)))
}

# ============================ Element-wise comparison on a single dataset ============================
cat("================= Element-wise comparison on a single dataset (verifying implementation correctness) =================\n")
data1 <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)
frame1 <- build_lme4_frame(data1)

options(hlm_reml = FALSE)
em_ml <- run_single_em_or_mcem(data1, gamma_0, D_0, sigma2_0, max_iter = 400, tolerance = 1e-9)
lme_ml <- fit_lme4(frame1, reml = FALSE)
options(hlm_reml = TRUE)
em_reml <- run_single_em_or_mcem(data1, gamma_0, D_0, sigma2_0, max_iter = 400, tolerance = 1e-9)
lme_reml <- fit_lme4(frame1, reml = TRUE)
options(hlm_reml = FALSE)

report_pair <- function(tag, em_gamma, em_D, em_s2, em_ll, lm) {
  cat(sprintf("\n--- %s ---\n", tag))
  cat("  gamma hand-coded:", round(as.numeric(em_gamma), 3), "\n")
  cat("  gamma lme4      :", round(lm$gamma, 3), "\n")
  cat("  D diagonal hand-coded:", round(diag(em_D), 4), "  lme4:", round(diag(lm$D), 4), "\n")
  cat(sprintf("  sigma^2 hand-coded=%.4f  lme4=%.4f\n", em_s2, lm$sigma2))
  cat(sprintf("  [diff] max|gamma|=%.2e  max|D|=%.2e  |sigma^2|=%.2e  |logLik|=%.2e\n",
              max(abs(as.numeric(em_gamma) - lm$gamma)), max(abs(em_D - lm$D)),
              abs(em_s2 - lm$sigma2), abs(em_ll - lm$loglik)))
}
report_pair("ML: hand-coded EM(ML) vs lme4(REML=FALSE)", em_ml$final_gamma_hat, em_ml$final_D_hat,
            em_ml$final_sigma2_hat, em_ml$loglik, lme_ml)
report_pair("REML: hand-coded EM(REML) vs lme4(REML=TRUE)", em_reml$final_gamma_hat, em_reml$final_D_hat,
            em_reml$final_sigma2_hat, em_reml$loglik, lme_reml)

# ============================ N simulations: bias / MSE evaluation ============================
cat(sprintf("\n================= %d simulations aggregated (bias / MSE evaluation) =================\n", N_SIM))
acc <- function() list(g = matrix(0, 12, 0), D = array(0, c(3, 3, 0)), s2 = numeric(0))
res <- list(em_ml = acc(), em_reml = acc(), lme_ml = acc(), lme_reml = acc())
maxdiff_ml <- maxdiff_reml <- numeric(0)

for (s in 1:N_SIM) {
  d <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)
  fr <- build_lme4_frame(d)
  options(hlm_reml = FALSE)
  e1 <- run_single_em_or_mcem(d, gamma_0, D_0, sigma2_0, max_iter = 200, tolerance = 1e-7)
  options(hlm_reml = TRUE)
  e2 <- run_single_em_or_mcem(d, gamma_0, D_0, sigma2_0, max_iter = 200, tolerance = 1e-7)
  options(hlm_reml = FALSE)
  l1 <- fit_lme4(fr, FALSE); l2 <- fit_lme4(fr, TRUE)
  res$em_ml$g   <- cbind(res$em_ml$g, as.numeric(e1$final_gamma_hat))
  res$em_reml$g <- cbind(res$em_reml$g, as.numeric(e2$final_gamma_hat))
  res$lme_ml$g  <- cbind(res$lme_ml$g, l1$gamma)
  res$lme_reml$g<- cbind(res$lme_reml$g, l2$gamma)
  res$em_ml$D   <- abind3(res$em_ml$D, e1$final_D_hat)
  res$em_reml$D <- abind3(res$em_reml$D, e2$final_D_hat)
  res$lme_ml$D  <- abind3(res$lme_ml$D, l1$D)
  res$lme_reml$D<- abind3(res$lme_reml$D, l2$D)
  res$em_ml$s2   <- c(res$em_ml$s2, e1$final_sigma2_hat)
  res$em_reml$s2 <- c(res$em_reml$s2, e2$final_sigma2_hat)
  res$lme_ml$s2  <- c(res$lme_ml$s2, l1$sigma2)
  res$lme_reml$s2<- c(res$lme_reml$s2, l2$sigma2)
  maxdiff_ml   <- c(maxdiff_ml,   max(abs(as.numeric(e1$final_gamma_hat) - l1$gamma),
                                      abs(e1$final_D_hat - l1$D), abs(e1$final_sigma2_hat - l1$sigma2)))
  maxdiff_reml <- c(maxdiff_reml, max(abs(as.numeric(e2$final_gamma_hat) - l2$gamma),
                                      abs(e2$final_D_hat - l2$D), abs(e2$final_sigma2_hat - l2$sigma2)))
  if (s %% 25 == 0) cat(sprintf("  ...%d/%d\n", s, N_SIM))
}

# Per-simulation MSE (norm-based, consistent with paper chapter 4 get_mse), then averaged over simulations
mse_over_sims <- function(r) {
  n <- ncol(r$g); mg <- mD <- ms <- numeric(n)
  for (s in 1:n) {
    mg[s] <- norm(matrix(r$g[, s], ncol = 1) - gamma_real, "2")
    mD[s] <- norm(r$D[, , s] - D_real, "F")
    ms[s] <- abs(r$s2[s] - sigma2_real)
  }
  c(gamma = mean(mg), D = mean(mD), sigma2 = mean(ms), total = mean(mg + mD + ms))
}
mean_diag <- function(r) rowMeans(apply(r$D, 3, diag))
mean_s2 <- function(r) mean(r$s2)

methods <- c("em_ml", "em_reml", "lme_ml", "lme_reml")
labels <- c("hand-coded EM(ML)", "hand-coded EM(REML)", "lme4(ML)", "lme4(REML)")
cat("\nTrue values: D diagonal=(5,6,7)  sigma^2=3\n")
cat(sprintf("\n%-22s %-22s %-9s %-10s %-10s %-10s\n", "Method", "D diagonal mean", "sigma^2 mean", "MSE_gamma", "MSE_D", "MSE_total"))
for (i in seq_along(methods)) {
  r <- res[[methods[i]]]; m <- mse_over_sims(r)
  cat(sprintf("%-22s (%-5.3f,%-5.3f,%-5.3f)   %-9.4f %-10.4f %-10.4f %-10.4f\n",
              labels[i], mean_diag(r)[1], mean_diag(r)[2], mean_diag(r)[3], mean_s2(r),
              m["gamma"], m["D"], m["total"]))
}
cat(sprintf("\n[Element-wise] hand-coded EM(ML)  vs lme4(ML)  : N=%d sims avg max-diff=%.2e, max=%.2e\n",
            N_SIM, mean(maxdiff_ml), max(maxdiff_ml)))
cat(sprintf("[Element-wise] hand-coded EM(REML) vs lme4(REML): N=%d sims avg max-diff=%.2e, max=%.2e\n",
            N_SIM, mean(maxdiff_reml), max(maxdiff_reml)))

# ============================ MCEM vs lme4 (small number of simulations) ============================
cat(sprintf("\n================= MCEM(M=%d) vs lme4: %d simulations =================\n", MC_TIMES, N_SIM_MCEM))
options(hlm_reml = TRUE)
mc_g <- matrix(0, 12, 0); mc_D <- array(0, c(3, 3, 0)); mc_s2 <- numeric(0); mc_diff <- numeric(0)
set.seed(2025)
for (s in 1:N_SIM_MCEM) {
  d <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)
  mc <- run_single_em_or_mcem(d, gamma_0, D_0, sigma2_0, is_mcem = TRUE, mc_times = MC_TIMES,
                              max_iter = 30, tolerance = 1e-6)
  l2 <- fit_lme4(build_lme4_frame(d), TRUE)
  mc_g <- cbind(mc_g, as.numeric(mc$final_gamma_hat)); mc_D <- abind3(mc_D, mc$final_D_hat)
  mc_s2 <- c(mc_s2, mc$final_sigma2_hat)
  mc_diff <- c(mc_diff, max(abs(as.numeric(mc$final_gamma_hat) - l2$gamma)))
}
options(hlm_reml = FALSE)
mc_res <- list(g = mc_g, D = mc_D, s2 = mc_s2)
mmc <- mse_over_sims(mc_res)
cat(sprintf("MCEM(REML) D diagonal mean=(%.3f,%.3f,%.3f) sigma^2=%.4f  MSE_total=%.4f\n",
            mean_diag(mc_res)[1], mean_diag(mc_res)[2], mean_diag(mc_res)[3], mean(mc_s2), mmc["total"]))
cat(sprintf("MCEM vs lme4(REML) gamma average max-diff=%.4f (includes Monte Carlo noise, hence larger than EM element-wise agreement)\n", mean(mc_diff)))

# ============================ Save summary ============================
sink(file.path(out_dir, "library_compare_summary.txt"))
cat("Two-level linear model: hand-coded EM/MCEM vs lme4 comparison (seed=2025, J=20)\n\n")
cat(sprintf("Element-wise comparison on a single dataset:\n  EM(ML)  vs lme4(ML)  max|gamma|=%.2e max|D|=%.2e |sigma^2|=%.2e |logLik|=%.2e\n",
            max(abs(as.numeric(em_ml$final_gamma_hat) - lme_ml$gamma)),
            max(abs(em_ml$final_D_hat - lme_ml$D)), abs(em_ml$final_sigma2_hat - lme_ml$sigma2),
            abs(em_ml$loglik - lme_ml$loglik)))
cat(sprintf("  EM(REML) vs lme4(REML) max|gamma|=%.2e max|D|=%.2e |sigma^2|=%.2e |logLik|=%.2e\n\n",
            max(abs(as.numeric(em_reml$final_gamma_hat) - lme_reml$gamma)),
            max(abs(em_reml$final_D_hat - lme_reml$D)), abs(em_reml$final_sigma2_hat - lme_reml$sigma2),
            abs(em_reml$loglik - lme_reml$loglik)))
cat(sprintf("%d simulations aggregated (true D diagonal=(5,6,7) sigma^2=3):\n", N_SIM))
for (i in seq_along(methods)) {
  r <- res[[methods[i]]]; m <- mse_over_sims(r)
  cat(sprintf("  %-22s D diagonal=(%.3f,%.3f,%.3f) sigma^2=%.4f MSE_total=%.4f\n",
              labels[i], mean_diag(r)[1], mean_diag(r)[2], mean_diag(r)[3], mean_s2(r), m["total"]))
}
cat(sprintf("\nEM(ML) vs lme4(ML)  avg max-diff=%.2e\nEM(REML) vs lme4(REML) avg max-diff=%.2e\n",
            mean(maxdiff_ml), mean(maxdiff_reml)))
sink()
cat("\n[done] Summary written to", file.path(out_dir, "library_compare_summary.txt"), "\n")
