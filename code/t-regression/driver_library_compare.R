# Student's t-regression: comparison and evaluation of hand-coded EM/MCEM against library routines
# ---------------------------------------------------------------------------
# Reference methods (degrees of freedom nu assumed known and fixed at true value):
#   1) optim maximizing the marginal t log-likelihood  -- pure MLE (no prior), reference for beta and sigma^2_MLE.
#   2) optim maximizing the marginal t log-posterior   -- with prior pi proportional to 1/sigma^2, should match hand-coded EM MAP estimates exactly.
#   3) MASS::rlm                                       -- Huber/Tukey M-estimation (different loss), reference for beta direction.
#   4) hett::tlm                                       -- third-party t-regression MLE library, reference for beta and sigma^2.
#
# Important note on parameterization: the hand-coded EM uses prior pi(beta, sigma^2) proportional to 1/sigma^2,
# so the sigma^2 update divides by n+2 (MAP); pure MLE divides by n.
# Therefore sigma^2_MLE approximately equals sigma^2_MAP * (n+2)/n. Both are reported to avoid confusion.
# ---------------------------------------------------------------------------
suppressMessages({library(this.path); library(MASS)})
HAS_HETT <- requireNamespace("hett", quietly = TRUE)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

set.seed(2025)
N_SIM <- as.integer(Sys.getenv("N_SIM", "500"))
N_SIM_MCEM <- as.integer(Sys.getenv("N_SIM_MCEM", "50"))
MC_TIMES <- 100
n <- 500; p <- 2; t_df <- 10; t_sigma2_real <- 0.4; beta_real <- c(2, 3, 5)
beta_0 <- c(0, 0, 0); sigma2_0 <- 0.1
out_dir <- this.dir()

# Marginal t log-likelihood (pure MLE objective)
t_loglik <- function(X, y, beta, sigma2, nu) {
  r <- as.numeric(y - X %*% beta); s <- sqrt(sigma2)
  sum(dt(r / s, df = nu, log = TRUE) - log(s))
}
fit_optim_mle <- function(X, y, nu, b0, s20) {
  negll <- function(par) -t_loglik(X, y, par[1:length(b0)], exp(par[length(b0) + 1]), nu)
  o <- optim(c(b0, log(s20)), negll, method = "BFGS", control = list(maxit = 800, reltol = 1e-12))
  list(beta = o$par[1:length(b0)], sigma2 = exp(o$par[length(b0) + 1]))
}
fit_optim_map <- function(X, y, nu, b0, s20) {     # with 1/sigma^2 prior, reference for hand-coded EM MAP
  neglp <- function(par) -(t_loglik(X, y, par[1:length(b0)], exp(par[length(b0) + 1]), nu) - log(exp(par[length(b0) + 1])))
  o <- optim(c(b0, log(s20)), neglp, method = "BFGS", control = list(maxit = 800, reltol = 1e-12))
  list(beta = o$par[1:length(b0)], sigma2 = exp(o$par[length(b0) + 1]))
}
fit_hett <- function(X, y, nu) {
  if (!HAS_HETT) return(NULL)
  d <- data.frame(y = y, X[, -1, drop = FALSE]); names(d) <- c("y", paste0("x", 1:(ncol(X) - 1)))
  form <- as.formula(paste0("y ~ ", paste0("x", 1:(ncol(X) - 1), collapse = " + ")))
  fit <- tryCatch(hett::tlm(form, sform = ~1, data = d, estDof = FALSE, start = list(dof = nu)),
                  error = function(e) NULL)
  if (is.null(fit)) return(NULL)
  list(beta = as.numeric(fit$loc.fit$coefficients),
       sigma2 = exp(as.numeric(fit$scale.fit$coefficients)[1]))
}

# ============================ Single-dataset point-by-point comparison ============================
cat("================= Single-dataset point-by-point comparison (implementation verification) =================\n")
d1 <- generate_t_simulated_data(n, p, t_df, 0, t_sigma2_real, beta_real)
X <- d1$X; y <- d1$y_real
em <- run_single_em_or_mcem(X, y, t_df, beta_0, sigma2_0, max_iter = 300, tolerance = 1e-10)
s2_em_mle <- em$final_sigma2_hat * (n + 2) / n          # convert MAP sigma^2 to MLE scale
o_mle <- fit_optim_mle(X, y, t_df, beta_0, t_sigma2_real)
o_map <- fit_optim_map(X, y, t_df, beta_0, t_sigma2_real)
rl <- rlm(y ~ X[, 2] + X[, 3]); beta_rlm <- as.numeric(coef(rl))
he <- fit_hett(X, y, t_df)
cat("beta hand-coded EM:", round(em$final_beta_hat, 5), "\n")
cat("beta optim MLE    :", round(o_mle$beta, 5), "\n")
cat("beta optim MAP    :", round(o_map$beta, 5), "\n")
cat("beta rlm          :", round(beta_rlm, 5), "\n")
if (!is.null(he)) cat("beta hett         :", round(he$beta, 5), "\n")
cat(sprintf("sigma^2: hand-coded MAP=%.5f  converted MLE=%.5f  optimMLE=%.5f  optimMAP=%.5f%s\n",
            em$final_sigma2_hat, s2_em_mle, o_mle$sigma2, o_map$sigma2,
            if (!is.null(he)) sprintf("  hett=%.5f", he$sigma2) else ""))
cat(sprintf("[diff] EM(MAP) vs optim(MAP): max|beta|=%.2e |sigma^2|=%.2e   <-- same parameterization, should match exactly\n",
            max(abs(em$final_beta_hat - o_map$beta)), abs(em$final_sigma2_hat - o_map$sigma2)))
cat(sprintf("[diff] EM converted MLE vs optim(MLE): max|beta|=%.2e |sigma^2|=%.2e\n",
            max(abs(em$final_beta_hat - o_mle$beta)), abs(s2_em_mle - o_mle$sigma2)))
cat(sprintf("[corr] cor(beta_EM, beta_rlm)=%.5f%s\n", cor(em$final_beta_hat, beta_rlm),
            if (!is.null(he)) sprintf("  cor(beta_EM, beta_hett)=%.5f", cor(em$final_beta_hat, he$beta)) else ""))

# ============================ N simulations: bias/MSE evaluation ============================
cat(sprintf("\n================= %d simulations aggregated (bias/MSE evaluation) =================\n", N_SIM))
# MSE definition matches paper Chapter 3: sum of squared errors for beta + squared error for sigma^2
sse <- function(b, s2) sum((b - beta_real)^2) + (s2 - t_sigma2_real)^2
acc_b <- function() matrix(0, 3, 0)
EM <- list(b = acc_b(), s2 = numeric(0), mse = numeric(0))
OM <- list(b = acc_b(), s2 = numeric(0), mse = numeric(0))   # optim MLE
RL <- list(b = acc_b(), mse = numeric(0))
HE <- if (HAS_HETT) list(b = acc_b(), s2 = numeric(0), mse = numeric(0)) else NULL
dmap <- dmle <- numeric(0)
for (s in 1:N_SIM) {
  d <- generate_t_simulated_data(n, p, t_df, 0, t_sigma2_real, beta_real)
  X <- d$X; y <- d$y_real
  e <- run_single_em_or_mcem(X, y, t_df, beta_0, sigma2_0, max_iter = 300, tolerance = 1e-9)
  om <- fit_optim_mle(X, y, t_df, beta_0, t_sigma2_real)
  omap <- fit_optim_map(X, y, t_df, beta_0, t_sigma2_real)
  rl <- rlm(y ~ X[, 2] + X[, 3]); br <- as.numeric(coef(rl))
  EM$b <- cbind(EM$b, e$final_beta_hat); EM$s2 <- c(EM$s2, e$final_sigma2_hat); EM$mse <- c(EM$mse, sse(e$final_beta_hat, e$final_sigma2_hat))
  OM$b <- cbind(OM$b, om$beta); OM$s2 <- c(OM$s2, om$sigma2); OM$mse <- c(OM$mse, sse(om$beta, om$sigma2))
  RL$b <- cbind(RL$b, br); RL$mse <- c(RL$mse, sum((br - beta_real)^2))
  dmap <- c(dmap, max(abs(e$final_beta_hat - omap$beta)))
  dmle <- c(dmle, max(abs(e$final_beta_hat * 1 - om$beta)))
  if (HAS_HETT) { he <- fit_hett(X, y, t_df); if (!is.null(he)) { HE$b <- cbind(HE$b, he$beta); HE$s2 <- c(HE$s2, he$sigma2); HE$mse <- c(HE$mse, sse(he$beta, he$sigma2)) } }
  if (s %% 50 == 0) cat(sprintf("  ...%d/%d\n", s, N_SIM))
}
cat("\nTrue values: beta=(2,3,5)  sigma^2=0.4\n")
cat(sprintf("%-20s %-26s %-10s %-12s\n", "Method", "beta mean", "sigma^2 mean", "MSE_total"))
cat(sprintf("%-20s (%6.4f,%6.4f,%6.4f) %-10.4f %-12.3e\n", "hand-coded EM(MAP)", rowMeans(EM$b)[1], rowMeans(EM$b)[2], rowMeans(EM$b)[3], mean(EM$s2), mean(EM$mse)))
cat(sprintf("%-20s (%6.4f,%6.4f,%6.4f) %-10.4f %-12.3e\n", "optim(MLE)", rowMeans(OM$b)[1], rowMeans(OM$b)[2], rowMeans(OM$b)[3], mean(OM$s2), mean(OM$mse)))
if (HAS_HETT && ncol(HE$b) > 0) cat(sprintf("%-20s (%6.4f,%6.4f,%6.4f) %-10.4f %-12.3e\n", "hett::tlm", rowMeans(HE$b)[1], rowMeans(HE$b)[2], rowMeans(HE$b)[3], mean(HE$s2), mean(HE$mse)))
cat(sprintf("%-20s (%6.4f,%6.4f,%6.4f) %-10s %-12.3e\n", "MASS::rlm", rowMeans(RL$b)[1], rowMeans(RL$b)[2], rowMeans(RL$b)[3], "(no sigma^2)", mean(RL$mse)))
cat(sprintf("\n[point-by-point] hand-coded EM(MAP) vs optim(MAP): mean max|beta|=%.2e, max=%.2e\n", mean(dmap), max(dmap)))
cat(sprintf("[point-by-point] hand-coded EM(beta) vs optim(MLE): mean max|beta|=%.2e (beta unaffected by prior, should agree)\n", mean(dmle)))

# ============================ MCEM vs library routines (fewer simulations) ============================
cat(sprintf("\n================= MCEM(M=%d) vs optim/hett: %d simulations =================\n", MC_TIMES, N_SIM_MCEM))
MCb <- acc_b(); MCs <- numeric(0); MCmse <- numeric(0); MCd <- numeric(0)
set.seed(2025)
for (s in 1:N_SIM_MCEM) {
  d <- generate_t_simulated_data(n, p, t_df, 0, t_sigma2_real, beta_real)
  X <- d$X; y <- d$y_real
  mc <- run_single_em_or_mcem(X, y, t_df, beta_0, sigma2_0, is_mcem = TRUE, mc_times = MC_TIMES, max_iter = 100, tolerance = 1e-7)
  om <- fit_optim_mle(X, y, t_df, beta_0, t_sigma2_real)
  MCb <- cbind(MCb, mc$final_beta_hat); MCs <- c(MCs, mc$final_sigma2_hat); MCmse <- c(MCmse, sse(mc$final_beta_hat, mc$final_sigma2_hat))
  MCd <- c(MCd, max(abs(mc$final_beta_hat - om$beta)))
}
cat(sprintf("MCEM beta mean=(%.4f,%.4f,%.4f) sigma^2=%.4f MSE_total=%.3e\n",
            rowMeans(MCb)[1], rowMeans(MCb)[2], rowMeans(MCb)[3], mean(MCs), mean(MCmse)))
cat(sprintf("MCEM vs optim(MLE): mean max|beta|=%.4f (includes Monte Carlo noise)\n", mean(MCd)))

# ============================ Save summary ============================
sink(file.path(out_dir, "library_compare_summary.txt"))
cat("Student-t regression: hand-coded EM/MCEM vs optim / rlm / hett (seed=2025, n=500, nu=10)\n\n")
cat(sprintf("Single-dataset point-by-point: EM(MAP) vs optim(MAP) max|beta|=%.2e |sigma^2|=%.2e\n",
            max(abs(em$final_beta_hat - o_map$beta)), abs(em$final_sigma2_hat - o_map$sigma2)))
cat(sprintf("                               EM converted MLE vs optim(MLE) max|beta|=%.2e |sigma^2|=%.2e\n\n",
            max(abs(em$final_beta_hat - o_mle$beta)), abs(s2_em_mle - o_mle$sigma2)))
cat(sprintf("%d simulations (true beta=(2,3,5) sigma^2=0.4):\n", N_SIM))
cat(sprintf("  hand-coded EM(MAP) beta=(%.4f,%.4f,%.4f) sigma^2=%.4f MSE=%.3e\n", rowMeans(EM$b)[1], rowMeans(EM$b)[2], rowMeans(EM$b)[3], mean(EM$s2), mean(EM$mse)))
cat(sprintf("  optim(MLE)         beta=(%.4f,%.4f,%.4f) sigma^2=%.4f MSE=%.3e\n", rowMeans(OM$b)[1], rowMeans(OM$b)[2], rowMeans(OM$b)[3], mean(OM$s2), mean(OM$mse)))
if (HAS_HETT && ncol(HE$b) > 0) cat(sprintf("  hett::tlm          beta=(%.4f,%.4f,%.4f) sigma^2=%.4f MSE=%.3e\n", rowMeans(HE$b)[1], rowMeans(HE$b)[2], rowMeans(HE$b)[3], mean(HE$s2), mean(HE$mse)))
cat(sprintf("  MASS::rlm          beta=(%.4f,%.4f,%.4f) (M-estimation, no sigma^2)\n", rowMeans(RL$b)[1], rowMeans(RL$b)[2], rowMeans(RL$b)[3]))
cat(sprintf("\nEM(MAP) vs optim(MAP): mean max|beta|=%.2e\n", mean(dmap)))
sink()
cat("\n[done] Summary written to", file.path(out_dir, "library_compare_summary.txt"), "\n")
