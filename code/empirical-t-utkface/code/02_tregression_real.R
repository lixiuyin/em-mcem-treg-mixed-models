# -*- coding: utf-8 -*-
# UTKFace age ~ deep embeddings: Student-t regression empirical study.
# Algorithm matches Chapter 3 of the thesis exactly:
#   E-step weights eq.(3-14): gamma_i = sigma2*(nu+1) / (r_i^2 + sigma2*nu)
#   M-step weighted least squares eq.(3-19), sigma^2 eq.(3-22)
#     (denominator n+2 comes from prior pi proportional to 1/sigma^2).
# Implementation uses row-scaling (w*X) instead of diag(w)%*%X to avoid
# an n x n dense matrix, enabling n=8000.
# Real-data extension: error degrees of freedom nu unknown ->
#   profile marginal likelihood selects nu* on a grid.
suppressMessages({library(MASS)})
ROOT <- dirname(dirname(this.path::this.dir()))   # derived from script location, points to code/
OUT  <- file.path(ROOT, "empirical-t-utkface/output"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

df <- read.csv(file.path(ROOT, "empirical-t-utkface/data/utkface_t_regression.csv"))
pc_cols <- grep("^PC", names(df), value = TRUE)
y <- df$age
X <- as.matrix(cbind(Intercept = 1, df[, pc_cols]))
n <- nrow(X); P <- ncol(X)
cat(sprintf("Data: n=%d, predictors (incl. intercept)=%d, age[min/med/max]=%d/%d/%d\n",
            n, P, min(y), as.integer(median(y)), max(y)))

# ---- Chapter 3 EM / MCEM (efficient implementation, mathematically equivalent) ----
treg_em <- function(X, y, nu, beta0, sigma20, max_iter = 300, tol = 1e-7,
                    is_mcem = FALSE, M = 100, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  beta <- beta0; s2 <- sigma20; N <- length(y)
  bh <- matrix(NA_real_, max_iter + 1, ncol(X)); s2h <- numeric(max_iter + 1)
  bh[1, ] <- beta; s2h[1] <- s2; it <- 0
  for (k in 1:max_iter) {
    r <- as.numeric(y - X %*% beta)
    if (is_mcem) {                       # E-step: sample z_i|y_i ~ Gamma((nu+1)/2, r^2/(2s2)+nu/2) and average
      a <- (nu + 1) / 2; lam <- r^2 / (2 * s2) + nu / 2
      draws <- matrix(rgamma(N * M, shape = a, rate = rep(lam, each = M)), nrow = M)
      w <- colMeans(draws)
    } else {                             # E-step: analytic weights eq.(3-14)
      w <- s2 * (nu + 1) / (r^2 + s2 * nu)
    }
    bprev <- beta; sprev <- s2
    beta <- solve(crossprod(X, X * w), crossprod(X, w * y))   # eq.(3-19) WLS
    r2 <- as.numeric(y - X %*% beta)
    s2 <- sum(w * r2^2) / (N + 2)                              # eq.(3-22)
    it <- it + 1; bh[it + 1, ] <- beta; s2h[it + 1] <- s2
    if (max(sqrt(sum((beta - bprev)^2)), abs(s2 - sprev)) < tol) break
  }
  list(final_beta_hat = as.numeric(beta), final_sigma2_hat = s2,
       beta_history = bh[1:(it + 1), , drop = FALSE],
       sigma2_history = s2h[1:(it + 1)], final_iter_count = it)
}

t_loglik <- function(beta, sigma2, nu) {   # observed-data (marginal) t-regression log-likelihood
  r <- as.numeric(y - X %*% beta); s <- sqrt(sigma2)
  sum(dt(r / s, df = nu, log = TRUE) - log(s))
}

# ---- OLS baseline ----
ols <- lm(y ~ ., data = data.frame(y = y, df[, pc_cols]))
beta_ols <- coef(ols); sigma2_ols <- summary(ols)$sigma^2
ll_normal <- sum(dnorm(residuals(ols), 0, sqrt(sigma2_ols), log = TRUE))
cat(sprintf("OLS: sigma=%.3f, R^2=%.3f, loglik=%.1f\n",
            sqrt(sigma2_ols), summary(ols)$r.squared, ll_normal))

# ---- Profile marginal likelihood for nu: grid search ----
nu_grid <- c(1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50)
t0 <- Sys.time()
prof <- sapply(nu_grid, function(nu) {
  r <- treg_em(X, y, nu, rep(0, P), sigma2_ols, max_iter = 200, tol = 1e-8)
  t_loglik(r$final_beta_hat, r$final_sigma2_hat, nu)
})
nu_star <- nu_grid[which.max(prof)]
cat(sprintf("Profile likelihood: nu*=%g (loglik=%.1f), grid elapsed %.1fs\n",
            nu_star, max(prof), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
cat(sprintf("t(nu*) log-likelihood improvement over normal (OLS) = %.1f\n", max(prof) - ll_normal))

# ---- At nu*: EM vs MCEM (varying M) vs OLS ----
em <- treg_em(X, y, nu_star, rep(0, P), sigma2_ols, max_iter = 300, tol = 1e-9)
cat(sprintf("EM converged in %d iterations, sigma=%.3f\n", em$final_iter_count, sqrt(em$final_sigma2_hat)))
mc <- list()
for (M in c(20, 50, 200))
  mc[[as.character(M)]] <- treg_em(X, y, nu_star, rep(0, P), sigma2_ols,
                                   max_iter = 300, tol = 1e-6, is_mcem = TRUE, M = M, seed = 20250529)

# ---- Independent validation 1: optim directly maximizes marginal t log-likelihood ----
negll <- function(par) -t_loglik(par[1:P], exp(par[P + 1]), nu_star)
opt <- optim(c(beta_ols, log(sigma2_ols)), negll, method = "BFGS",
             control = list(maxit = 800, reltol = 1e-12))
beta_opt <- opt$par[1:P]; sigma2_opt <- exp(opt$par[P + 1])
maxabs <- max(abs(em$final_beta_hat - beta_opt))
cat(sprintf("[Validation 1] EM vs optim marginal MLE: max abs coef diff=%.5f, sigma2 diff=%.5f\n",
            maxabs, abs(em$final_sigma2_hat - sigma2_opt)))
# ---- Independent validation 2: MASS::rlm (M-estimation, different loss, should be highly concordant) ----
rl <- tryCatch(rlm(y ~ ., data = data.frame(y = y, df[, pc_cols])), error = function(e) NULL)
if (!is.null(rl)) cat(sprintf("[Validation 2] Correlation of EM and MASS::rlm coefficients=%.4f\n",
                              cor(em$final_beta_hat, as.numeric(coef(rl)))))

# ---- E-step weights gamma_i (final residuals at nu*) ----
r_em <- as.numeric(y - X %*% em$final_beta_hat)
gamma_i <- em$final_sigma2_hat * (nu_star + 1) / (r_em^2 + em$final_sigma2_hat * nu_star)
ord <- order(gamma_i)
cat(sprintf("gamma_i: min=%.3f med=%.3f max=%.3f; ages of 10 most down-weighted samples: %s\n",
            min(gamma_i), median(gamma_i), max(gamma_i), paste(df$age[ord[1:10]], collapse = ",")))
cat(sprintf("Among the most down-weighted 10%% of samples, proportion with age>=60 = %.1f%% (full sample = %.1f%%)\n",
            100 * mean(df$age[ord[1:round(0.1 * n)]] >= 60), 100 * mean(df$age >= 60)))

# ---- Comparison table ----
show <- 1:6
cmp <- data.frame(term = c("Intercept", paste0("PC", 1:5)),
                  OLS = round(beta_ols[show], 4),
                  EM_t = round(em$final_beta_hat[show], 4),
                  MCEM_M20 = round(mc[["20"]]$final_beta_hat[show], 4),
                  MCEM_M200 = round(mc[["200"]]$final_beta_hat[show], 4),
                  optim = round(beta_opt[show], 4))
cat("\n===== Regression coefficient comparison (intercept + first 5 PCs) =====\n"); print(cmp, row.names = FALSE)
cat(sprintf("Median absolute residual: OLS=%.3f, t-EM=%.3f\n", median(abs(residuals(ols))), median(abs(r_em))))

# ---- Figure 1: OLS residual normal QQ plot ----
png(file.path(OUT, "71_qq_normal.png"), 1400, 1000, res = 200)
qqnorm(residuals(ols), pch = 16, cex = .3, col = "#3182bd", main = "OLS Residuals: Normal QQ Plot (tail deviation indicates heavy tails)")
qqline(residuals(ols), col = "red", lwd = 2); dev.off()
# ---- Figure 2: profile likelihood ----
png(file.path(OUT, "72_profile_nu.png"), 1400, 1000, res = 200)
plot(nu_grid, prof, type = "o", pch = 16, col = "#756bb1", log = "x",
     xlab = "Degrees of freedom nu (log scale)", ylab = "Marginal log-likelihood",
     main = "Student-t Regression: Profile Likelihood for Degrees of Freedom")
abline(v = nu_star, lty = 2, col = "red"); abline(h = ll_normal, lty = 3, col = "gray40")
legend("bottomright", c(sprintf("nu*=%g", nu_star), "Normal baseline"), col = c("red", "gray40"),
       lty = c(2, 3), bty = "n"); dev.off()
# ---- Figure 3: gamma_i down-weighting ----
png(file.path(OUT, "73_weights.png"), 2200, 950, res = 200); par(mfrow = c(1, 2))
plot(abs(r_em), gamma_i, pch = 16, cex = .3, col = "#31a354", xlab = "|Residual|",
     ylab = expression(gamma[i]), cex.main = 0.95, main = "E-step weights vs |residual|")
plot(df$age, gamma_i, pch = 16, cex = .3, col = "#e6550d", xlab = "True Age",
     ylab = expression(gamma[i]), cex.main = 0.95, main = "Down-weighting concentrates at older ages"); dev.off()
# ---- Figure 4: t(nu*) standardized residual QQ plot ----
png(file.path(OUT, "74_qq_t.png"), 1400, 1000, res = 200)
sr <- r_em / sqrt(em$final_sigma2_hat)
qqplot(qt(ppoints(n), df = nu_star), sr, pch = 16, cex = .3, col = "#2c7fb8",
       xlab = sprintf("Theoretical t(%g) quantiles", nu_star), ylab = "Standardized residuals",
       main = "t(nu*) QQ Plot (heavy tails absorbed by model)"); abline(0, 1, col = "red", lwd = 2); dev.off()

saveRDS(list(em = em, mcem = mc, nu_grid = nu_grid, prof = prof, nu_star = nu_star,
             beta_ols = beta_ols, sigma2_ols = sigma2_ols, beta_opt = beta_opt,
             ll_normal = ll_normal, gamma_i = gamma_i, cmp = cmp, age = df$age),
        file.path(OUT, "treg_results.rds"))
sink(file.path(OUT, "treg_summary.txt"))
cat("UTKFace age ~ deep embeddings: Student-t regression empirical study\n")
cat(sprintf("n=%d predictors=%d nu*=%g\n", n, P, nu_star))
cat(sprintf("OLS R^2=%.3f sigma=%.3f loglik=%.1f\n", summary(ols)$r.squared, sqrt(sigma2_ols), ll_normal))
cat(sprintf("t(nu*) loglik=%.1f (improvement %.1f)\n", max(prof), max(prof) - ll_normal))
cat(sprintf("[Validation] EM vs optim max coef diff=%.5f\n", maxabs)); print(cmp, row.names = FALSE)
sink()
cat("\n[done] Output written to", OUT, "\n")
