# -*- coding: utf-8 -*-
# CIFAR-10H reaction-time two-level linear model empirical study:
# reuses the EM/MCEM implementation from Chapter 4, replacing only the data accessors.
# Level-1: logRT_ij = b0_j + b1_j*difficulty + b2_j*correct + b3_j*trial + eps,  eps~N(0,sigma2)
# Level-2: beta_j ~ N(gamma, D)   (annotator random coefficients)
suppressMessages({library(MASS); library(Matrix)})
ROOT <- dirname(dirname(this.path::this.dir()))   # derived from script location, points to code/
source(file.path(ROOT, "two-level-model/utils.R"), encoding = "UTF-8")

# ---- Override accessors with real grouped data (no large block-diagonal matrix; scalable to large J) ----
get_X_j <- function(data, j) data$X_list[[j]]
get_y_j <- function(data, j) data$y_list[[j]]
get_W_j <- function(data, j) data$W_list[[j]]

OUT <- file.path(ROOT, "empirical-2level-cifar10h/output"); dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
df <- read.csv(file.path(ROOT, "empirical-2level-cifar10h/data/cifar10h_lmm_main.csv"))
cat(sprintf("Read: rows=%d  annotators (groups)=%d\n", nrow(df), length(unique(df$group))))

# Drop annotators whose within-group design matrix is rank-deficient: if an annotator's
# accuracy is 0% or 100%, the correct column is constant, making the slope for "correct"
# unidentifiable and X_j not full rank. The GLS-gamma update in eq. (4-25b) relies on
# per-group OLS existing (i.e., X_j'X_j invertible); for rank-deficient groups ginv()
# silently returns the minimum-norm pseudoinverse, driving EM to the wrong gamma fixed point.
# We therefore drop such annotators before modelling to keep the algorithm consistent
# with the lme4 ML gold standard.
drop_rankdef <- function(df) {
  gs <- sort(unique(df$group))
  bad <- gs[vapply(gs, function(g) {
    d <- df[df$group == g, ]
    qr(cbind(1, d$difficulty_z, d$correct, d$trial_z))$rank < 4L
  }, logical(1))]
  if (length(bad)) {
    cat(sprintf("Dropping %d rank-deficient annotator(s) (within-group design matrix not full rank): %s\n", length(bad), paste(bad, collapse = ",")))
    df <- df[!df$group %in% bad, ]
  }
  df$group <- as.integer(factor(df$group))
  df
}
df <- drop_rankdef(df)
cat(sprintf("After dropping rank-deficient: rows=%d  annotators (groups)=%d\n", nrow(df), length(unique(df$group))))

build_data <- function(df) {
  J <- length(unique(df$group)); p <- 3L; q <- 0L
  X_list <- y_list <- W_list <- vector("list", J); gs <- integer(J)
  I4 <- diag(p + 1)
  for (j in 1:J) {
    d <- df[df$group == j, ]
    X_list[[j]] <- cbind(1, d$difficulty_z, d$correct, d$trial_z)
    y_list[[j]] <- d$logRT
    W_list[[j]] <- I4
    gs[j] <- nrow(d)
  }
  list(J = J, p = p, q = q, group_sizes = gs, sample_size = nrow(df),
       X_list = X_list, y_list = y_list, W_list = W_list)
}
data <- build_data(df)

# ---- Initial values: pooled OLS warm start ----
Xall <- cbind(1, df$difficulty_z, df$correct, df$trial_z)
ols  <- lm.fit(Xall, df$logRT)
gamma_0  <- matrix(ols$coefficients, ncol = 1)
sigma2_0 <- as.numeric(var(ols$residuals))
D_0      <- diag(c(0.10, 0.03, 0.03, 0.01))
cat("Initial gamma_0 =", round(as.numeric(gamma_0), 4), " sigma2_0 =", round(sigma2_0, 4), "\n")

t_em <- system.time(
  em <- run_single_em_or_mcem(data, gamma_0, D_0, sigma2_0,
                              is_mcem = FALSE, max_iter = 300, tolerance = 1e-6)
)["elapsed"]
cat(sprintf("[EM] elapsed %.1fs, iterations %d\n", t_em, em$final_iter_count))

# ---- MCEM with different Monte Carlo sample sizes M (seed fixed before each call for reproducibility) ----
mcem <- list()
for (M in c(20, 50, 200)) {
  set.seed(20250529L)
  tt <- system.time(
    mcem[[as.character(M)]] <- run_single_em_or_mcem(
      data, gamma_0, D_0, sigma2_0, is_mcem = TRUE, mc_times = M,
      max_iter = 120, tolerance = 1e-5)
  )["elapsed"]
  cat(sprintf("[MCEM M=%d] elapsed %.1fs, iterations %d\n", M, tt,
              mcem[[as.character(M)]]$final_iter_count))
}

# ---- lme4 gold standard (ML, same criterion as marginal MLE) ----
gold <- tryCatch({
  suppressMessages(library(lme4))
  m <- lmer(logRT ~ difficulty_z + correct + trial_z +
              (1 + difficulty_z + correct + trial_z | group),
            data = df, REML = FALSE,
            control = lmerControl(optimizer = "bobyqa",
                                  optCtrl = list(maxfun = 3e5)))
  list(gamma = as.numeric(fixef(m)),
       D = matrix(VarCorr(m)$group, 4, 4),
       sigma2 = sigma(m)^2,
       loglik = as.numeric(logLik(m)),
       singular = isSingular(m))
}, error = function(e) { cat("lme4 failed:", conditionMessage(e), "\n"); NULL })

# ---- Summary comparison table ----
labels <- c("Intercept", "Difficulty", "Correct", "Trial")
tab <- data.frame(
  Parameter = labels,
  EM   = round(as.numeric(em$final_gamma_hat), 4),
  `MCEM_M20`  = round(as.numeric(mcem[["20"]]$final_gamma_hat), 4),
  `MCEM_M50`  = round(as.numeric(mcem[["50"]]$final_gamma_hat), 4),
  `MCEM_M200` = round(as.numeric(mcem[["200"]]$final_gamma_hat), 4),
  check.names = FALSE
)
if (!is.null(gold)) tab$lme4 <- round(gold$gamma, 4)
cat("\n===== Fixed effects gamma comparison =====\n"); print(tab, row.names = FALSE)

D_em <- em$final_D_hat
icc_intercept <- D_em[1, 1] / (D_em[1, 1] + em$final_sigma2_hat)
cat(sprintf("\nEM: sigma2 = %.4f\n", em$final_sigma2_hat))
cat("EM: random-effect standard deviations (sqrt(diag(D))) =", round(sqrt(diag(D_em)), 4), "\n")
cat(sprintf("EM: intercept ICC = %.4f  (proportion of baseline logRT variance attributable to between-annotator differences)\n", icc_intercept))
cat(sprintf("EM: final marginal log-likelihood = %.2f\n", em$loglik))
if (!is.null(gold)) {
  cat(sprintf("lme4: sigma2 = %.4f, logLik = %.2f, singular=%s\n",
              gold$sigma2, gold$loglik, gold$singular))
  cat("lme4: random-effect standard deviations =", round(sqrt(diag(gold$D)), 4), "\n")
  gap <- abs(em$loglik - gold$loglik)
  cat(sprintf("[Validation] EM vs lme4 marginal log-likelihood gap = %.4f  ->  %s\n",
              gap, ifelse(gap < 0.1, "PASS (<0.1, digit-level agreement)", "CHECK (>0.1)")))
}

# ---- Figure 1: EM marginal log-likelihood monotone increase (verify monotonicity theorem on real data) ----
ll <- sapply(em$history, function(h) h$loglik)
png(file.path(OUT, "61_loglik_monotone.png"), width = 1500, height = 1000, res = 200)
plot(0:(length(ll) - 1), ll, type = "o", pch = 16, col = "#2c7fb8",
     xlab = "Iteration k", ylab = "Marginal log-likelihood ln p(y|gamma,D,sigma2)",
     main = "CIFAR-10H two-level model: EM marginal log-likelihood monotone increase")
grid(); dev.off()

# ---- Figure 2: Annotator random intercept caterpillar plot (BLUP) ----
J <- data$J
blup <- numeric(J); se <- numeric(J)
for (j in 1:J) {
  mu <- get_mu_jk_star(data, j, em$final_gamma_hat, D_em, em$final_sigma2_hat)
  V  <- get_V_jk_star(data, j, D_em, em$final_sigma2_hat)
  blup[j] <- mu[1]; se[j] <- sqrt(max(V[1, 1], 0))
}
o <- order(blup)
png(file.path(OUT, "62_caterpillar_intercept.png"), width = 1700, height = 1000, res = 200)
plot(seq_len(J), blup[o], pch = 16, cex = .4, col = "#225ea8",
     ylim = range(blup[o] - 1.96 * se[o], blup[o] + 1.96 * se[o]),
     xlab = "Annotator (sorted by random intercept)", ylab = "Random intercept mu_j[1] (log-RT deviation from population mean)",
     main = sprintf("Random effects for annotator response speed (J=%d)", J))
arrows(seq_len(J), blup[o] - 1.96 * se[o], seq_len(J), blup[o] + 1.96 * se[o],
       length = 0, col = "#a6bddb")
points(seq_len(J), blup[o], pch = 16, cex = .4, col = "#225ea8")
abline(h = 0, lty = 2, col = "red"); dev.off()

# ---- Figure 3: gamma convergence trajectory across EM iterations ----
gh <- sapply(em$history, function(h) h$gamma)   # (p+1) x (iter+1)
png(file.path(OUT, "63_gamma_convergence.png"), width = 1600, height = 1000, res = 200)
matplot(0:(ncol(gh) - 1), t(gh), type = "l", lwd = 2, lty = 1,
        col = c("#000000", "#e41a1c", "#377eb8", "#4daf4a"),
        xlab = "Iteration k", ylab = "gamma estimate",
        main = "EM convergence trajectory for fixed effects gamma")
legend("topright", legend = labels, col = c("#000000", "#e41a1c", "#377eb8", "#4daf4a"),
       lwd = 2, bty = "n"); grid(); dev.off()

# ---- Save results ----
saveRDS(list(em = em, mcem = mcem, gold = gold, tab = tab,
             icc = icc_intercept, D = D_em), file.path(OUT, "lmm_results.rds"))
sink(file.path(OUT, "lmm_summary.txt"))
cat("CIFAR-10H two-level linear model empirical results\n")
cat(sprintf("J = %d  N = %d\n\n", data$J, data$sample_size))
print(tab, row.names = FALSE)
cat(sprintf("\nEM sigma2=%.4f  intercept ICC=%.4f  logLik=%.2f\n", em$final_sigma2_hat, icc_intercept, em$loglik))
cat(sprintf("EM random-effect SD = %s\n", paste(round(sqrt(diag(D_em)), 4), collapse = " ")))
if (!is.null(gold)) cat(sprintf("lme4 sigma2=%.4f logLik=%.2f singular=%s\n", gold$sigma2, gold$loglik, gold$singular))
sink()
cat("\n[done] output written to", OUT, "\n")
