# -*- coding: utf-8 -*-
# Redraw all figures from lmm_results.rds using macOS quartz device with English labels.
suppressMessages({library(MASS); library(Matrix)})
CJK <- "sans"
ROOT <- dirname(dirname(this.path::this.dir()))   # derived from script location, points to code/
source(file.path(ROOT, "two-level-model/utils.R"), encoding = "UTF-8")
get_X_j <- function(data, j) data$X_list[[j]]
get_y_j <- function(data, j) data$y_list[[j]]
get_W_j <- function(data, j) data$W_list[[j]]
OUT <- file.path(ROOT, "empirical-2level-cifar10h/output")
result_file <- file.path(OUT, "lmm_results.rds")
if (!file.exists(result_file)) {
  stop("Missing ", result_file, ". Run code/02_lmm_real.R first to regenerate saved results.")
}
r <- readRDS(result_file); em <- r$em
df <- read.csv(file.path(ROOT, "empirical-2level-cifar10h/data/cifar10h_lmm_main.csv"))
# Consistent with 02_lmm_real.R: drop rank-deficient annotators and re-index (J=299),
# so that data matches the stored em object
gs <- sort(unique(df$group))
bad <- gs[vapply(gs, function(g){d<-df[df$group==g,]; qr(cbind(1,d$difficulty_z,d$correct,d$trial_z))$rank<4L}, logical(1))]
if (length(bad)) df <- df[!df$group %in% bad, ]
df$group <- as.integer(factor(df$group))
J <- length(unique(df$group)); p <- 3L; I4 <- diag(p + 1)
Xl <- yl <- Wl <- vector("list", J)
for (j in 1:J) { d <- df[df$group == j, ]
  Xl[[j]] <- cbind(1, d$difficulty_z, d$correct, d$trial_z); yl[[j]] <- d$logRT; Wl[[j]] <- I4 }
data <- list(J = J, p = p, q = 0L, sample_size = nrow(df), X_list = Xl, y_list = yl, W_list = Wl)
labels <- c("Intercept", "Difficulty", "Correct", "Trial")
op <- function(fn, w, h) png(file.path(OUT, fn), w, h, res = 200, type = "quartz")

ll <- sapply(em$history, function(h) h$loglik)
op("61_loglik_monotone.png", 1500, 1000); par(family = CJK)
plot(0:(length(ll) - 1), ll, type = "o", pch = 16, col = "#2c7fb8",
     xlab = "Iteration k", ylab = "Marginal log-likelihood ln p(y|gamma,D,sigma^2)",
     main = "CIFAR-10H two-level model: EM marginal log-likelihood monotone increase"); grid(); dev.off()

blup <- se <- numeric(J)
for (j in 1:J) { mu <- get_mu_jk_star(data, j, em$final_gamma_hat, r$D, em$final_sigma2_hat)
  V <- get_V_jk_star(data, j, r$D, em$final_sigma2_hat); blup[j] <- mu[1]; se[j] <- sqrt(max(V[1, 1], 0)) }
o <- order(blup)
op("62_caterpillar_intercept.png", 1700, 1000); par(family = CJK)
plot(seq_len(J), blup[o], pch = 16, cex = .4, col = "#225ea8",
     ylim = range(blup[o] - 1.96 * se[o], blup[o] + 1.96 * se[o]),
     xlab = "Annotator (sorted by random intercept)", ylab = "Random intercept mu_j[1] (log-RT deviation from population mean)",
     main = sprintf("Random effects for annotator response speed (J=%d)", J))
arrows(seq_len(J), blup[o] - 1.96 * se[o], seq_len(J), blup[o] + 1.96 * se[o], length = 0, col = "#a6bddb")
points(seq_len(J), blup[o], pch = 16, cex = .4, col = "#225ea8"); abline(h = 0, lty = 2, col = "red"); dev.off()

gh <- sapply(em$history, function(h) h$gamma)
op("63_gamma_convergence.png", 1600, 1000); par(family = CJK)
matplot(0:(ncol(gh) - 1), t(gh), type = "l", lwd = 2, lty = 1,
        col = c("#000000", "#e41a1c", "#377eb8", "#4daf4a"),
        xlab = "Iteration k", ylab = "gamma estimate", main = "EM convergence trajectory for fixed effects gamma")
legend("right", legend = labels, col = c("#000000", "#e41a1c", "#377eb8", "#4daf4a"),
       lwd = 2, bty = "n"); grid(); dev.off()
cat("[done] LMM figures redrawn\n")
