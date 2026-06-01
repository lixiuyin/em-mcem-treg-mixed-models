# -*- coding: utf-8 -*-
# Redraw all figures from treg_results.rds using macOS quartz device (English labels).
suppressMessages({library(MASS)})
CJK <- "sans"   # labels are now English; no CJK font needed
ROOT <- dirname(dirname(this.path::this.dir()))   # derived from script location, points to code/
OUT  <- file.path(ROOT, "empirical-t-utkface/output")
result_file <- file.path(OUT, "treg_results.rds")
if (!file.exists(result_file)) {
  stop("Missing ", result_file, ". Run code/02_tregression_real.R first to regenerate saved results.")
}
r <- readRDS(result_file)
df <- read.csv(file.path(ROOT, "empirical-t-utkface/data/utkface_t_regression.csv"))
pc <- grep("^PC", names(df), value = TRUE)
X <- as.matrix(cbind(1, df[, pc])); y <- df$age; n <- nrow(X)
em <- r$em; nu <- r$nu_star
ols_res <- y - as.numeric(X %*% c(r$beta_ols))
r_em <- as.numeric(y - X %*% em$final_beta_hat)
gi <- r$gamma_i
pf <- function(f) file.path(OUT, f)
op <- function(fn, w, h) png(fn, w, h, res = 200, type = "quartz")

op(pf("71_qq_normal.png"), 1400, 1000); par(family = CJK)
qqnorm(ols_res, pch = 16, cex = .3, col = "#3182bd", main = "OLS Residuals: Normal QQ Plot (tail deviation indicates heavy tails)")
qqline(ols_res, col = "red", lwd = 2); dev.off()

op(pf("72_profile_nu.png"), 1400, 1000); par(family = CJK)
plot(r$nu_grid, r$prof, type = "o", pch = 16, col = "#756bb1", log = "x",
     xlab = "Degrees of freedom nu (log scale)", ylab = "Marginal log-likelihood",
     main = "Student-t Regression: Profile Likelihood for Degrees of Freedom")
abline(v = nu, lty = 2, col = "red"); abline(h = r$ll_normal, lty = 3, col = "gray40")
legend("bottomright", c(sprintf("nu* = %g", nu), "Normal baseline"), col = c("red", "gray40"),
       lty = c(2, 3), bty = "n"); dev.off()

op(pf("73_weights.png"), 2200, 950); par(mfrow = c(1, 2), family = CJK)
plot(abs(r_em), gi, pch = 16, cex = .3, col = "#31a354", xlab = "|Residual|",
     ylab = expression(gamma[i]), cex.main = 0.95, main = "E-step weights vs |residual|")
plot(df$age, gi, pch = 16, cex = .3, col = "#e6550d", xlab = "True Age",
     ylab = expression(gamma[i]), cex.main = 0.95, main = "Down-weighting concentrates at older ages"); dev.off()

op(pf("74_qq_t.png"), 1400, 1000); par(family = CJK)
sr <- r_em / sqrt(em$final_sigma2_hat)
qqplot(qt(ppoints(n), df = nu), sr, pch = 16, cex = .3, col = "#2c7fb8",
       xlab = sprintf("Theoretical t(%g) quantiles", nu), ylab = "Standardized residuals",
       main = "t(nu*) QQ Plot (heavy tails absorbed by model)"); abline(0, 1, col = "red", lwd = 2); dev.off()
cat("[done] t-regression figures redrawn\n")
