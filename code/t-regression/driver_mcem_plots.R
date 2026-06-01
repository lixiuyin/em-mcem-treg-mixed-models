# t-regression: single-run MCEM figures for M=1 and M=100 Monte Carlo samples.
# Mirrors two-level-model/driver_mcem_plots.R; regenerates thesis figure 图 3.5.
# The 500-simulation MCEM panels (23/24) are produced by driver_compare500.R.
library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

set.seed(2025)

# Settings from paper Section 3
n <- 500
p <- 2
t_df <- 10
t_mu <- 0
t_sigma2_real <- 0.4
beta_real <- c(2, 3, 5)
fixed_init_beta <- c(0, 0, 0)
fixed_init_sigma2 <- 0.1

out_dir <- file.path(this.dir(), "figures")

save_plot <- function(name, expr, w = 1400, h = 1000) {
  png(file.path(out_dir, paste0(name, ".png")), width = w, height = h, res = 150)
  force(expr)
  dev.off()
  cat("Saved:", name, ".png\n")
}

data <- generate_t_simulated_data(n, p, t_df, t_mu, t_sigma2_real, beta_real)

# ------- MCEM single run, M = 1 sample (图 3.5 left) -------
cat("\n>>> Single MCEM run (M=1)\n")
mcem_m1 <- run_single_em_or_mcem(X = data$X, y = data$y_real, t_df = t_df,
                                 beta_0 = fixed_init_beta, sigma2_0 = fixed_init_sigma2,
                                 max_iter = 40, tolerance = 1e-7,
                                 is_mcem = TRUE, mc_times = 1)
save_plot("31_t_mcem_M1_single_iter",
          plot_single_iteration(mcem_m1, beta_real, t_sigma2_real,
                                title = "t-regression MCEM (M=1): single-run iteration path"))

# ------- MCEM single run, M = 100 samples (图 3.5 right) -------
cat("\n>>> Single MCEM run (M=100)\n")
mcem_m100 <- run_single_em_or_mcem(X = data$X, y = data$y_real, t_df = t_df,
                                   beta_0 = fixed_init_beta, sigma2_0 = fixed_init_sigma2,
                                   max_iter = 40, tolerance = 1e-7,
                                   is_mcem = TRUE, mc_times = 100)
save_plot("32_t_mcem_M100_single_iter",
          plot_single_iteration(mcem_m100, beta_real, t_sigma2_real,
                                title = "t-regression MCEM (M=100): single-run iteration path"))

cat("\n=========== Done: figures 31-32 written to", out_dir, "===========\n")
