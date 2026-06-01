# t-regression: single-run, initialization-strategy, and degrees-of-freedom EM figures.
# Mirrors two-level-model/driver_em_plots.R; regenerates the thesis Chapter 3 EM
# figures that driver_compare500.R (500-sim panels 21-24) does not produce.
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

# ------- Group 1: single EM run (图 3.1 left, 图 3.2 left) -------
cat("\n>>> Single EM run\n")
data <- generate_t_simulated_data(n, p, t_df, t_mu, t_sigma2_real, beta_real)
em_single <- run_single_em_or_mcem(X = data$X, y = data$y_real, t_df = t_df,
                                   beta_0 = fixed_init_beta, sigma2_0 = fixed_init_sigma2,
                                   max_iter = 100, tolerance = 1e-7, is_mcem = FALSE)
save_plot("25_t_em_single_iter",
          plot_single_iteration(em_single, beta_real, t_sigma2_real,
                                title = "t-regression EM: single-run iteration path"))
save_plot("26_t_em_single_mse",
          plot_single_mse(em_single, beta_real, t_sigma2_real,
                          title = "t-regression EM: single-run MSE evolution"))

# ------- Group 2: initialisation strategies, 500 simulations (图 3.3) -------
cat("\n>>> Conservative random initialisation: 500 simulations\n")
em_normal_init <- run_multiple_em_or_mcem(n_sim = 500, p = p,
                                          max_iter = 100, tolerance = 1e-7,
                                          is_mcem = FALSE, reset_dataset = TRUE,
                                          bold_random = FALSE,
                                          n = n, t_df = t_df, t_mu = t_mu,
                                          t_sigma2_real = t_sigma2_real, beta_real = beta_real)
save_plot("27_t_em_normal_init_iter",
          plot_multiple_simulation_iterations(em_normal_init, beta_real, t_sigma2_real,
                                              title = "t-regression EM: conservative-init iteration path (500 simulations)"))

cat("\n>>> Wide-range random initialisation: 500 simulations\n")
em_bold_init <- run_multiple_em_or_mcem(n_sim = 500, p = p,
                                        max_iter = 100, tolerance = 1e-7,
                                        is_mcem = FALSE, reset_dataset = TRUE,
                                        bold_random = TRUE,
                                        n = n, t_df = t_df, t_mu = t_mu,
                                        t_sigma2_real = t_sigma2_real, beta_real = beta_real)
save_plot("28_t_em_bold_init_iter",
          plot_multiple_simulation_iterations(em_bold_init, beta_real, t_sigma2_real,
                                              title = "t-regression EM: wide-range-init iteration path (500 simulations)"))

# ------- Group 3: degrees-of-freedom sweep (图 3.4) -------
# Coarse df grid (vs the thesis 5:105 full grid) to keep runtime feasible; each df
# point still runs 500 EM simulations, so the trend matches the thesis.
cat("\n>>> Degrees-of-freedom sweep\n")
df_grid <- c(5, 9, 10, 13, 18, 25, 40, 60, 85, 105)
df_study <- study_df_effect(n = n, p = p, t_mu = t_mu, t_sigma2_real = t_sigma2_real,
                            beta_real = beta_real, df_range = df_grid)
save_plot("29_t_df_error", print(df_study$plots$error_plot))
save_plot("30_t_df_iter", print(df_study$plots$iter_plot))

cat("\n=========== Done: figures 25-30 written to", out_dir, "===========\n")
