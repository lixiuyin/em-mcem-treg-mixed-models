library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")
options(hlm_reml = TRUE)   # Use REML to estimate variance component D in simulations, correcting the inherent downward bias at small numbers of groups

set.seed(2025)

J <- 20; p <- 2; q <- 3   # J=20; D estimated with REML to correct downward bias of variance component at small numbers of groups
true_params <- generate_true_2level_params(p, q)  # D = Q diag(5,6,7) Q^T (random orthogonal Q); gamma, sigma^2 ~ Uniform
gamma_real <- true_params$gamma_real
D_real <- true_params$D_real
sigma2_real <- true_params$sigma2_real

gamma_0 <- matrix(rep(0.1, (p + 1)*(q + 1)), ncol = 1)
D_0 <- matrix(c(4, 2, 1,
                2, 5, 3,
                1, 3, 6), ncol = 3, byrow = TRUE)
sigma2_0 <- 0.1

out_dir <- file.path(this.dir(), "figures")

save_plot <- function(name, expr, w = 1400, h = 1000) {
  png(file.path(out_dir, paste0(name, ".png")), width = w, height = h, res = 150)
  force(expr)
  dev.off()
  cat("Saved:", name, ".png\n")
}

# Shared dataset
data <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)

# ------- Group 1: single MCEM run (M=10 and M=100) -------
cat("\n>>> Single MCEM run (M=10)\n")
mcem_10 <- run_single_em_or_mcem(data = data,
                                  gamma_0 = gamma_0, D_0 = D_0, sigma2_0 = sigma2_0,
                                  max_iter = 50, tolerance = 1e-6,
                                  is_mcem = TRUE, mc_times = 10)
save_plot("11_mcem_M10_single_iter",
          plot_single_iteration(mcem_10, title = "MCEM (M=10) single-run iteration path"))
save_plot("12_mcem_M10_single_mse",
          plot_single_mse(mcem_10, title = "MCEM (M=10) single-run MSE evolution"))

cat("\n>>> Single MCEM run (M=100)\n")
mcem_100 <- run_single_em_or_mcem(data = data,
                                   gamma_0 = gamma_0, D_0 = D_0, sigma2_0 = sigma2_0,
                                   max_iter = 50, tolerance = 1e-6,
                                   is_mcem = TRUE, mc_times = 100)
save_plot("13_mcem_M100_single_iter",
          plot_single_iteration(mcem_100, title = "MCEM (M=100) single-run iteration path"))
save_plot("14_mcem_M100_single_mse",
          plot_single_mse(mcem_100, title = "MCEM (M=100) single-run MSE evolution"))

# ------- Group 2: multiple MCEM runs (averaged) -------
cat("\n>>> 500 simulations averaged MCEM (M=10)\n")
mcem_multi_10 <- run_multiple_em_or_mcem(data = data,
                                         n_sim = 500,
                                         max_iter = 50,
                                         tolerance = 1e-6,
                                         is_mcem = TRUE,
                                         mc_times = 10,
                                         reset_dataset = TRUE,
                                         gamma_0 = gamma_0, D_0 = D_0, sigma2_0 = sigma2_0)
save_plot("15_mcem_M10_500sim_iter",
          plot_multiple_simulation_iterations(mcem_multi_10, title = "MCEM (M=10): average iteration path over 500 simulations"))
save_plot("16_mcem_M10_500sim_mse",
          plot_multiple_simulation_mse(mcem_multi_10, title = "MCEM (M=10): MSE evolution averaged over 500 simulations"))

cat("\n>>> 500 simulations averaged MCEM (M=100)\n")
mcem_multi_100 <- run_multiple_em_or_mcem(data = data,
                                          n_sim = 500,
                                          max_iter = 50,
                                          tolerance = 1e-6,
                                          is_mcem = TRUE,
                                          mc_times = 100,
                                          reset_dataset = TRUE,
                                          gamma_0 = gamma_0, D_0 = D_0, sigma2_0 = sigma2_0)
save_plot("17_mcem_M100_500sim_iter",
          plot_multiple_simulation_iterations(mcem_multi_100, title = "MCEM (M=100): average iteration path over 500 simulations"))
save_plot("18_mcem_M100_500sim_mse",
          plot_multiple_simulation_mse(mcem_multi_100, title = "MCEM (M=100): MSE evolution averaged over 500 simulations"))

cat("\n=========== Numerical summary ===========\n")
cat("MCEM(M=10)  single run gamma:", round(as.vector(mcem_10$final_gamma_hat), 2), "\n")
cat("MCEM(M=100) single run gamma:", round(as.vector(mcem_100$final_gamma_hat), 2), "\n")
cat("MCEM(M=10)  mean gamma:", round(as.vector(mcem_multi_10$final_gamma_hat), 2), "\n")
cat("MCEM(M=100) mean gamma:", round(as.vector(mcem_multi_100$final_gamma_hat), 2), "\n")
cat("gamma true values:  ", round(as.vector(gamma_real), 2), "\n")
cat("sigma^2 true=", round(sigma2_real, 3), ", estimated:",
    round(mcem_10$final_sigma2_hat, 3),
    round(mcem_100$final_sigma2_hat, 3),
    round(mcem_multi_10$final_sigma2_hat, 3),
    round(mcem_multi_100$final_sigma2_hat, 3), "\n")
