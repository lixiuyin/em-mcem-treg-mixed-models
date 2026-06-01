library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")
options(hlm_reml = TRUE)   # Use REML to estimate variance component D in simulations, correcting the inherent downward bias at small numbers of groups

set.seed(2025)

# Consistent with paper section 4.1.2 (J=20); D estimated with REML to correct downward bias of variance component at small numbers of groups
J <- 20; p <- 2; q <- 3
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

# ------- Group 1: single EM run -------
data <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)
cat("\n>>> Single EM run\n")
em_result <- run_single_em_or_mcem(data = data,
                                   gamma_0 = gamma_0, D_0 = D_0, sigma2_0 = sigma2_0,
                                   max_iter = 50, tolerance = 1e-6, is_mcem = FALSE)
save_plot("01_em_single_iter",
          plot_single_iteration(em_result, title = "EM algorithm single-run iteration path"))
save_plot("02_em_single_mse",
          plot_single_mse(em_result, title = "EM algorithm single-run MSE evolution"))

# ------- Group 2: 500 simulations averaged -------
cat("\n>>> 500 EM simulations (data regenerated each time)\n")
multiple_em_result <- run_multiple_em_or_mcem(data = data,
                                              n_sim = 500,
                                              max_iter = 50,
                                              tolerance = 1e-6,
                                              is_mcem = FALSE,
                                              reset_dataset = TRUE,
                                              gamma_0 = gamma_0, D_0 = D_0, sigma2_0 = sigma2_0)
save_plot("03_em_500sim_iter",
          plot_multiple_simulation_iterations(multiple_em_result, title = "EM algorithm: average iteration path over 500 simulations"))
save_plot("04_em_500sim_mse",
          plot_multiple_simulation_mse(multiple_em_result, title = "EM algorithm: MSE evolution averaged over 500 simulations"))

# ------- Group 3: different initialisation strategies -------
cat("\n>>> Standard-distribution initialisation: 500 simulations\n")
multiple_em_result_normal_init <- run_multiple_em_or_mcem(data = data,
                                                          n_sim = 500,
                                                          max_iter = 50,
                                                          tolerance = 1e-6,
                                                          is_mcem = FALSE,
                                                          reset_dataset = TRUE,
                                                          bold_random = FALSE)
save_plot("05_em_normal_init_iter",
          plot_multiple_simulation_iterations(multiple_em_result_normal_init, title = "Standard-distribution initialisation: iteration path"))
save_plot("06_em_normal_init_mse",
          plot_multiple_simulation_mse(multiple_em_result_normal_init, title = "Standard-distribution initialisation: MSE evolution"))

cat("\n>>> Wide-range initialisation: 500 simulations\n")
multiple_em_result_bold_init <- run_multiple_em_or_mcem(data = data,
                                                        n_sim = 500,
                                                        max_iter = 50,
                                                        tolerance = 1e-6,
                                                        is_mcem = FALSE,
                                                        reset_dataset = TRUE,
                                                        bold_random = TRUE)
save_plot("07_em_bold_init_iter",
          plot_multiple_simulation_iterations(multiple_em_result_bold_init, title = "Wide-range initialisation: iteration path"))
save_plot("08_em_bold_init_mse",
          plot_multiple_simulation_mse(multiple_em_result_bold_init, title = "Wide-range initialisation: MSE evolution"))

# ------- Numerical summary -------
cat("\n=========== Numerical summary ===========\n")
cat("EM single run gamma:", round(as.vector(em_result$final_gamma_hat), 2), "\n")
cat("EM 500-sim mean gamma:", round(as.vector(multiple_em_result$final_gamma_hat), 2), "\n")
cat("Standard-dist init gamma:", round(as.vector(multiple_em_result_normal_init$final_gamma_hat), 2), "\n")
cat("Wide-range init gamma:", round(as.vector(multiple_em_result_bold_init$final_gamma_hat), 2), "\n")
cat("gamma true values:  ", round(as.vector(gamma_real), 2), "\n\n")
cat("sigma^2 true=", round(sigma2_real, 3), ", estimated:",
    round(em_result$final_sigma2_hat, 3),
    round(multiple_em_result$final_sigma2_hat, 3),
    round(multiple_em_result_normal_init$final_sigma2_hat, 3),
    round(multiple_em_result_bold_init$final_sigma2_hat, 3), "\n")
