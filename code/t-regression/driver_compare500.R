# t-regression: comparability test for EM vs MCEM over 500 simulations
# Same data, same fixed initial values - EM and MCEM differ only in the E-step implementation
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

# ===== 500 simulations, regenerating data each time =====

cat("\n>>>>> EM: 500 simulations <<<<<\n")
t_em_start <- Sys.time()
em_500 <- run_multiple_em_or_mcem(n_sim = 500,
                                  max_iter = 100,
                                  tolerance = 1e-7,
                                  is_mcem = FALSE,
                                  reset_dataset = TRUE,
                                  beta_0 = fixed_init_beta,
                                  sigma2_0 = fixed_init_sigma2)
t_em_elapsed <- as.numeric(Sys.time() - t_em_start, units = "secs")

save_plot("21_t_em_500sim_iter",
          plot_multiple_simulation_iterations(em_500, beta_real, t_sigma2_real,
                                              title = "t-regression EM: mean iteration path over 500 simulations"))
save_plot("22_t_em_500sim_mse",
          plot_multiple_simulation_mse(em_500, beta_real, t_sigma2_real,
                                       title = "t-regression EM: MSE over 500 simulations"))

cat("\n>>>>> MCEM(M=100): 500 simulations <<<<<\n")
t_mc_start <- Sys.time()
mcem_500 <- run_multiple_em_or_mcem(n_sim = 500,
                                    max_iter = 100,
                                    tolerance = 1e-7,
                                    is_mcem = TRUE,
                                    mc_times = 100,
                                    reset_dataset = TRUE,
                                    beta_0 = fixed_init_beta,
                                    sigma2_0 = fixed_init_sigma2)
t_mc_elapsed <- as.numeric(Sys.time() - t_mc_start, units = "secs")

save_plot("23_t_mcem_500sim_iter",
          plot_multiple_simulation_iterations(mcem_500, beta_real, t_sigma2_real,
                                              title = "t-regression MCEM(M=100): mean iteration path over 500 simulations"))
save_plot("24_t_mcem_500sim_mse",
          plot_multiple_simulation_mse(mcem_500, beta_real, t_sigma2_real,
                                       title = "t-regression MCEM(M=100): MSE over 500 simulations"))

# ===== Numerical summary (analogous to Table 3-3 comparison in the paper) =====
mse_em <- get_mse(em_500$final_beta_hat, em_500$final_sigma2_hat, beta_real, t_sigma2_real)
mse_mc <- get_mse(mcem_500$final_beta_hat, mcem_500$final_sigma2_hat, beta_real, t_sigma2_real)

cat("\n========== Comparison of 500-simulation means ==========\n")
cat(sprintf("Parameter    |  EM mean (500)   |  MCEM(M=100) mean  | True value\n"))
cat(sprintf("beta0        |   %8.4f      |   %8.4f         | %.2f\n",
            em_500$final_beta_hat[1], mcem_500$final_beta_hat[1], beta_real[1]))
cat(sprintf("beta1        |   %8.4f      |   %8.4f         | %.2f\n",
            em_500$final_beta_hat[2], mcem_500$final_beta_hat[2], beta_real[2]))
cat(sprintf("beta2        |   %8.4f      |   %8.4f         | %.2f\n",
            em_500$final_beta_hat[3], mcem_500$final_beta_hat[3], beta_real[3]))
cat(sprintf("sigma^2      |   %8.4f      |   %8.4f         | %.2f\n",
            em_500$final_sigma2_hat, mcem_500$final_sigma2_hat, t_sigma2_real))
cat(sprintf("MSE_beta     |   %.3e     |   %.3e        |\n",
            mse_em$beta_mse, mse_mc$beta_mse))
cat(sprintf("MSE_sigma^2  |   %.3e     |   %.3e        |\n",
            mse_em$sigma2_mse, mse_mc$sigma2_mse))
cat(sprintf("MSE_total    |   %.3e     |   %.3e        |\n",
            mse_em$total_mse, mse_mc$total_mse))
cat(sprintf("Mean iters   |   %.2f          |   %.2f             |\n",
            em_500$final_iter_count, mcem_500$final_iter_count))
cat(sprintf("Total time   |   %.1f sec      |   %.1f sec         |\n",
            t_em_elapsed, t_mc_elapsed))
