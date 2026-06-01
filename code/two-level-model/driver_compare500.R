# Two-level linear model: comparability test of EM vs MCEM over 500 simulations
# Same data scale, same fixed initial values - ensures EM and MCEM differ only in the E-step implementation
library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")
options(hlm_reml = TRUE)   # Use REML to estimate variance component D in simulations, correcting the inherent downward bias at small numbers of groups

set.seed(2025)

# Paper section 4 setup (J=20). Variance component D is estimated with REML to correct the
# degrees-of-freedom loss from estimating gamma, making D approximately unbiased even at small
# numbers of groups (see README sections 4.5.2 and 6.2).
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

# Use one baseline dataset as the starting point (subsequent runs redraw with reset_dataset=TRUE)
data <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)

out_dir <- file.path(this.dir(), "figures")

save_plot <- function(name, expr, w = 1400, h = 1000) {
  png(file.path(out_dir, paste0(name, ".png")), width = w, height = h, res = 150)
  force(expr)
  dev.off()
  cat("Saved:", name, ".png\n")
}

# ===== 500 simulations EM =====
cat("\n>>>>> EM 500 simulations <<<<<\n")
t_em_start <- Sys.time()
em_500 <- run_multiple_em_or_mcem(data = data,
                                  n_sim = 500,
                                  max_iter = 30,
                                  tolerance = 1e-6,
                                  is_mcem = FALSE,
                                  reset_dataset = TRUE,
                                  gamma_0 = gamma_0,
                                  D_0 = D_0,
                                  sigma2_0 = sigma2_0)
t_em_elapsed <- as.numeric(Sys.time() - t_em_start, units = "secs")

save_plot("51_hlm_em_500sim_iter",
          plot_multiple_simulation_iterations(em_500, title = "Two-level linear model EM: average iteration path over 500 simulations"))
save_plot("52_hlm_em_500sim_mse",
          plot_multiple_simulation_mse(em_500, title = "Two-level linear model EM: MSE evolution over 500 simulations"))

# ===== 500 simulations MCEM(M=50) =====
cat("\n>>>>> MCEM(M=50) 500 simulations <<<<<\n")
t_mc_start <- Sys.time()
mcem_500 <- run_multiple_em_or_mcem(data = data,
                                    n_sim = 500,
                                    max_iter = 30,
                                    tolerance = 1e-6,
                                    is_mcem = TRUE,
                                    mc_times = 50,
                                    reset_dataset = TRUE,
                                    gamma_0 = gamma_0,
                                    D_0 = D_0,
                                    sigma2_0 = sigma2_0)
t_mc_elapsed <- as.numeric(Sys.time() - t_mc_start, units = "secs")

save_plot("53_hlm_mcem_500sim_iter",
          plot_multiple_simulation_iterations(mcem_500, title = "Two-level linear model MCEM(M=50): average iteration path over 500 simulations"))
save_plot("54_hlm_mcem_500sim_mse",
          plot_multiple_simulation_mse(mcem_500, title = "Two-level linear model MCEM(M=50): MSE evolution over 500 simulations"))

# ===== Numerical summary =====
mse_em <- get_mse(em_500$final_gamma_hat, em_500$final_D_hat, em_500$final_sigma2_hat,
                  gamma_real, D_real, sigma2_real)
mse_mc <- get_mse(mcem_500$final_gamma_hat, mcem_500$final_D_hat, mcem_500$final_sigma2_hat,
                  gamma_real, D_real, sigma2_real)

cat("\n========== Two-level model: comparison of 500-simulation averages ==========\n")
cat("gamma estimate (EM):    ", round(as.vector(em_500$final_gamma_hat), 3), "\n")
cat("gamma estimate (MCEM):  ", round(as.vector(mcem_500$final_gamma_hat), 3), "\n")
cat("gamma true values:      ", round(as.vector(gamma_real), 2), "\n\n")

cat("D diagonal (EM):    ", round(diag(em_500$final_D_hat), 3), "\n")
cat("D diagonal (MCEM):  ", round(diag(mcem_500$final_D_hat), 3), "\n")
cat("D diagonal true:    ", round(diag(D_real), 2), "\n\n")

cat(sprintf("sigma^2:  EM=%.4f  MCEM=%.4f  true=%.4f\n",
            em_500$final_sigma2_hat, mcem_500$final_sigma2_hat, sigma2_real))

cat(sprintf("\nMSE_gamma:  EM=%.5f   MCEM=%.5f\n", mse_em$mse_gamma, mse_mc$mse_gamma))
cat(sprintf("MSE_D:      EM=%.5f   MCEM=%.5f\n", mse_em$mse_D,     mse_mc$mse_D))
cat(sprintf("MSE_sigma2: EM=%.5f   MCEM=%.5f\n", mse_em$mse_sigma2, mse_mc$mse_sigma2))
cat(sprintf("MSE_total: EM=%.5f   MCEM=%.5f\n", mse_em$mse,        mse_mc$mse))
cat(sprintf("Avg iterations:  EM=%.2f   MCEM=%.2f\n", em_500$final_iter_count, mcem_500$final_iter_count))
cat(sprintf("Total time:      EM=%.1fs  MCEM=%.1fs  (ratio: %.1fx)\n",
            t_em_elapsed, t_mc_elapsed, t_mc_elapsed / t_em_elapsed))
