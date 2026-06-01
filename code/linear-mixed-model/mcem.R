library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

# ===== General Linear Mixed Model (LMM): MCEM Algorithm Example =====
# E-step uses Monte Carlo samples to approximate updates for G0 and sigma^2; beta still uses closed-form GLS (as in EM)
set.seed(2025)

m <- 50
group_sizes <- sample(c(15, 20, 25), m, replace = TRUE)
beta_real <- c(1, 2, -1, 0.5)
G0_real <- matrix(c(4, 1,
                    1, 2), nrow = 2, byrow = TRUE)
sigma2_real <- 1

data <- generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real)
cat("LMM data generated: n =", data$n, "\n")

beta_0 <- matrix(rep(0, data$p_fixed), ncol = 1)
G0_0 <- diag(data$k)
sigma2_0 <- 0.5

# Single MCEM run (M=10 and M=100 Monte Carlo samples)
mcem_10 <- run_single_em_or_mcem(data, beta_0 = beta_0, G0_0 = G0_0, sigma2_0 = sigma2_0,
                                 max_iter = 100, tolerance = 1e-6, is_mcem = TRUE, mc_times = 10)
plot_single_iteration(mcem_10, title = "Single MCEM (M=10): iteration trajectory")

mcem_100 <- run_single_em_or_mcem(data, beta_0 = beta_0, G0_0 = G0_0, sigma2_0 = sigma2_0,
                                  max_iter = 100, tolerance = 1e-6, is_mcem = TRUE, mc_times = 100)
plot_single_iteration(mcem_100, title = "Single MCEM (M=100): iteration trajectory")

cat("\nMCEM(M=10)  beta:", round(mcem_10$final_beta_hat, 3),
    " G0 diagonal:", round(diag(mcem_10$final_G0_hat), 3), " sigma2:", round(mcem_10$final_sigma2_hat, 3), "\n")
cat("MCEM(M=100) beta:", round(mcem_100$final_beta_hat, 3),
    " G0 diagonal:", round(diag(mcem_100$final_G0_hat), 3), " sigma2:", round(mcem_100$final_sigma2_hat, 3), "\n")
cat("True values: beta = 1 2 -1 0.5,  G0 diagonal = 4 2,  sigma2 = 1\n")

# Multiple simulation runs, averaged (M=100 Monte Carlo samples)
multiple_mcem <- run_multiple_em_or_mcem(data, n_sim = 100, max_iter = 50, tolerance = 1e-6,
                                         is_mcem = TRUE, mc_times = 100, reset_dataset = TRUE,
                                         beta_0 = beta_0, G0_0 = G0_0, sigma2_0 = sigma2_0)
mse_mcem <- get_mse(multiple_mcem$final_beta_hat, multiple_mcem$final_G0_hat, multiple_mcem$final_sigma2_hat,
                    beta_real, G0_real, sigma2_real)
cat("\n== MCEM(M=100) average over 100 simulation runs ==\n")
cat("beta_hat:", round(multiple_mcem$final_beta_hat, 3), "\n")
cat("G0_hat diagonal:", round(diag(multiple_mcem$final_G0_hat), 3), "  sigma2:", round(multiple_mcem$final_sigma2_hat, 3), "\n")
cat(sprintf("MSE: total=%.4f beta=%.4f G0=%.4f sigma2=%.4f\n",
            mse_mcem$mse, mse_mcem$mse_beta, mse_mcem$mse_G0, mse_mcem$mse_sigma2))
