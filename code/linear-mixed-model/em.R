library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

# ===== General Linear Mixed Model (LMM): EM Algorithm Example =====
# Model: y_i = X_i beta + Z_i u_i + epsilon_i,  u_i ~ N(0, G0),  epsilon_i ~ N(0, sigma^2 I)
#        Random-effects design Z_i = X_i[, 1:k] (random intercept + random slopes for first k-1 covariates)
set.seed(2025)

m <- 50                                  # number of groups (random-effect levels)
group_sizes <- sample(c(15, 20, 25), m, replace = TRUE)
beta_real <- c(1, 2, -1, 0.5)            # fixed effects (p_fixed = 4: intercept + 3 covariates)
G0_real <- matrix(c(4, 1,
                    1, 2), nrow = 2, byrow = TRUE)   # random-effects covariance (k = 2)
sigma2_real <- 1

data <- generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real)
cat("LMM data generated: n =", data$n, "  p_fixed =", data$p_fixed, "  k =", data$k, "\n")

# Fixed initial values
beta_0 <- matrix(rep(0, data$p_fixed), ncol = 1)
G0_0 <- diag(data$k)
sigma2_0 <- 0.5

# Single EM run
em_result <- run_single_em_or_mcem(data, beta_0 = beta_0, G0_0 = G0_0, sigma2_0 = sigma2_0,
                                   max_iter = 100, tolerance = 1e-7, is_mcem = FALSE)
plot_single_iteration(em_result, title = "Single EM run: iteration trajectory")
cat("\nEM converged at iteration", em_result$final_iter_count, "\n")
cat("beta_hat:", round(em_result$final_beta_hat, 3), " (true:", beta_real, ")\n")
cat("G0_hat diagonal:", round(diag(em_result$final_G0_hat), 3), " (true: 4 2)\n")
cat("sigma2_hat:", round(em_result$final_sigma2_hat, 3), " (true: 1)\n")

# Multiple simulation runs, averaged (re-draw data each time, fixed initial values)
multiple_em <- run_multiple_em_or_mcem(data, n_sim = 500, max_iter = 100, tolerance = 1e-7,
                                       is_mcem = FALSE, reset_dataset = TRUE,
                                       beta_0 = beta_0, G0_0 = G0_0, sigma2_0 = sigma2_0)
mse_em <- get_mse(multiple_em$final_beta_hat, multiple_em$final_G0_hat, multiple_em$final_sigma2_hat,
                  beta_real, G0_real, sigma2_real)
cat("\n== Average over 500 simulation runs ==\n")
cat("beta_hat:", round(multiple_em$final_beta_hat, 3), "\n")
cat("G0_hat diagonal:", round(diag(multiple_em$final_G0_hat), 3), "  sigma2:", round(multiple_em$final_sigma2_hat, 3), "\n")
cat(sprintf("MSE: total=%.4f beta=%.4f G0=%.4f sigma2=%.4f\n",
            mse_em$mse, mse_em$mse_beta, mse_em$mse_G0, mse_em$mse_sigma2))
