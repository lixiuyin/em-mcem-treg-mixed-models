library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")
set.seed(2025)  # reproducibility: matches LMM em.R/mcem.R and the figure-producing drivers
# Basic parameter settings
n <- 500  # sample size
p <- 2    # number of features
t_df <- 10  # degrees of freedom for error term
t_mu <- 0  # location parameter of error term
t_sigma2_real <- 0.4  # true scale parameter of error term
beta_real <- c(2, 3, 5)  # true regression coefficients
fixed_init_beta <- c(0, 0, 0)  # fixed initialization values
fixed_init_sigma2 <- 0.1 # fixed initialization value

# Parameter initialization
initial_params <- random_init(p)

# Run MCEM once (MC sample size = 1)
single_mcem_data <- generate_t_simulated_data(n, p, t_df, t_mu, t_sigma2_real, beta_real)
mcem_result <- run_single_em_or_mcem(
  X = single_mcem_data$X,
  y = single_mcem_data$y_real,
  t_df = t_df,
  beta_0 = fixed_init_beta,
  sigma2_0 = fixed_init_sigma2,
  max_iter = 100,
  tolerance = 1e-7,
  is_mcem = TRUE,
  mc_times = 1
)
single_mcem_mse <- get_mse(mcem_result$final_beta_hat, mcem_result$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_single_iteration(mcem_result, c(2, 3, 5), 0.4, title = "Iteration path for a single MCEM run (MC sample size = 1)")
# Run MCEM once (MC sample size = 100)
mcem_result <- run_single_em_or_mcem(
  X = single_mcem_data$X,
  y = single_mcem_data$y_real,
  t_df = t_df,
  beta_0 = fixed_init_beta,
  sigma2_0 = fixed_init_sigma2,
  max_iter = 100,
  tolerance = 1e-7,
  is_mcem = TRUE,
  mc_times = 100
)
single_mcem_mse <- get_mse(mcem_result$final_beta_hat, mcem_result$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_single_iteration(mcem_result, c(2, 3, 5), 0.4, title = "Iteration path for a single MCEM run (MC sample size = 100)")

# Run EM algorithm 500 times, regenerating data each time
system.time({
  multiple_em_results <- run_multiple_em_or_mcem(n_sim = 500,
                                               max_iter = 100,
                                               tolerance = 1e-7,
                                               is_mcem = FALSE,
                                               reset_dataset = TRUE,
                                               beta_0 = fixed_init_beta,
                                               sigma2_0 = fixed_init_sigma2)
})
multiple_em_mse <- get_mse(multiple_em_results$final_beta_hat, multiple_em_results$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_multiple_simulation_iterations(multiple_em_results, c(2, 3, 5), 0.4, title = "Mean iteration path for EM algorithm over 500 simulations")
plot_multiple_simulation_mse(multiple_em_results, c(2, 3, 5), 0.4, title = "MSE over 500 simulations of the EM algorithm")
# Run MCEM algorithm, regenerating data each time
system.time({
  multiple_mcem_results <- run_multiple_em_or_mcem(n_sim = 50,
                                               max_iter = 100,
                                               tolerance = 1e-7,
                                               is_mcem = TRUE,
                                               mc_times = 100,
                                               reset_dataset = TRUE,
                                               beta_0 = fixed_init_beta,
                                               sigma2_0 = fixed_init_sigma2)
})
multiple_mcem_mse <- get_mse(multiple_mcem_results$final_beta_hat, multiple_mcem_results$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_multiple_simulation_iterations(multiple_mcem_results, c(2, 3, 5), 0.4, title = "Mean iteration path for MCEM algorithm over 50 simulations")
plot_multiple_simulation_mse(multiple_mcem_results, c(2, 3, 5), 0.4, title = "MSE over 50 simulations of the MCEM algorithm")
