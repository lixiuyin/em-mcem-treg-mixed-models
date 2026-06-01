library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

# Example run (consistent with paper section 4.2.2)
J <- 20
p <- 2
q <- 3
set.seed(2025)
true_params <- generate_true_2level_params(p, q)  # section 4.2.2: D = Q diag(5,6,7) Q^T (random Q); gamma, sigma^2 ~ Uniform
gamma_real <- true_params$gamma_real
D_real <- true_params$D_real
sigma2_real <- true_params$sigma2_real

# Generate simulation data
data <- generate_2levellm_data(J, p, q, gamma_real, D_real, sigma2_real)
print("Simulation data generated successfully!")

# Set initial values
gamma_0 <- matrix(rep(0.1,(p + 1)*(q + 1)), ncol = 1)
D_0 <- matrix(c(4, 2, 1,
                   2, 5, 3,
                   1, 3, 6), ncol = 3, byrow = TRUE)
sigma2_0 <- 0.1

# MCEM algorithm single run (M=10 samples)
system.time({
  mcem_result_10 <- run_single_em_or_mcem(data = data,
                                         gamma_0 = gamma_0,
                                         D_0 = D_0,
                                         sigma2_0 = sigma2_0,
                                         max_iter = 100,
                                         tolerance = 1e-6,
                                         is_mcem = TRUE,
                                         mc_times = 10)
})
print("MCEM (M=10) run complete")
plot_single_iteration(mcem_result_10, title = 'MCEM algorithm (M=10): iteration path and evaluation metrics')
plot_single_mse(mcem_result_10, title = 'MCEM algorithm (M=10): MSE evolution')

# MCEM algorithm single run (M=100 samples)
system.time({
  mcem_result_100 <- run_single_em_or_mcem(data = data,
                                          gamma_0 = gamma_0,
                                          D_0 = D_0,
                                          sigma2_0 = sigma2_0,
                                          max_iter = 100,
                                          tolerance = 1e-6,
                                          is_mcem = TRUE,
                                          mc_times = 100)
})
print("MCEM (M=100) run complete")
plot_single_iteration(mcem_result_100, title = 'MCEM algorithm (M=100): iteration path and evaluation metrics')
plot_single_mse(mcem_result_100, title = 'MCEM algorithm (M=100): MSE evolution')

# MCEM algorithm multiple simulations (M=10 samples)
system.time({
  multiple_mcem_10 <- run_multiple_em_or_mcem(data = data,
                                             n_sim = 500,
                                             max_iter = 100,
                                             tolerance = 1e-6,
                                             is_mcem = TRUE,
                                             mc_times = 10,
                                             reset_dataset = TRUE,
                                             bold_random = FALSE)
})
print("MCEM (M=10) multiple simulations complete")
plot_multiple_simulation_iterations(multiple_mcem_10, title = 'MCEM (M=10): average iteration path over 500 simulations')
plot_multiple_simulation_mse(multiple_mcem_10, title = 'MCEM (M=10): MSE evolution over 500 simulations')

# MCEM algorithm multiple simulations (M=100 samples)
system.time({
  multiple_mcem_100 <- run_multiple_em_or_mcem(data = data,
                                              n_sim = 500,
                                              max_iter = 100,
                                              tolerance = 1e-6,
                                              is_mcem = TRUE,
                                              mc_times = 100,
                                              reset_dataset = TRUE,
                                              bold_random = FALSE)
})
print("MCEM (M=100) multiple simulations complete")
plot_multiple_simulation_iterations(multiple_mcem_100, title = 'MCEM (M=100): average iteration path over 500 simulations')
plot_multiple_simulation_mse(multiple_mcem_100, title = 'MCEM (M=100): MSE evolution over 500 simulations')