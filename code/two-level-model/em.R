library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")
# Example run (consistent with paper section 4.1.2)
J <- 20
p <- 2
q <- 3
set.seed(2025)
true_params <- generate_true_2level_params(p, q)  # section 4.1.2: D = Q diag(5,6,7) Q^T (random Q); gamma, sigma^2 ~ Uniform
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

# EM algorithm single run
system.time({ 
  em_result <- run_single_em_or_mcem(data = data,
                                    gamma_0 = gamma_0,
                                    D_0 = D_0,
                                    sigma2_0 = sigma2_0,
                                    max_iter = 100,
                                    tolerance = 1e-6,
                                    is_mcem = FALSE) 
})
# Plot single-run results
plot_single_iteration(em_result, title = 'EM algorithm single-run iteration path')
plot_single_mse(em_result, title = 'EM algorithm single-run MSE evolution')

system.time({
  multiple_em_result <- run_multiple_em_or_mcem(data = data,
                                                n_sim = 500,
                                                max_iter = 50,
                                                tolerance = 1e-6,
                                                is_mcem = FALSE,
                                                reset_dataset = TRUE,
                                                gamma_0 = gamma_0,
                                                D_0 = D_0,
                                                sigma2_0 = sigma2_0)
})
# Plot average iteration path over 500 simulations
plot_multiple_simulation_iterations(multiple_em_result, title = "Average iteration path over 500 simulations")
plot_multiple_simulation_mse(multiple_em_result, title = "MSE evolution averaged over 500 simulations")
multiple_em_mse <- get_mse(multiple_em_result$final_gamma_hat,
                           multiple_em_result$final_D_hat,
                           multiple_em_result$final_sigma2_hat,
                           gamma_real,
                           D_real,
                           sigma2_real)

# Study sensitivity of EM algorithm to initial values
multiple_em_result_normal_init <- run_multiple_em_or_mcem(data = data,
                                                          n_sim = 500,
                                                          max_iter = 50,
                                                          tolerance = 1e-6,
                                                          is_mcem = FALSE,
                                                          reset_dataset = TRUE,
                                                          bold_random = FALSE)

multiple_em_result_bold_init <- run_multiple_em_or_mcem(data = data,
                                                        n_sim = 500,
                                                        max_iter = 50,
                                                        tolerance = 1e-6,
                                                        is_mcem = FALSE,
                                                        reset_dataset = TRUE,
                                                        bold_random = TRUE)


## Overall performance across multiple simulations (based on actual algorithm output; fake_* functions no longer used)
plot_multiple_simulation_iterations(multiple_em_result, title = "EM algorithm: average iteration path over 500 simulations")
plot_multiple_simulation_mse(multiple_em_result, title = "MSE evolution averaged over 500 simulations")

## True iteration paths under different initialisation strategies
plot_multiple_simulation_iterations(multiple_em_result_normal_init, title = "Standard-distribution initialisation: iteration path")
plot_multiple_simulation_mse(multiple_em_result_normal_init, title = "Standard-distribution initialisation: MSE evolution")
multiple_em_result_normal_init_mse <- get_mse(multiple_em_result_normal_init$final_gamma_hat,
                                              multiple_em_result_normal_init$final_D_hat,
                                              multiple_em_result_normal_init$final_sigma2_hat,
                                              gamma_real,
                                              D_real,
                                              sigma2_real)

plot_multiple_simulation_iterations(multiple_em_result_bold_init, title = "Wide-range initialisation: iteration path")
plot_multiple_simulation_mse(multiple_em_result_bold_init, title = "Wide-range initialisation: MSE evolution")
multiple_em_result_bold_init_mse <- get_mse(multiple_em_result_bold_init$final_gamma_hat,
                                            multiple_em_result_bold_init$final_D_hat,
                                            multiple_em_result_bold_init$final_sigma2_hat,
                                            gamma_real,
                                            D_real,
                                            sigma2_real)

## Comparison with MCEM should use actual MCEM runs; fake data substitution is no longer used
multiple_mcem_result_10 <- run_multiple_em_or_mcem(data = data,
                                                   n_sim = 500,
                                                   max_iter = 50,
                                                   tolerance = 1e-6,
                                                   is_mcem = TRUE,
                                                   mc_times = 10,
                                                   reset_dataset = TRUE,
                                                   gamma_0 = gamma_0,
                                                   D_0 = D_0,
                                                   sigma2_0 = sigma2_0)
plot_multiple_simulation_iterations(multiple_mcem_result_10, title = "MCEM algorithm (M=10): iteration path and evaluation metrics")
