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

# Run EM or MCEM once
single_em_data <- generate_t_simulated_data(n, p, t_df, t_mu, t_sigma2_real, beta_real)
em_result <- run_single_em_or_mcem(
  X = single_em_data$X,
  y = single_em_data$y_real,
  t_df = t_df,
  beta_0 = fixed_init_beta,
  sigma2_0 = fixed_init_sigma2,
  max_iter = 100,
  tolerance = 1e-7,
  is_mcem = FALSE
)
single_em_mse <- get_mse(em_result$final_beta_hat, em_result$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_single_iteration(em_result, c(2, 3, 5), 0.4, title = "Iteration path for a single EM run")
plot_single_mse(em_result, c(2, 3, 5), 0.4, title = "MSE over iterations for a single EM run")

# Run EM algorithm 500 times, regenerating data each time
multiple_em_results <- run_multiple_em_or_mcem(n_sim = 500,
                                               max_iter = 1000,
                                               tolerance = 1e-7,
                                               is_mcem = FALSE,
                                               reset_dataset = TRUE,
                                               beta_0 = fixed_init_beta,
                                               sigma2_0 = fixed_init_sigma2)
multiple_em_mse <- get_mse(multiple_em_results$final_beta_hat, multiple_em_results$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_multiple_simulation_iterations(multiple_em_results, c(2, 3, 5), 0.4, title = "Mean iteration path for EM algorithm over 500 simulations")
plot_multiple_simulation_mse(multiple_em_results, c(2, 3, 5), 0.4, title = "MSE over 500 simulations of the EM algorithm")

# Study sensitivity to initial values
normal_init_em <- run_multiple_em_or_mcem(n_sim = 500,
                                          max_iter = 1000,
                                          tolerance = 1e-7,
                                          bold_random = FALSE,
                                          fixed_data = NULL)
normal_init_mse <- get_mse(normal_init_em$final_beta_hat, normal_init_em$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_multiple_simulation_iterations(normal_init_em, c(2, 3, 5), 0.4, title = "Mean iteration path with conservative initialization")
plot_multiple_simulation_mse(normal_init_em, c(2, 3, 5), 0.4, title = "MSE with conservative initialization of EM algorithm")

bold_init_em <- run_multiple_em_or_mcem(n_sim = 500,
                                          max_iter = 1000,
                                          tolerance = 1e-7,
                                          bold_random = TRUE,
                                          fixed_data = NULL)
bold_init_mse <- get_mse(bold_init_em$final_beta_hat, bold_init_em$final_sigma2_hat, c(2, 3, 5), 0.4)
plot_multiple_simulation_iterations(bold_init_em, c(2, 3, 5), 0.4, title = "Mean iteration path with bold random initialization")
plot_multiple_simulation_mse(bold_init_em, c(2, 3, 5), 0.4, title = "MSE with bold random initialization of EM algorithm")

# Study the effect of different degrees of freedom
# Vary the df in the EM update formula without changing the data-generating df
various_df_data <- generate_t_simulated_data(n, p, t_df, t_mu, t_sigma2_real, beta_real)
df_study_fixed_data <- study_df_effect(df_range = 5:105, bold_random = FALSE, specific_data = various_df_data)
grid.arrange(df_study_fixed_data$plots$error_plot, df_study_fixed_data$plots$iter_plot, ncol = 1)

# Vary the df in both the EM update formula and the data-generating distribution
df_study_flexible_data <- study_df_effect(df_range = 5:105, bold_random = FALSE)
grid.arrange(df_study_flexible_data$plots$error_plot, df_study_flexible_data$plots$iter_plot, ncol = 1)
