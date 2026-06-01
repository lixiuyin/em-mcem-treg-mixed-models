library(this.path)
source(file.path(this.dir(), "data_generation.R"), encoding = "UTF-8")

# ============================================================================
# Two-Level Linear Model: EM / MCEM Estimation
#   y_j = X_j W_j gamma + X_j mu_j + epsilon_j,  mu_j ~ N(0, D),  epsilon_j ~ N(0, sigma^2 I)
#   Latent variables = group-level random coefficients mu = (mu_1, ..., mu_J)
#
#   Prior: flat non-informative pi(gamma, D, sigma^2) proportional to 1
#   (EM M-step divides D by J; MCEM divides by M*J).
# ============================================================================

# --- Split module sources (dependency order) ---
source(file.path(this.dir(), "helpers.R"),      encoding = "UTF-8")   # data access: get_X_j, get_y_j, get_W_j, ...
source(file.path(this.dir(), "em_updates.R"),   encoding = "UTF-8")   # EM E-step / M-step: em_update_gamma, em_update_D, ...
source(file.path(this.dir(), "mcem_updates.R"), encoding = "UTF-8")   # MCEM sampling and updates: get_mu_samples, mcem_update_all, ...
source(file.path(this.dir(), "metrics.R"),      encoding = "UTF-8")   # MSE, log-likelihood, print_result
source(file.path(this.dir(), "simulation.R"),   encoding = "UTF-8")   # run_single_em_or_mcem, run_multiple_em_or_mcem
source(file.path(this.dir(), "plotting.R"),     encoding = "UTF-8")   # plot_single_iteration, plot_multiple_simulation_*, ...
