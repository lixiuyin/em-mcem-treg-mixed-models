# ============================================================================
# MCEM Update Functions
#   Monte Carlo sampling and M-step implementations for the MCEM algorithm.
#   Model: y_j = X_j W_j gamma + X_j mu_j + epsilon_j,
#          mu_j ~ N(0, D),  epsilon_j ~ N(0, sigma^2 I)
# ============================================================================

# Sample from the random-effects posterior. When reml=TRUE the sampling covariance is inflated
# from V_j to V_j + B_j C B_j', so that the sample second moment E[mu mu'] equals the REML
# quantity m_j m_j' + V_j + B_j C B_j' (used in the MCEM update of D).
# The sigma^2 update calls this with reml=FALSE (ML sampling): its REML fixed-effect correction
# tr(A_j C A_j') is added explicitly inside mcem_update_sigma2 instead, to avoid double counting.
get_mu_samples <- function(data, gamma_k, D_k, sigma2_k, M, J, reml = FALSE) {
  mu_samples <- vector("list", J)
  rp <- if (reml) get_reml_pieces(data, D_k, sigma2_k) else NULL
  for (j in 1:J) {
    mu_jk_star <- get_mu_jk_star(data, j, gamma_k, D_k, sigma2_k)
    Sigma_j <- get_V_jk_star(data, j, D_k, sigma2_k)
    if (reml) {
      W_j <- get_W_j(data, j)
      B_j <- D_k %*% rp$Linv[[j]] %*% W_j
      Sigma_j <- Sigma_j + B_j %*% rp$C %*% t(B_j)
    }
    samples <- mvrnorm(n = M, mu = as.numeric(mu_jk_star), Sigma = Sigma_j)
    # Explicitly normalise to M x d to avoid wrong transpose direction when M=1 causes mvrnorm to return a vector
    mu_samples[[j]] <- t(matrix(samples, nrow = M))   # d x M
  }
  return(mu_samples)
}

# MCEM update for gamma: uses the same closed-form GLS as EM (eq. 4-25b).
# The strict 4-25 + MC form is mathematically correct, but the convergence rate approaches 1
# when variance components are large, requiring thousands of steps; in practice the closed-form
# 4-25b is used - one step to GLS = MLE - and MC is applied only to the D and sigma^2 updates.
mcem_update_gamma <- function(data, gamma_k, D_k, sigma2_k, M, mu_samples) {
  return(em_update_gamma(data, D_k, sigma2_k))
}

# MCEM update for D (paper eq. 4-42). Under a flat non-informative prior the divisor is M*J (consistent with the EM /J).
mcem_update_D <- function(data, M, mu_samples) {
  M <- as.integer(M[1])
  J <- data$J
  p <- data$p
  D_new <- matrix(0, nrow = p + 1, ncol = p + 1)

  # Compute Monte Carlo average
  for (m in 1:M) {
    for (j in 1:J) {
      mu_jm <- mu_samples[[j]][, m, drop=FALSE]
      D_new <- D_new + mu_jm %*% t(mu_jm)
    }
  }
  D_new <- D_new / (M * J)

  # Enforce symmetry
  D_new <- (D_new + t(D_new)) / 2

  # Enforce positive definiteness (eigenvalue lower-bound clipping only; no extra 1e-6*I to avoid
  # persistent upward bias in D; consistent with the general LMM)
  eigen_decomp <- eigen(D_new, symmetric = TRUE)
  min_eigenvalue <- min(eigen_decomp$values)
  if (min_eigenvalue < 1e-6) {
    eigen_decomp$values[eigen_decomp$values < 1e-6] <- 1e-6
    D_new <- eigen_decomp$vectors %*% diag(eigen_decomp$values) %*% t(eigen_decomp$vectors)
  }

  return(D_new)
}

# MCEM update for sigma2. The mu samples are drawn WITHOUT REML inflation (ML sampling); under REML
# the fixed-effect uncertainty is instead added explicitly as M * sum_j tr(A_j C A_j') with
# A_j = X_j W_j - X_j D Lambda_j^{-1} W_j, matching the EM REML update and avoiding double counting
# (mirrors the general LMM mcem_update_sigma2).
mcem_update_sigma2 <- function(data, gamma_k, M, mu_samples, D = NULL, sigma2 = NULL, reml = FALSE) {
  J <- data$J
  N <- data$sample_size
  sigma2_new <- 0
  fixed_effect_total <- 0
  if (reml) {
    if (is.null(D) || is.null(sigma2)) stop("D and sigma2 are required for REML MCEM sigma2 update")
    rp <- get_reml_pieces(data, D, sigma2)
    for (j in 1:J) {
      X_j <- get_X_j(data, j); W_j <- get_W_j(data, j)
      A_j <- X_j %*% W_j - X_j %*% D %*% rp$Linv[[j]] %*% W_j
      fixed_effect_total <- fixed_effect_total + sum(diag(A_j %*% rp$C %*% t(A_j)))
    }
  }
  for (m in 1:M) {
    for (j in 1:J) {
      X_j <- get_X_j(data, j)
      y_j <- get_y_j(data, j)
      W_j <- get_W_j(data, j)
      mu_jm <- mu_samples[[j]][, m, drop=FALSE]
      residual <- y_j - X_j %*% W_j %*% gamma_k - X_j %*% mu_jm
      sigma2_new <- sigma2_new + sum(residual^2)
    }
  }
  if (reml) sigma2_new <- sigma2_new + M * fixed_effect_total
  sigma2_new <- sigma2_new / (M * N)
  return(sigma2_new)
}

mcem_update_all <- function(data, gamma_k, D_k, sigma2_k, mc_times){
  # Aligned with the ECM structure of EM: re-sample using the latest parameters before each
  # update step, to prevent large spikes in D and sigma^2 at the first step when gamma_0 is
  # far from the true value.
  reml <- isTRUE(getOption("hlm_reml", FALSE))
  # 1) gamma update is already a closed-form GLS and does not depend on mu_samples
  gamma_new <- mcem_update_gamma(data, gamma_k, D_k, sigma2_k, mc_times, NULL)
  # 2) D update: sample under (gamma_new, D_k, sigma^2_k); inflate covariance under REML to correct gamma-hat uncertainty
  mu_samples_for_D <- get_mu_samples(data, gamma_new, D_k, sigma2_k, mc_times, data$J, reml = reml)
  D_new <- mcem_update_D(data, mc_times, mu_samples_for_D)
  # 3) sigma^2 update: re-sample under (gamma_new, D_new, sigma^2_k) with ML sampling; under REML the
  #    gamma-hat uncertainty is added explicitly inside mcem_update_sigma2 (tr(A_j C A_j')), not by inflation
  mu_samples_for_sigma <- get_mu_samples(data, gamma_new, D_new, sigma2_k, mc_times, data$J)
  sigma2_new <- mcem_update_sigma2(data, gamma_new, mc_times, mu_samples_for_sigma,
                                   D = D_new, sigma2 = sigma2_k, reml = reml)
  return(list(gamma_new = gamma_new, D_new = D_new, sigma2_new = sigma2_new))
}
