library(Matrix)
library(MASS)

# ============================================================================
# Two-Level Linear Model — Data Generation
#
#   y_j = X_j W_j gamma + X_j mu_j + epsilon_j,   j = 1, ..., J
#   mu_j ~ N(0, D)        (group-level random coefficients, i.i.d. across groups)
#   epsilon_j ~ N(0, sigma^2 I)
# ============================================================================

# Random true-parameter generation (thesis section 4.1.2 setup):
#   D = Q Lambda Q^T, where Q is a random orthogonal factor (from the QR
#   decomposition of a Gaussian matrix) and Lambda = diag(5, 6, 7) is a fixed
#   spectrum; the fixed effects gamma and the individual-level variance sigma^2
#   are drawn from uniform distributions. Call under a fixed seed (set.seed)
#   for reproducibility.
generate_true_2level_params <- function(p, q,
                                        eigenvalues = c(5, 6, 7),
                                        gamma_range = c(0, 12),
                                        sigma2_range = c(1, 5)) {
  if (p <= 0 || q < 0) stop("p must be positive and q must be non-negative")
  d <- p + 1
  if (length(eigenvalues) != d) stop("length(eigenvalues) must equal p + 1")
  Q <- qr.Q(qr(matrix(rnorm(d * d), nrow = d)))      # random orthogonal factor
  D_real <- Q %*% diag(eigenvalues) %*% t(Q)
  D_real <- (D_real + t(D_real)) / 2                 # clean numerical asymmetry
  gamma_real <- matrix(runif((p + 1) * (q + 1), gamma_range[1], gamma_range[2]), ncol = 1)
  sigma2_real <- runif(1, sigma2_range[1], sigma2_range[2])
  list(gamma_real = gamma_real, D_real = D_real, sigma2_real = sigma2_real)
}

generate_2levellm_data <- function(J, p, q, gamma_real, D_real, sigma2_real) {
  # Parameter validation
  d <- p + 1
  gamma_dim <- d * (q + 1)
  if (J <= 0 || p <= 0 || q < 0 || sigma2_real <= 0 ||
      !is.matrix(D_real) || nrow(D_real) != d || ncol(D_real) != d ||
      length(gamma_real) != gamma_dim ||
      kappa(D_real) > 1e10) {
    stop("Invalid parameters")
  }
  # Generate group sample sizes (consistent with paper section 4.1.2)
  group_sizes <- sample(c(60, 80, 100), size = J, replace = TRUE)
  sample_size <- sum(group_sizes)
  # Build the full random-effects covariance matrix
  T_real <- as.matrix(bdiag(rep(list(D_real), J)))
  # Generate random effects
  mu <- matrix(mvrnorm(n = 1, mu = rep(0, J * (p + 1)), Sigma = T_real), ncol = 1)
  # Generate group-level design matrices
  W_list <- list()
  for (j in 1:J) {
    W_j <- matrix(0, nrow = (p + 1), ncol = (p + 1) * (q + 1))
    group_covariates <- c(1, rnorm(q, mean = 0, sd = 0.5))
    for (i in 1:(p + 1)) {
      start_col <- (i - 1) * (q + 1) + 1
      W_j[i, start_col:(start_col + q)] <- group_covariates
    }
    W_list[[j]] <- W_j
  }
  W <- do.call(rbind, W_list)
  # Generate fixed-effects design matrices
  X_list <- list()
  for (group_index in 1:J) {
    raw_design_matrix <- matrix(rnorm(group_sizes[group_index] * p),
                                nrow = group_sizes[group_index])
    centered_design_matrix <- scale(raw_design_matrix,
                                    center = TRUE, scale = TRUE)
    design_matrix_with_intercept <- cbind(1, centered_design_matrix)
    X_list[[group_index]] <- design_matrix_with_intercept
  }
  X <- as.matrix(bdiag(X_list))
  # Generate error terms
  epsilon_list <- list()
  for (i in 1:J) {
    epsilon_list[[i]] <- rnorm(group_sizes[i], mean = 0, sd = sqrt(sigma2_real))
  }
  epsilon <- matrix(unlist(epsilon_list), ncol = 1)
  # Generate response variable
  beta <- W %*% gamma_real + mu
  y_real <- X %*% beta + epsilon
  y_imag <- y_real - epsilon
  # Return results
  list(
    J = J,
    p = p,
    q = q,
    group_sizes = group_sizes,
    sample_size = sample_size,
    y_imag = y_imag,
    y_real = y_real,
    X = X,
    beta = beta,
    epsilon = epsilon,
    W = W,
    mu = mu,
    gamma_real = gamma_real,
    D_real = D_real,
    sigma2_real = sigma2_real
  )
}

random_init <- function(p, q){
  gamma_dim <- (p + 1) * (q + 1)
  # Initialise gamma (consistent with generate_2levellm_data)
  gamma_init <- matrix(runif(gamma_dim, -1, 1), ncol = 1)
  # Initialise sigma2
  sigma2_init <- runif(1, 0.1, 1)
  # Generate a random-effects covariance matrix
  # Generate a random orthogonal matrix
  Q <- qr.Q(qr(matrix(rnorm((p + 1)^2), nrow = p + 1)))
  # Generate a diagonal matrix with positive entries
  D_diag <- diag(runif(p + 1, 0.1, 1))
  # Form a positive definite matrix via orthogonal transformation
  D_init <- Q %*% D_diag %*% t(Q)
  # Check condition number
  cond_num <- kappa(D_init)
  if (cond_num > 1e10) {
    warning("Random-effects covariance matrix condition number too large: ", cond_num)
  }
  # Return initial parameters
  return(list(
    gamma_init = gamma_init,
    D_init = D_init,
    sigma2_init = sigma2_init
  ))
}

bold_random_init <- function(p, q){
  gamma_dim <- (p + 1) * (q + 1)
  # Initialise gamma with a wide range (bold initialisation)
  gamma_init <- matrix(runif(gamma_dim, -10, 10), ncol = 1)
  # Initialise sigma2
  sigma2_init <- runif(1, 0.01, 10)
  # Generate a random-effects covariance matrix
  # Generate a random orthogonal matrix
  Q <- qr.Q(qr(matrix(rnorm((p + 1)^2)*5, nrow = p + 1)))
  # Generate a diagonal matrix with positive entries
  D_diag <- diag(runif(p + 1, 0.01, 10))
  # Form a positive definite matrix via orthogonal transformation
  D_init <- Q %*% D_diag %*% t(Q)
  # Check condition number
  cond_num <- kappa(D_init)
  if (cond_num > 1e10) {
    warning("Random-effects covariance matrix condition number too large: ", cond_num)
  }
  # Return initial parameters
  return(list(
    gamma_init = gamma_init,
    D_init = D_init,
    sigma2_init = sigma2_init
  ))
}
