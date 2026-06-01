library(Matrix)
library(MASS)

# ============================================================================
# General Linear Mixed Model (LMM) Data Generation
#
#   y_i = X_i beta + Z_i u_i + epsilon_i,   i = 1, ..., m
#   u_i ~ N(0, G0)   (k-dimensional random effects, i.i.d. across groups)
#   epsilon_i ~ N(0, sigma^2 I_{n_i})
#
# Fixed-effects design X_i is n_i x p_fixed (includes intercept column).
# Random-effects design Z_i = X_i[, 1:k] (random intercept + random slopes for first k-1 covariates).
# This is the standard "random coefficient" mixed model, covering random intercepts,
# random slopes, and other common specifications.
#
# The two-level linear model is a special case (with X_i^{fixed}=X_j W_j, Z_i=X_j, u_i=mu_j, G0=D).
# ============================================================================

generate_lmm_data <- function(m, group_sizes, beta_real, G0_real, sigma2_real) {
  p_fixed <- length(beta_real)
  k <- nrow(G0_real)
  # Input validation at system boundary
  if (m <= 0 || any(group_sizes <= 0)) stop("m and all group sizes must be positive")
  if (length(group_sizes) != m) stop("length of group_sizes must equal m")
  if (sigma2_real <= 0) stop("sigma2_real must be positive")
  if (nrow(G0_real) != ncol(G0_real)) stop("G0_real must be a square matrix")
  if (k > p_fixed) stop("random-effects dimension k cannot exceed fixed-effects dimension p_fixed")
  if (kappa(G0_real) > 1e10) stop("G0_real condition number is too large (near singular)")

  n <- sum(group_sizes)
  beta_real <- matrix(beta_real, ncol = 1)

  X_list <- vector("list", m)
  u <- matrix(0, nrow = m * k, ncol = 1)
  y <- numeric(n)
  row_end <- 0
  for (i in 1:m) {
    n_i <- group_sizes[i]
    # Fixed-effects design: intercept + (p_fixed-1) standard-normal covariates
    covariates <- matrix(rnorm(n_i * (p_fixed - 1)), nrow = n_i)
    X_i <- cbind(1, covariates)                 # n_i x p_fixed
    Z_i <- X_i[, 1:k, drop = FALSE]             # n_i x k random-effects design
    u_i <- matrix(mvrnorm(1, mu = rep(0, k), Sigma = G0_real), ncol = 1)
    eps_i <- rnorm(n_i, mean = 0, sd = sqrt(sigma2_real))
    y_i <- X_i %*% beta_real + Z_i %*% u_i + eps_i

    X_list[[i]] <- X_i
    u[((i - 1) * k + 1):(i * k), 1] <- u_i
    y[(row_end + 1):(row_end + n_i)] <- as.numeric(y_i)
    row_end <- row_end + n_i
  }
  X <- do.call(rbind, X_list)                   # n x p_fixed (stacked; fixed effects shared across groups)

  list(
    m = m,
    k = k,
    p_fixed = p_fixed,
    group_sizes = group_sizes,
    n = n,
    X = X,
    y = y,
    u = u,
    beta_real = beta_real,
    G0_real = G0_real,
    sigma2_real = sigma2_real
  )
}

# Conservative random initialisation (moderate spread)
random_init <- function(p_fixed, k) {
  beta_init <- matrix(runif(p_fixed, -1, 1), ncol = 1)
  sigma2_init <- runif(1, 0.1, 1)
  Q <- qr.Q(qr(matrix(rnorm(k * k), nrow = k)))
  G0_init <- Q %*% diag(runif(k, 0.1, 1), nrow = k) %*% t(Q)
  list(beta_init = beta_init, G0_init = G0_init, sigma2_init = sigma2_init)
}

# Aggressive random initialisation (larger spread)
bold_random_init <- function(p_fixed, k) {
  beta_init <- matrix(runif(p_fixed, -10, 10), ncol = 1)
  sigma2_init <- runif(1, 0.01, 10)
  Q <- qr.Q(qr(matrix(rnorm(k * k) * 5, nrow = k)))
  G0_init <- Q %*% diag(runif(k, 0.01, 10), nrow = k) %*% t(Q)
  list(beta_init = beta_init, G0_init = G0_init, sigma2_init = sigma2_init)
}
