# ============================================================================
# Data Access Helpers
#   Functions for extracting group-level sub-matrices and vectors from the
#   packed data structure produced by generate_2levellm_data().
# ============================================================================

get_X_j <- function(data, j) {
  group_sizes <- data$group_sizes
  p <- data$p
  # Compute row indices for group j
  if (j == 1) {
    row_start <- 1
  } else {
    row_start <- sum(group_sizes[1:(j-1)]) + 1
  }
  row_end <- sum(group_sizes[1:j])
  # Compute column indices for group j
  col_start <- (j-1)*(p+1) + 1
  col_end <- j*(p+1)
  # Extract the design matrix for group j
  X_j <- data$X[row_start:row_end, col_start:col_end]
  return(X_j)
}

get_y_j <- function(data, j) {
  group_sizes <- data$group_sizes
  # Compute row indices for group j
  if (j == 1) {
    row_start <- 1
  } else {
    row_start <- sum(group_sizes[1:(j-1)]) + 1
  }
  row_end <- sum(group_sizes[1:j])
  # Extract the response variable for group j
  y_j <- data$y_real[row_start:row_end, 1]

  return(y_j)
}

get_mu_j <- function(data, j){
  p <- data$p
  mu <- data$mu
  return(mu[((j-1)*(p+1) + 1):(j*(p+1)), 1])
}

get_W_j <- function(data, j) {
  p <- data$p
  q <- data$q
  # Compute row indices of W_j within the W matrix
  row_start <- (j-1)*(p+1) + 1
  row_end <- j*(p+1)
  # Compute column indices of W_j within the W matrix
  col_start <- 1
  col_end <- (p+1)*(q+1)
  # Extract the group-level design matrix for group j
  W_j <- data$W[row_start:row_end, col_start:col_end]
  return(W_j)
}

get_D_k <- function(data, T_k){
  return(T_k[1:(data$p + 1), 1:(data$p + 1)])
}

get_inv_txxplussigma2invd <- function(data, j, D_k, sigma2_k) {
  sigma2_k <- as.numeric(sigma2_k)[1]
  X_j <- get_X_j(data, j)

  # Use a numerically stable inversion (tryCatch return value ensures fallback takes effect on singularity)
  inv_matrix <- tryCatch({
    # First attempt direct solve
    solve(t(X_j) %*% X_j + ginv(D_k) * sigma2_k)
  }, error = function(e) {
    # If that fails, add a small regularisation term and use ginv
    A <- t(X_j) %*% X_j + ginv(D_k) * sigma2_k
    A <- A + 1e-6 * diag(nrow(A))
    ginv(A)
  })

  return(inv_matrix)
}

get_mu_jk_star <- function(data, j, gamma_k, D_k, sigma2_k) {
  sigma2_k <- as.numeric(sigma2_k)[1]
  X_j <- get_X_j(data, j)
  y_j <- get_y_j(data, j)
  W_j <- get_W_j(data, j)

  # Use a numerically stable computation (tryCatch return value ensures fallback takes effect on singularity)
  mu_jk_star <- tryCatch({
    part_1 <- get_inv_txxplussigma2invd(data, j, D_k, sigma2_k)
    part_2 <- t(X_j) %*% (y_j - X_j %*% W_j %*% gamma_k)
    part_1 %*% part_2
  }, error = function(e) {
    A <- t(X_j) %*% X_j + ginv(D_k) * sigma2_k
    A <- A + 1e-6 * diag(nrow(A))
    b <- t(X_j) %*% (y_j - X_j %*% W_j %*% gamma_k)
    ginv(A) %*% b
  })

  return(mu_jk_star)
}

get_V_jk_star <- function(data, j, D_k, sigma2_k) {
  sigma2_k <- as.numeric(sigma2_k)[1]

  # Use tryCatch return value to ensure the fallback takes effect on singularity
  V_jk_star <- tryCatch({
    part_1 <- get_inv_txxplussigma2invd(data, j, D_k, sigma2_k)
    part_1 * sigma2_k
  }, error = function(e) {
    X_j <- get_X_j(data, j)
    A <- t(X_j) %*% X_j + ginv(D_k) * sigma2_k
    A <- A + 1e-6 * diag(nrow(A))
    ginv(A) * sigma2_k
  })

  return(V_jk_star)
}
