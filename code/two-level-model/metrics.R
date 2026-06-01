# ============================================================================
# Metrics and Log-Likelihood Functions
#   MSE computation, marginal log-likelihood evaluation, and result printing.
# ============================================================================

get_mse <- function(gamma_hat, D_hat, sigma2_hat, gamma_real, D_real, sigma2_real) {
  sigma2_hat <- as.numeric(sigma2_hat)[1]
  sigma2_real <- as.numeric(sigma2_real)[1]

  # Ensure gamma_hat and gamma_real are column vectors
  gamma_hat <- as.matrix(gamma_hat)
  gamma_real <- as.matrix(gamma_real)

  # Check dimensions
  if(nrow(gamma_hat) != nrow(gamma_real) || ncol(gamma_hat) != ncol(gamma_real)) {
    stop("gamma dimension mismatch: gamma_hat is ", nrow(gamma_hat), "x", ncol(gamma_hat),
         ", gamma_real is ", nrow(gamma_real), "x", ncol(gamma_real))
  }

  # Check D matrix dimensions
  if(nrow(D_hat) != nrow(D_real) || ncol(D_hat) != ncol(D_real)) {
    stop("D matrix dimension mismatch: D_hat is ", nrow(D_hat), "x", ncol(D_hat),
         ", D_real is ", nrow(D_real), "x", ncol(D_real))
  }

  # Compute MSE
  mse_gamma <- norm(gamma_hat - gamma_real, "2")
  mse_D <- norm(D_hat - D_real, "F")
  mse_sigma2 <- abs(sigma2_hat - sigma2_real)

  return(list(
    mse_gamma = mse_gamma,
    mse_D = mse_D,
    mse_sigma2 = mse_sigma2,
    mse = mse_gamma + mse_D + mse_sigma2
  ))
}

#' Compute the marginal log-likelihood of the observed data: ln p(y | gamma, D, sigma^2)
#' Model: y_j ~ N(X_j W_j gamma, V_j), where V_j = X_j D X_j^T + sigma^2 I_{n_j}.
#' The EM algorithm guarantees that this observed-data log-likelihood is non-decreasing,
#' so this function is used to assess convergence.
get_loglik <- function(data, gamma, D, sigma2) {
  sigma2 <- as.numeric(sigma2)[1]
  J <- data$J
  loglik <- 0

  # Ensure D is positive definite
  eigen_decomp <- eigen(D)
  if (min(eigen_decomp$values) < 1e-6) {
    eigen_decomp$values[eigen_decomp$values < 1e-6] <- 1e-6
    D <- eigen_decomp$vectors %*% diag(eigen_decomp$values) %*% t(eigen_decomp$vectors)
  }

  for (j in 1:J) {
    X_j <- get_X_j(data, j)
    y_j <- get_y_j(data, j)
    W_j <- get_W_j(data, j)
    n_j <- length(y_j)

    V_j <- X_j %*% D %*% t(X_j) + sigma2 * diag(n_j)
    residual <- y_j - X_j %*% W_j %*% gamma

    # Use Cholesky to compute log|V_j| and the quadratic form r^T V_j^{-1} r simultaneously for numerical stability
    chol_V <- tryCatch(chol(V_j), error = function(e) NULL)
    if (is.null(chol_V)) {
      V_j <- V_j + 1e-6 * diag(n_j)
      chol_V <- chol(V_j)
    }
    log_det_V_j <- 2 * sum(log(diag(chol_V)))
    quad_form <- sum(backsolve(chol_V, residual, transpose = TRUE)^2)

    loglik <- loglik - 0.5 * n_j * log(2 * pi) - 0.5 * log_det_V_j - 0.5 * quad_form
  }

  # REML (restricted / residual likelihood): add -1/2 log|sum_j W_j' Lambda_j^{-1} W_j|
  # = +1/2 log|Cov(gamma-hat)| to the ML marginal log-likelihood (evaluated at the GLS gamma),
  # and adjust the constant to -(N - p_gamma)/2 * log(2*pi).
  # This is the REML objective that is non-decreasing over REML iterations.
  if (isTRUE(getOption("hlm_reml", FALSE))) {
    p_gamma <- (data$p + 1) * (data$q + 1)
    Cinv <- get_reml_pieces(data, D, sigma2)$Cinv
    loglik <- loglik - 0.5 * as.numeric(determinant(Cinv, logarithm = TRUE)$modulus) +
      0.5 * p_gamma * log(2 * pi)
  }

  return(as.numeric(loglik))
}

#' Print a comparison of the estimated parameters against the true values for a single run
#' @param result return value of run_single_em_or_mcem
#' @param data_true data list containing the true-value fields gamma_real/D_real/sigma2_real
print_result <- function(result, data_true) {
  m <- get_mse(result$final_gamma_hat, result$final_D_hat, result$final_sigma2_hat,
               data_true$gamma_real, data_true$D_real, data_true$sigma2_real)
  cat("\nFinal results:\n")
  cat("Iterations:", result$final_iter_count, "\n")
  cat("Final log-likelihood:", tail(sapply(result$history, function(x) x$loglik), 1), "\n")
  cat(sprintf("Total MSE = %.6f (gamma=%.6f, D=%.6f, sigma2=%.6f)\n",
              m$mse, m$mse_gamma, m$mse_D, m$mse_sigma2))

  cat("\ngamma parameter comparison:\n")
  print(data.frame(
    index = seq_along(as.vector(data_true$gamma_real)),
    true_value = round(as.vector(data_true$gamma_real), 4),
    estimated = round(as.vector(result$final_gamma_hat), 4),
    difference = round(as.vector(data_true$gamma_real) - as.vector(result$final_gamma_hat), 4)
  ), row.names = FALSE)

  cat("\nD matrix - true / estimated / difference:\n")
  print(round(data_true$D_real, 4))
  print(round(result$final_D_hat, 4))
  print(round(data_true$D_real - result$final_D_hat, 4))

  cat(sprintf("\nsigma2: true=%.4f  estimated=%.4f  difference=%.4f\n",
              as.numeric(data_true$sigma2_real), as.numeric(result$final_sigma2_hat),
              as.numeric(data_true$sigma2_real) - as.numeric(result$final_sigma2_hat)))
}
