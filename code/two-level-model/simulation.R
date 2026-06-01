# ============================================================================
# Simulation Runners
#   Single-run and multi-simulation EM/MCEM drivers with convergence tracking.
# ============================================================================

run_single_em_or_mcem <- function(data,
                                  gamma_0 = NULL,
                                  D_0 = NULL,
                                  sigma2_0 = NULL,
                           max_iter = 100,
                                  tolerance = 1e-6,
                                  bold_random = FALSE,
                           is_mcem = FALSE,
                                  mc_times = 100) {
  # Parameter validation
  if (max_iter <= 0 || tolerance <= 0 || mc_times <= 0) {
    stop("Iteration parameters must be positive")
  }

  p <- data$p
  q <- data$q

  # Only draw random initial values when they are actually missing, to avoid wasteful consumption
  # of the random-number stream (ensures reproducibility when a seed is set)
  if (is.null(gamma_0) || is.null(D_0) || is.null(sigma2_0)) {
    params_init <- if (bold_random) bold_random_init(p, q) else random_init(p, q)
    if (is.null(gamma_0)) {
      cat("No initial value supplied for gamma; using random initialisation\n")
      gamma_0 <- params_init$gamma_init
    }
    if (is.null(D_0)) {
      cat("No initial value supplied for D; using random initialisation\n")
      D_0 <- params_init$D_init
    }
    if (is.null(sigma2_0)) {
      cat("No initial value supplied for sigma^2; using random initialisation\n")
      sigma2_0 <- params_init$sigma2_init
    }
  }

  # Initialise history record
  history_list <- vector("list", length = max_iter + 1)  # extra slot to hold initial values
  for (k in 1:(max_iter + 1)) {
    history_list[[k]] <- list(
      gamma = rep(NA, (p + 1) * (q + 1)),
      D = matrix(NA, nrow = p + 1, ncol = p + 1),
      sigma2 = NA_real_,
      loglik = NA_real_
    )
  }

  # Print algorithm info
  cat("Running", ifelse(is_mcem, "MCEM", "EM"), "algorithm",
      ifelse(is_mcem, paste("MC samples:", mc_times), ""), "\n")

  gamma_hat <- gamma_0
  D_hat <- D_0
  sigma2_hat <- sigma2_0

  history_list[[1]]$gamma <- gamma_0
  history_list[[1]]$D <- D_0
  history_list[[1]]$sigma2 <- sigma2_0
  history_list[[1]]$loglik <- get_loglik(data, gamma_0, D_0, sigma2_0)

  # Main loop
  iter_count <- 0  # actual iteration counter
  for (idx in 2:(max_iter + 1)) {

    # Update parameters
    if (is_mcem) {
      params_new <- mcem_update_all(data, gamma_hat, D_hat, sigma2_hat, mc_times)
      gamma_new <- params_new$gamma_new
      D_new <- params_new$D_new
      sigma2_new <- params_new$sigma2_new
    } else {
      gamma_new <- em_update_gamma(data, D_hat, sigma2_hat)
      D_new <- em_update_D(data, gamma_new, D_hat, sigma2_hat)
      sigma2_new <- em_update_sigma2(data, gamma_new, D_new, sigma2_hat)
    }

    # Compute convergence metric
    max_diff <- max(norm(gamma_new - gamma_hat, '2'),
                    norm(D_new - D_hat, 'F'),
                    abs(sigma2_new - sigma2_hat))

    # Compute log-likelihood
    loglik <- get_loglik(data, gamma_new, D_new, sigma2_new)

    # Store updated parameters
    gamma_hat <- gamma_new
    D_hat <- D_new
    sigma2_hat <- sigma2_new

    # Record history
    history_list[[idx]]$gamma <- gamma_hat
    history_list[[idx]]$D <- D_hat
    history_list[[idx]]$sigma2 <- sigma2_hat
    history_list[[idx]]$loglik <- loglik

    iter_count <- iter_count + 1  # increment iteration counter
    # Print progress
    cat("Iteration", iter_count, "max change:", max_diff, "log-likelihood:", loglik, "\n")

    # Check convergence
    if (max_diff < tolerance) {
      cat("Converged after", iter_count, "iterations\n")
      break
    }
    else if (iter_count == max_iter)  {
      cat("Reached maximum iterations without convergence\n")
    }
  }

  cat("Final estimates:\n",
      "\nsigma2 = ", round(sigma2_hat, 4),
      "iterations:", iter_count, "\n")

  # Return results
  result <- list(
    final_iter_count = iter_count,
    final_gamma_hat = gamma_hat,
    final_D_hat = D_hat,
    final_sigma2_hat = sigma2_hat,
    history = history_list[1:(iter_count + 1)],
    loglik = loglik,
    data = data
  )

  return(result)
}

#' Run EM/MCEM algorithm multiple times and average results as parameter estimates
#' @param data data list
#' @param n_sim number of simulations
#' @param max_iter maximum number of iterations
#' @param tolerance convergence tolerance
#' @param is_mcem whether to use MCEM
#' @param mc_times number of MC samples
#' @param reset_dataset whether to regenerate the dataset each simulation
#' @param bold_random whether to use the wide-range (bold) random initialisation
#' @param gamma_0 initial value of gamma
#' @param D_0 initial D matrix
#' @param sigma2_0 initial value of sigma2
#' @param fixed_data fixed dataset (if not NULL, this dataset is used)
#' @return list containing averages over multiple simulations
run_multiple_em_or_mcem <- function(data,
                                    n_sim = 50,
                                    max_iter = 100,
                                    tolerance = 1e-6,
                                    is_mcem = FALSE,
                                    mc_times = 100,
                                    reset_dataset = FALSE,
                                    bold_random = FALSE,
                                    gamma_0 = NULL,
                                    D_0 = NULL,
                                    sigma2_0 = NULL,
                                    fixed_data = NULL) {
  # Parameter validation
  if (n_sim <= 0 || max_iter <= 0 || tolerance <= 0 || mc_times <= 0) {
    stop("Iteration parameters must be positive")
  }

  # Get parameter dimensions
  J <- data$J
  p <- data$p
  q <- data$q
  gamma_dim <- (p + 1) * (q + 1)
  D_dim <- p + 1

  # Initialise result storage
  details <- vector("list", n_sim)
  final_iter_counts <- integer(n_sim)
  final_gamma_hats <- matrix(0, nrow = gamma_dim, ncol = n_sim)
  final_D_hats <- array(0, dim = c(D_dim, D_dim, n_sim))
  final_sigma2_hats <- numeric(n_sim)

  # Use fixed dataset if provided
  if (!is.null(fixed_data)) {
    cat("Using fixed dataset\n")
    data <- fixed_data
    if (reset_dataset) {
      stop("A fixed dataset is already provided; reset_dataset cannot be TRUE simultaneously")
    }
  }

  # Retrieve true parameter values
  gamma_real <- data$gamma_real
  D_real <- data$D_real
  sigma2_real <- data$sigma2_real

  # Run multiple simulations
  for (i in 1:n_sim) {
    cat(sprintf("\nRunning simulation %d...\n", i))

    # Regenerate dataset if requested
    if (reset_dataset) {
      data <- generate_2levellm_data(J = J, p = p, q = q,
                                      gamma_real = gamma_real, D_real = D_real,
                                      sigma2_real = sigma2_real)
      cat("Simulation data regenerated\n")
    }

    # Set initial values
    # Fix: the original implementation wrote random initial values back into the function
    # parameters gamma_0/D_0/sigma2_0, which caused is.null() to be FALSE from the second
    # simulation onward, making all simulations reuse the first random initial values.
    # Use per-simulation local variables *_init_i instead, ensuring fresh draws when NULL.
    if (bold_random){
      params_init <- bold_random_init(p, q)
    } else {
      params_init <- random_init(p, q)
    }
    gamma_init_i  <- if (is.null(gamma_0))  params_init$gamma_init  else gamma_0
    D_init_i      <- if (is.null(D_0))      params_init$D_init      else D_0
    sigma2_init_i <- if (is.null(sigma2_0)) params_init$sigma2_init else sigma2_0

    # Run EM/MCEM algorithm
    result <- run_single_em_or_mcem(data = data,
                           gamma_0 = gamma_init_i,
                           D_0 = D_init_i,
                           sigma2_0 = sigma2_init_i,
                           max_iter = max_iter,
                           tolerance = tolerance,
                           is_mcem = is_mcem,
                           mc_times = mc_times)

    # Store results
    details[[i]] <- result
    final_iter_counts[i] <- result$final_iter_count
    final_gamma_hats[, i] <- result$final_gamma_hat
    final_D_hats[, , i] <- result$final_D_hat
    final_sigma2_hats[i] <- result$final_sigma2_hat
    }

    # Compute averages of final estimates
    final_gamma_hat_avg <- rowMeans(final_gamma_hats)
    final_D_hat_avg <- apply(final_D_hats, c(1, 2), mean)
    final_sigma2_hat_avg <- mean(final_sigma2_hats)

    # Compute per-iteration averages
    # Determine the maximum iteration count across simulations
    max_iter_count <- max(final_iter_counts)

    # Initialise matrices for storing per-iteration averages
    gamma_iter_avg <- matrix(0, nrow = gamma_dim, ncol = max_iter_count + 1)
    D_iter_avg <- array(0, dim = c(D_dim, D_dim, max_iter_count + 1))
    sigma2_iter_avg <- numeric(max_iter_count + 1)
    loglik_iter_avg <- numeric(max_iter_count + 1)
    record_counts <- integer(max_iter_count + 1)
    loglik_counts <- integer(max_iter_count + 1)

    # Average initial values and each iteration slot
    # Key: forward-fill each sim's converged value to max_iter_count; otherwise the tail average
    # is dominated by the few simulations that ran the most iterations.
    n_slots <- max_iter_count + 1
    for (i in 1:n_sim) {
      history <- details[[i]]$history
      H <- length(history)

      # Find the last non-null record
      last_gamma <- NULL; last_D <- NULL; last_sigma2 <- NULL; last_loglik <- NULL  # track last valid values
      for (k in 1:n_slots) {
        if (k <= H && !is.null(history[[k]]$gamma) && !is.null(history[[k]]$D) && !is.null(history[[k]]$sigma2)) {
          last_gamma  <- history[[k]]$gamma
          last_D      <- history[[k]]$D
          last_sigma2 <- history[[k]]$sigma2
        }
        if (!is.null(last_gamma)) {
          gamma_iter_avg[, k]  <- gamma_iter_avg[, k]  + last_gamma
          D_iter_avg[, , k]    <- D_iter_avg[, , k]    + last_D
          sigma2_iter_avg[k]   <- sigma2_iter_avg[k]   + last_sigma2
          record_counts[k]     <- record_counts[k] + 1
        }

        ll_k <- if (k <= H) history[[k]]$loglik else NULL
        if (!is.null(ll_k) && !is.na(ll_k) && is.finite(ll_k)) last_loglik <- ll_k
        if (!is.null(last_loglik)) {
          loglik_iter_avg[k] <- loglik_iter_avg[k] + last_loglik
          loglik_counts[k]   <- loglik_counts[k] + 1
        }
      }
    }

    # Compute averages
    valid_indices <- which(record_counts > 0)
    if (length(valid_indices) == 0) {
      stop("No valid iteration slots available for computing averages!")
    }

    # Compute per-iteration averages
    for (k in valid_indices) {
      gamma_iter_avg[, k] <- gamma_iter_avg[, k] / record_counts[k]
      D_iter_avg[, , k] <- D_iter_avg[, , k] / record_counts[k]
      sigma2_iter_avg[k] <- sigma2_iter_avg[k] / record_counts[k]
    }
    valid_ll_indices <- which(loglik_counts > 0)
    loglik_iter_avg[valid_ll_indices] <- loglik_iter_avg[valid_ll_indices] / loglik_counts[valid_ll_indices]
    loglik_iter_avg[loglik_counts == 0] <- NA

    # Retain only valid iteration slots
    gamma_iter_avg <- gamma_iter_avg[, valid_indices, drop = FALSE]
    D_iter_avg <- D_iter_avg[, , valid_indices, drop = FALSE]
    sigma2_iter_avg <- sigma2_iter_avg[valid_indices]
    loglik_iter_avg <- loglik_iter_avg[valid_indices]

    # Return results
    result <- list(
      n_sim = n_sim,                             # number of simulations
      final_iter_count = mean(final_iter_counts), # average number of iterations
      final_gamma_hat = final_gamma_hat_avg,     # average final gamma estimate
      final_D_hat = final_D_hat_avg,             # average final D estimate
      final_sigma2_hat = final_sigma2_hat_avg,   # average final sigma2 estimate
      gamma_real = gamma_real,                   # true gamma
      D_real = D_real,                          # true D
      sigma2_real = sigma2_real,                # true sigma2
      gamma_iter_avg_history = gamma_iter_avg,   # per-iteration average gamma estimate
      D_iter_avg_history = D_iter_avg,           # per-iteration average D estimate
      sigma2_iter_avg_history = sigma2_iter_avg, # per-iteration average sigma2 estimate
      loglik_iter_avg_history = loglik_iter_avg, # per-iteration average log-likelihood
      details = details                          # per-simulation detailed results
    )
    return(result)
}
