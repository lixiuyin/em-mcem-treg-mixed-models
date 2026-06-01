library(this.path)
source(file.path(this.dir(), "data_generation.R"), encoding = "UTF-8")

# Log posterior of observed data: under prior pi(beta, sigma^2) proportional to 1/sigma^2,
# this is the quantity that EM monotonically non-decreases.
#   ln p(beta, sigma^2 | y) proportional to sum_i ln f(y_i | beta, sigma^2) - ln sigma^2
# where f is the scaled t_nu(0, sigma^2) density: f(r) = (1/sigma) * g_nu(r/sigma),
# g_nu is the standard t_nu density, sigma = sqrt(sigma^2).
get_loglik <- function(X, y, beta, sigma2, t_df) {
  sigma2 <- as.numeric(sigma2)[1]
  sigma <- sqrt(sigma2)
  r <- as.numeric(y - X %*% beta)
  n <- length(r)
  ll_marginal <- sum(dt(r / sigma, df = t_df, log = TRUE)) - n * log(sigma)
  ll_marginal - log(sigma2)   # add the 1/sigma^2 prior term
}

run_single_em_or_mcem <- function(X, y, t_df, beta_0, sigma2_0,
                           max_iter = 1000, tolerance = 1e-7,
                           is_mcem = FALSE, mc_times = 100) {
  p <- ncol(X) - 1
  n <- nrow(X)

  # Initialize history matrices
  beta_history <- matrix(0, nrow = max_iter + 1, ncol = ncol(X))
  sigma2_history <- numeric(max_iter + 1)
  loglik_history <- numeric(max_iter + 1)
  colnames(beta_history) <- c("Intercept", paste0("X", 1:p))
  cat("Initial regression coefficients: (", beta_0, "), initial sigma2:", sigma2_0, "\n")
  beta_history[1, ] <- beta_0
  sigma2_history[1] <- sigma2_0
  loglik_history[1] <- get_loglik(X, y, beta_0, sigma2_0, t_df)

  # Starting values
  beta_k <- beta_0
  sigma2_k <- sigma2_0

  cat("Running", ifelse(is_mcem, "MCEM", "EM"), "algorithm",
      ifelse(is_mcem, paste(", MC sample size:", mc_times), ""), "\n")

  iter_count <- 0  # actual iteration counter
  for (idx in 2:(max_iter + 1)){
    res_k <- y - X %*% beta_k

    if (is_mcem) {
      mc_k <- numeric(n)
      gamma_alpha <- (t_df + 1) / 2 # zi | yi follows Gamma(alpha, beta)
      for (i in 1:n){
        gamma_lambda <- ((res_k[i])^2 / (2 * sigma2_k)) + (t_df / 2)
        samples <- rgamma(mc_times,
                          shape = gamma_alpha, scale = 1 / gamma_lambda)
        mc_k[i] <- mean(samples)
      }
      w_mc_k <- diag(mc_k)
    } else {
      numerator <- sigma2_k * (t_df + 1)
      denominator <- as.numeric(res_k)^2 + sigma2_k * t_df
      gamma_k <- numerator / denominator
      w_mc_k <- diag(gamma_k)
    }

    beta_prev <- beta_k
    sigma2_prev <- sigma2_k

    beta_k <- solve(t(X) %*% w_mc_k %*% X) %*% t(X) %*% w_mc_k %*% y

    res_kp1 <- y - X %*% beta_k
    sigma2_k <- as.numeric(t(res_kp1) %*% w_mc_k %*% res_kp1) / (n + 2)

    # Record current values
    beta_history[idx, ] <- beta_k
    sigma2_history[idx] <- sigma2_k
    loglik_history[idx] <- get_loglik(X, y, beta_k, sigma2_k, t_df)

    iter_count <- iter_count + 1  # increment iteration counter
    cat("Iteration", iter_count, ": beta =", round(beta_k, 4),
        ", sigma^2 =", round(sigma2_k, 4), "\n")

    # Convergence check
    if (max(norm(beta_k - beta_prev, "2"),
            abs(sigma2_k - sigma2_prev)) < tolerance) {
      cat("Converged after", iter_count, "iterations\n")
      break
    } else if (iter_count == max_iter) {
      cat("Reached maximum iterations without convergence\n")
    }
  }

  cat("Final estimates: beta =", round(beta_k, 4),
      ", sigma^2 =", round(sigma2_k, 4),
      "  Iterations:", iter_count, "\n")

  return(list(
    final_iter_count = iter_count,  # total number of iterations
    final_beta_hat = beta_k,  # estimated regression coefficients
    final_sigma2_hat = sigma2_k,  # estimated error variance
    beta_history = beta_history[1:(iter_count + 1), ],  # beta iteration history
    sigma2_history = sigma2_history[1:(iter_count + 1)], # sigma^2 iteration history
    loglik_history = loglik_history[1:(iter_count + 1)]  # log posterior history
  ))
}

run_multiple_em_or_mcem <- function(n_sim = 500,
                                    p = 2,
                                    max_iter = 1000,
                                    tolerance = 1e-7,
                                    is_mcem = FALSE,
                                    mc_times = 100,
                                    reset_dataset = FALSE,
                                    bold_random = FALSE,
                                    beta_0 = NULL,
                                    sigma2_0 = NULL,
                                    fixed_data = NULL,
                                    specific_df = NULL,
                                    n = 500,
                                    t_df = 10,
                                    t_mu = 0,
                                    t_sigma2_real = 0.4,
                                    beta_real = c(2, 3, 5)) {
  # Parameter validation
  if (n_sim <= 0 || max_iter <= 0 || tolerance <= 0 || mc_times <= 0) {
    stop("Iteration parameters must be positive")
  }

  # Initialize result storage
  details <- vector("list", n_sim)
  final_iter_counts <- integer(n_sim)
  final_beta_hats <- matrix(0, nrow = n_sim, ncol = p + 1)
  final_sigma2_hats <- numeric(n_sim)

  # Use fixed dataset if provided
  if (!is.null(fixed_data)) {
    cat("Using fixed dataset\n")
    data <- fixed_data
    cat(sprintf("\nSimulating t-distribution data with degrees of freedom %d\n", data$t_df))
    if (reset_dataset){
      stop("Cannot reset a fixed dataset")
    }
  } else {
    # Generate initial dataset
    data <- generate_t_simulated_data(n, p, t_df, t_mu, t_sigma2_real, beta_real)
    cat(sprintf("\nInitial dataset generated with degrees of freedom %d\n", t_df))
  }

  # Run multiple simulations
  for (i in 1:n_sim) {
    cat(sprintf("\nRunning simulation %d:\n", i))
    # Reset dataset if requested
    if (reset_dataset) {
      data <- generate_t_simulated_data(n, p, t_df, t_mu,
                                        t_sigma2_real, beta_real)
      cat("Simulated data regenerated\n")
    }
    # Generate initial parameters from current data
    if(!is.null(beta_0) && !is.null(sigma2_0)){
      initial_params <- list(beta_init = beta_0, sigma2_init = sigma2_0)
      cat("Fixed initial values provided; using fixed initialization\n")
    } else {
      if (bold_random){
        initial_params <- bold_random_init(p)
        cat("No fixed initial values; using bold random initialization\n")
      } else {
        initial_params <- random_init(p)
        cat("No fixed initial values; using conservative random initialization\n")
      }
    }

    # Run EM/MCEM algorithm
    if (is.null(specific_df)){
      current_df <- data$t_df
    }
    else {
      current_df <- specific_df
    }

    result <- run_single_em_or_mcem(X = data$X,
                                    y = data$y_real,
                                    t_df = current_df,
                                    beta_0 = initial_params$beta_init,
                                    sigma2_0 = initial_params$sigma2_init,
                                    max_iter = max_iter,
                                    tolerance = tolerance,
                                    is_mcem = is_mcem,
                                    mc_times = mc_times)

    # Store results
    details[[i]] <- result
    final_iter_counts[i] <- result$final_iter_count
    final_beta_hats[i, ] <- result$final_beta_hat
    final_sigma2_hats[i] <- result$final_sigma2_hat
  }

  # Compute mean of final estimates
  final_beta_hat_avg <- colMeans(final_beta_hats)
  final_sigma2_hat_avg <- mean(final_sigma2_hats)

  # Compute per-iteration averages
  # Find maximum iteration count
  max_iter_count <- max(final_iter_counts)

  # Initialize matrices for per-iteration averages
  beta_iter_avg <- matrix(0, nrow = max_iter_count + 1, ncol = p + 1)
  sigma2_iter_avg <- matrix(0, nrow = max_iter_count + 1, ncol = 1)
  record_counts <- numeric(max_iter_count + 1)

  # Accumulate initial values and per-iteration values.
  # Key: after each simulation converges, forward-fill its last value to max_iter_count,
  # otherwise late-stage averages would be dominated by the few simulations still
  # iterating, causing spurious MSE rebounds at the tail.
  n_slots <- max_iter_count + 1
  for(i in 1:n_sim) {
    record_length <- final_iter_counts[i] + 1  # +1 to include initial value
    bh <- details[[i]]$beta_history
    sh <- details[[i]]$sigma2_history
    # Accumulate within recorded range
    beta_iter_avg[1:record_length, ]  <- beta_iter_avg[1:record_length, ]  + bh[1:record_length, ]
    sigma2_iter_avg[1:record_length]  <- sigma2_iter_avg[1:record_length]  + sh[1:record_length]
    record_counts[1:record_length]    <- record_counts[1:record_length]    + 1
    # Forward-fill: propagate last converged value to n_slots
    if (record_length < n_slots) {
      last_beta   <- bh[record_length, ]
      last_sigma2 <- sh[record_length]
      tail_idx <- (record_length + 1):n_slots
      beta_iter_avg[tail_idx, ]  <- beta_iter_avg[tail_idx, ]  + matrix(last_beta, nrow = length(tail_idx), ncol = ncol(bh), byrow = TRUE)
      sigma2_iter_avg[tail_idx]  <- sigma2_iter_avg[tail_idx]  + last_sigma2
      record_counts[tail_idx]    <- record_counts[tail_idx]    + 1
    }
  }

  # Compute averages
  valid_indices <- which(record_counts > 0)
  if (length(valid_indices) == 0) {
    stop("No valid iteration steps available for computing averages!")
  }

  # Average over valid iteration indices
  beta_iter_avg[valid_indices, ] <- beta_iter_avg[valid_indices, ] / record_counts[valid_indices]
  sigma2_iter_avg[valid_indices] <- sigma2_iter_avg[valid_indices] / record_counts[valid_indices]

  # Keep only valid iteration steps
  beta_iter_avg <- beta_iter_avg[valid_indices, , drop = FALSE]  # drop = FALSE preserves matrix structure
  sigma2_iter_avg <- sigma2_iter_avg[valid_indices]

  # Return results
  result <- list(
    n_sim = n_sim,                             # number of simulations
    final_iter_count = mean(final_iter_counts), # mean iteration count
    final_beta_hat = final_beta_hat_avg,       # mean final beta estimate
    final_sigma2_hat = final_sigma2_hat_avg,   # mean final sigma^2 estimate
    iter_count_history = final_iter_counts,     # per-simulation iteration counts
    beta_iter_avg_history = beta_iter_avg,     # per-iteration mean beta estimates
    sigma2_iter_avg_history = sigma2_iter_avg, # per-iteration mean sigma^2 estimates
    details = details                          # detailed results for each simulation
  )
  return(result)
}

plot_single_iteration <- function(result, beta_real, t_sigma2_real, title){
  iter_count <- result$final_iter_count
  beta_history <- result$beta_history
  sigma2_history <- result$sigma2_history

  # Get data dimensions
  p <- ncol(beta_history) - 1

  valid_indices <- 0:iter_count  # iteration indices starting from 0

  # Set up plot layout
  par(mfrow = c(2, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0), family = "sans")

  # 1. Beta parameter iteration path
  beta_cols <- get_nice_colors(p + 1)
  plot(valid_indices, beta_history[, 1], type = "l", col = beta_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(beta), main = expression(beta*" iteration path"),
       ylim = range(c(beta_history, beta_real), na.rm = TRUE),
       xaxt = "n", bty = "l")
  axis(1, at = seq(min(valid_indices), max(valid_indices), by = 1))
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

  # Draw estimated curves and true-value dashed lines
  for(i in 1:(p + 1)) {
    lines(valid_indices, beta_history[, i], col = beta_cols[i], lwd = 2)
    abline(h = beta_real[i], col = beta_cols[i], lty = 2, lwd = 1.5)
  }
  legend("bottomright",
         legend = c(
           expression(hat(beta)[0]),        # intercept estimate
           sapply(1:p, function(i) bquote(hat(beta)[.(i)])),  # beta1~beta_p estimates
           expression(beta[0]),             # intercept true value
           sapply(1:p, function(i) bquote(beta[.(i)]))        # beta1~beta_p true values
         ),
         col = rep(beta_cols, 2),
         lty = c(rep(1, p + 1), rep(2, p + 1)),
         lwd = c(rep(2, p + 1), rep(1.5, p + 1)),
         bty = "n", cex = 0.9,
         ncol = 2)

  # 2. Sigma^2 parameter iteration path
  y_min <- floor(min(c(sigma2_history, t_sigma2_real), na.rm = TRUE) * 5) / 5
  y_max <- ceiling(max(c(sigma2_history, t_sigma2_real), na.rm = TRUE) * 5) / 5
  if(y_max - y_min < 0.2) {
    y_min <- y_min - 0.1
    y_max <- y_max + 0.1
  }
  y_breaks <- seq(y_min, y_max, by = 0.2)
  plot(valid_indices, sigma2_history, type = "l", col = beta_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(sigma^2), main = expression(sigma^2*" iteration path"),
       ylim = c(y_min, y_max), yaxt = "n", xaxt = "n", bty = "l")
  axis(1, at = seq(min(valid_indices), max(valid_indices), by = 1))
  axis(2, at = y_breaks, labels = sprintf("%.1f", y_breaks))
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")
  abline(h = t_sigma2_real, col = beta_cols[1], lty = 2, lwd = 1.5)
  legend("topright",
         legend = c(expression(hat(sigma)^2), expression(sigma^2)),
         col = c(beta_cols[1], beta_cols[1]),
         lty = c(1, 2),
         lwd = c(2, 1.5),
         bty = "n", cex = 1)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2)

  # Reset plot parameters
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot mean iteration path across multiple simulations
#' @param result result list from multiple simulations
#' @param title plot title
plot_multiple_simulation_iterations <- function(result, beta_real, t_sigma2_real, title = "Mean iteration path across simulations") {
  # Validate input
  if (is.null(result$beta_iter_avg_history) || is.null(result$sigma2_iter_avg_history)) {
    stop("Input result is missing required iteration history data")
  }

  # Extract iteration histories
  beta_history <- result$beta_iter_avg_history
  sigma2_history <- result$sigma2_iter_avg_history

  # Get data dimensions
  p <- ncol(beta_history) - 1

  valid_indices <- 0:(nrow(beta_history) - 1)  # iteration indices starting from 0

  # Set up plot layout
  par(mfrow = c(2, 1), mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))

  # 1. Beta parameter iteration path
  beta_cols <- get_nice_colors(p + 1)
  plot(valid_indices, beta_history[, 1], type = "l", col = beta_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(beta), main = expression(beta*" iteration path"),
       ylim = range(c(beta_history, beta_real), na.rm = TRUE),
       xaxt = "n", bty = "l")
  axis(1, at = seq(min(valid_indices), max(valid_indices), by = 1))
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

  # Draw estimated curves and true-value dashed lines
  for(i in 1:(p + 1)) {
    lines(valid_indices, beta_history[, i], col = beta_cols[i], lwd = 2)
    abline(h = beta_real[i], col = beta_cols[i], lty = 2, lwd = 1.5)
  }
  legend("topright",
         legend = c(
           expression(hat(beta)[0]),        # intercept estimate
           sapply(1:p, function(i) bquote(hat(beta)[.(i)])),  # beta1~beta_p estimates
           expression(beta[0]),             # intercept true value
           sapply(1:p, function(i) bquote(beta[.(i)]))        # beta1~beta_p true values
         ),
         col = rep(beta_cols, 2),
         lty = c(rep(1, p + 1), rep(2, p + 1)),
         lwd = c(rep(2, p + 1), rep(1.5, p + 1)),
         bty = "n", cex = 1,
         ncol = 2)

  # 2. Sigma^2 parameter iteration path
  y_min <- floor(min(c(sigma2_history, t_sigma2_real), na.rm = TRUE) * 5) / 5
  y_max <- ceiling(max(c(sigma2_history, t_sigma2_real), na.rm = TRUE) * 5) / 5
  if(y_max - y_min < 0.2) {
    y_min <- y_min - 0.1
    y_max <- y_max + 0.1
  }
  y_breaks <- seq(y_min, y_max, by = 0.2)
  plot(valid_indices, sigma2_history, type = "l", col = "#1f77b4", lwd = 2,
       xlab = "Iteration", ylab = expression(sigma^2), main = expression(sigma^2*" iteration path"),
       ylim = c(y_min, y_max), yaxt = "n", xaxt = "n", bty = "l")
  axis(1, at = seq(min(valid_indices), max(valid_indices), by = 1))
  axis(2, at = y_breaks, labels = sprintf("%.1f", y_breaks))
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")
  abline(h = t_sigma2_real, col = "#1f77b4", lty = 2, lwd = 1.5)
  legend("topright",
         legend = c(expression(hat(sigma)^2), expression(sigma^2)),
         col = c("#1f77b4", "#1f77b4"),
         lty = c(1, 2),
         lwd = c(2, 1.5),
         bty = "n", cex = 1)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2)

  # Reset plot parameters
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot MSE over iterations for a single simulation run
#' @param result result from a single EM/MCEM run
#' @param beta_real true beta values
#' @param t_sigma2_real true sigma^2 value
#' @param title plot title
plot_single_mse <- function(result, beta_real, t_sigma2_real, title) {
  # Validate input
  if (is.null(result$beta_history) || is.null(result$sigma2_history)) {
    stop("Input result is missing required iteration history data")
  }

  # Set plot parameters
  par(mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))

  # Compute MSE
  valid_indices <- 0:(nrow(result$beta_history) - 1)
  mse_result <- get_mse(result$beta_history,
                        result$sigma2_history,
                        beta_real, t_sigma2_real)
  each_beta_mse <- mse_result$each_beta_mse
  sigma2_mse <- mse_result$sigma2_mse
  total_mse <- mse_result$total_mse

  # Get colors
  colors <- get_nice_colors(ncol(result$beta_history) + 2)

  # Plot MSE curves
  # 1. Per-beta MSE
  plot(valid_indices, each_beta_mse[,1], type = "l",
       col = colors[1], lwd = 2,
       xlab = "Iteration", ylab = "MSE",
       main = "MSE by parameter",
       ylim = c(0, max(each_beta_mse, sigma2_mse, total_mse)),
       xaxt = "n", bty = "l")
  axis(1, at = seq(min(valid_indices), max(valid_indices), by = 1))
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

  # Add remaining beta MSE curves
  for(i in 2:ncol(each_beta_mse)) {
    lines(valid_indices, each_beta_mse[,i],
          col = colors[i], lwd = 2)
  }

  # Add sigma^2 MSE
  lines(valid_indices, sigma2_mse, col = colors[ncol(each_beta_mse) + 1], lwd = 2)

  # Add total MSE
  lines(valid_indices, total_mse, col = colors[ncol(each_beta_mse) + 2], lwd = 2)

  # Add legend
  legend("topright",
         legend = c(
           paste0("MSE(beta", 0:(ncol(each_beta_mse)-1), ")"),
           expression(MSE[sigma^2]),
           expression(MSE[total])
         ),
         col = colors,
         lty = c(rep(1, ncol(each_beta_mse)), 1, 1),
         lwd = c(rep(2, ncol(each_beta_mse)), 2, 2),
         bty = "n", cex = 1,
         ncol = 2)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2)

  # Reset plot parameters
  par(mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot MSE over iterations averaged across multiple simulations
#' @param result result list from multiple simulations
#' @param title plot title
plot_multiple_simulation_mse <- function(result, beta_real, t_sigma2_real, title) {
  # Set plot parameters
  par(mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))

  # Compute MSE
  valid_indices <- 0:(nrow(result$beta_iter_avg_history) - 1)
  mse_result <- get_mse(result$beta_iter_avg_history,
                        result$sigma2_iter_avg_history,
                        beta_real, t_sigma2_real)
  each_beta_mse <- mse_result$each_beta_mse
  sigma2_mse <- mse_result$sigma2_mse
  total_mse <- mse_result$total_mse
  # Get colors
  colors <- get_nice_colors(ncol(result$beta_iter_avg_history) + 2)
  # Plot MSE curves
  # 1. Per-beta MSE
  plot(valid_indices, each_beta_mse[,1], type = "l",
       col = colors[1], lwd = 2,
       xlab = "Iteration", ylab = "MSE",
       main = "MSE by parameter",
       ylim = c(0, max(each_beta_mse, sigma2_mse, total_mse)),
       xaxt = "n", bty = "l")
  axis(1, at = seq(min(valid_indices), max(valid_indices), by = 1))
  grid(nx = NA, ny = NULL, col = "gray90", lty = "dotted")

  # Add remaining beta MSE curves
  for(i in 2:ncol(each_beta_mse)) {
    lines(valid_indices, each_beta_mse[,i],
          col = colors[i], lwd = 2)
  }
  # Add sigma^2 MSE
  lines(valid_indices, sigma2_mse, col = colors[ncol(each_beta_mse) + 1], lwd = 2)
  # Add total MSE
  lines(valid_indices, total_mse, col = colors[ncol(each_beta_mse) + 2], lwd = 2)
  # Add legend
  legend("topright",
         legend = c(
           paste0("MSE(beta", 0:(ncol(each_beta_mse)-1), ")"),
           expression(MSE[sigma^2]),
           expression(MSE[total])
         ),
         col = colors,
         lty = c(rep(1, ncol(each_beta_mse)), 1, 1),
         lwd = c(rep(2, ncol(each_beta_mse)), 2, 2),
         bty = "n", cex = 1,
         ncol = 2)
  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2)
  # Reset plot parameters
  par(mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

get_nice_colors <- function(n) {
if (requireNamespace("RColorBrewer", quietly = TRUE)) {
if (n <= 8) {
return(RColorBrewer::brewer.pal(n, "Set1"))
} else if (n <= 12) {
return(RColorBrewer::brewer.pal(n, "Paired"))
} else {
pal <- colorRampPalette(RColorBrewer::brewer.pal(12, "Paired"))
return(pal(n))
}
} else if (requireNamespace("viridis", quietly = TRUE)) {
return(viridis::viridis(n))
} else {
return(gray.colors(n, start = 0.2, end = 1))
}
}

get_mse <- function(beta_predicted, t_sigma2_predicted, beta_real, t_sigma2_real) {
  # Determine whether input is a final estimate (vector or single-column matrix)
  # or an iteration history (multi-row matrix)
  if (is.vector(beta_predicted) || (is.matrix(beta_predicted) && ncol(beta_predicted) == 1)) {
    # Convert single-column matrix to vector
    if (is.matrix(beta_predicted)) {
      beta_predicted <- as.vector(beta_predicted)
    }

    # Final estimate case
    # Per-beta MSE
    each_beta_mse <- (beta_predicted - beta_real)^2
    # Total beta MSE
    beta_mse <- sum(each_beta_mse)
    # Sigma^2 MSE
    sigma2_mse <- (t_sigma2_predicted - t_sigma2_real)^2
    # Overall MSE
    total_mse <- beta_mse + sigma2_mse

    # Return scalar MSE values
    return(list(
      each_beta_mse = each_beta_mse,    # per-beta MSE
      beta_mse = beta_mse,              # total beta MSE
      sigma2_mse = sigma2_mse,          # sigma^2 MSE
      total_mse = total_mse             # overall MSE
    ))
  } else {
    # Iteration history case (matrix)
    # Dimension check
    if (ncol(beta_predicted) != length(beta_real)) {
      stop("Number of columns in beta_predicted must equal the length of beta_real")
    }

    # Broadcast true values into a matrix
    matrix_beta_real <- matrix(rep(beta_real, each = nrow(beta_predicted)),
                              nrow = nrow(beta_predicted))

    # Per-beta MSE (row-wise)
    each_beta_mse <- (beta_predicted - matrix_beta_real)^2
    # Total beta MSE per row
    beta_mse <- rowSums(each_beta_mse)
    # Sigma^2 MSE (row-wise)
    sigma2_mse <- (t_sigma2_predicted - t_sigma2_real)^2
    # Overall MSE per row
    total_mse <- beta_mse + sigma2_mse

    # Return history MSE
    return(list(
      each_beta_mse = each_beta_mse,    # per-beta row-wise MSE
      beta_mse = beta_mse,              # total beta MSE per row
      sigma2_mse = sigma2_mse,          # sigma^2 row-wise MSE
      total_mse = total_mse             # overall row-wise MSE
    ))
  }
}

study_df_effect <- function(n = 500, p = 2, t_mu = 0, t_sigma2_real = 0.4,
                            beta_real = c(2, 3, 5), df_range = 5:105,
                            bold_random = FALSE, specific_data = NULL) {
  # Initialize result storage
  results <- data.frame(
    df = integer(0),
    beta_error = numeric(0),
    sigma2_error = numeric(0),
    final_iter = numeric(0)
  )

  # Optionally use fixed data
  for (df in df_range) {
    if (!is.null(specific_data)){
    data <- specific_data
    }
    else {
      data <- generate_t_simulated_data(n, p, df, t_mu, t_sigma2_real, beta_real)

    }
    result <- run_multiple_em_or_mcem(n_sim = 500,
                                      p = p,
                                      max_iter = 1000,
                                      tolerance = 1e-7,
                                      is_mcem = FALSE,
                                      mc_times = 100,
                                      bold_random = bold_random,
                                      fixed_data = data,
                                      specific_df = df,
                                      n = n,
                                      t_mu = t_mu,
                                      t_sigma2_real = t_sigma2_real,
                                      beta_real = beta_real)

    # Compute MSE
    mse_result <- get_mse(result$final_beta_hat, result$final_sigma2_hat, beta_real, t_sigma2_real)

    # Store results
    results <- rbind(results, data.frame(
      df = df,
      beta_error = mse_result$beta_mse,
      sigma2_error = mse_result$sigma2_mse,
      final_iter = result$final_iter_count
    ))
  }

  # Plot parameter estimation error vs degrees of freedom
  p1 <- ggplot(results, aes(x = df)) +
    geom_line(aes(y = beta_error), color = "blue") +
    geom_line(aes(y = sigma2_error), color = "red") +
    labs(title = "Parameter estimation error vs degrees of freedom",
         x = "Degrees of freedom (df)",
         y = "Estimation error",
         caption = "Blue: beta error; Red: sigma^2 error") +
    theme_minimal() +
    theme(text = element_text(family = "serif"))

  # Plot iteration count vs degrees of freedom
  p2 <- ggplot(results, aes(x = df, y = final_iter)) +
    geom_line(color = "darkgreen") +
    labs(title = "EM algorithm iterations vs degrees of freedom",
         x = "Degrees of freedom (df)",
         y = "Final iteration count") +
    theme_bw() +
    theme(text = element_text(family = "serif"))

  # Return results
  return(list(
    results = results,
    plots = list(
      error_plot = p1,
      iter_plot = p2
    )
  ))
}
