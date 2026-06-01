# ============================================================================
# Plotting Functions
#   Visualisation utilities for single-run and multi-simulation EM/MCEM results.
# ============================================================================

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

#' Plot detailed parameter iteration path for a single run
#' @param result result list from run_single_em_or_mcem
#' @param title plot title
plot_single_iteration <- function(result, title = "Parameter iteration path") {
  history <- result$history
  gamma_real <- result$data$gamma_real
  D_real <- result$data$D_real
  sigma2_real <- result$data$sigma2_real
  p <- result$data$p
  q <- result$data$q
  gamma_dim <- (p + 1) * (q + 1)
  D_dim <- p + 1

  # Set plot layout
  par(mfrow = c(2, 2), mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))

  # 1. gamma
  gamma_history <- sapply(history, function(x) as.vector(x$gamma))
  valid_gamma <- apply(gamma_history, 2, function(col) all(!is.na(col)))
  gamma_history <- gamma_history[, valid_gamma, drop=FALSE]
  n_iter <- ncol(gamma_history)
  gamma_cols <- get_nice_colors(gamma_dim)
  plot(0:(n_iter-1), gamma_history[1,], type = "l", col = gamma_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(gamma), main = expression(gamma*" iteration path"),
       ylim = range(c(gamma_history, gamma_real), na.rm=TRUE))
  abline(h = gamma_real[1], col = gamma_cols[1], lty = 2, lwd = 1.5)
  for(i in 2:gamma_dim) {
    lines(0:(n_iter-1), gamma_history[i,], col = gamma_cols[i], lwd = 2)
    abline(h = gamma_real[i], col = gamma_cols[i], lty = 2, lwd = 1.5)
  }
  gamma_labels <- paste0("gamma[", 1:gamma_dim, "]")
  legend("topright", legend = gamma_labels, col = gamma_cols, lty = 1, lwd = 2,
         bty = "n", cex = min(0.7, 2.5/gamma_dim), ncol = ceiling(gamma_dim/10))

  # 2. D
  D_history <- sapply(history, function(x) as.vector(x$D))
  D_real_vec <- as.vector(D_real)
  D_cols <- get_nice_colors(D_dim^2)
  for(i in 1:(D_dim^2)) {
    if(i == 1) {
      plot(0:(ncol(D_history)-1), D_history[i,], type = "l", col = D_cols[i], lwd = 2,
           xlab = "Iteration", ylab = expression(D), main = expression(D*" matrix iteration path"),
           ylim = range(c(D_history, D_real_vec)))
    } else {
      lines(0:(ncol(D_history)-1), D_history[i,], col = D_cols[i], lwd = 2)
    }
    abline(h = D_real_vec[i], col = D_cols[i], lty = 2, lwd = 1.5)
  }
  D_labels <- paste0("D[", rep(1:D_dim, each=D_dim), ",", rep(1:D_dim, times=D_dim), "]")
  legend("topright", legend = D_labels, col = D_cols, lty = 1, lwd = 2,
         bty = "n", cex = min(0.7, 2.5/(D_dim^2)), ncol = ceiling((D_dim^2)/10))

  # 3. sigma2
  # Retrieve sigma2 history
  sigma2_history <- sapply(history, function(x) x$sigma2)

  # Compute y-axis range including 0 and the true value
  y_min <- 0
  y_max <- max(c(sigma2_history, sigma2_real), na.rm = TRUE)
  y_max <- ceiling(y_max * 1.1)  # add 10% headroom

  # Determine suitable tick spacing (no more than 5 ticks)
  y_breaks <- pretty(c(y_min, y_max), n = 5)

  # Colours
  sigma2_cols <- get_nice_colors(3)[1:2]  # take 3 colours but use only the first 2

  plot(0:(length(sigma2_history)-1), sigma2_history, type = "l", col = sigma2_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(sigma^2), main = expression(sigma^2*" iteration path"),
       ylim = c(y_min, y_max), yaxt = "n")
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

  # Add y-axis ticks using %g format
  axis(2, at = y_breaks, labels = sprintf("%g", y_breaks))

  # Add dashed line for the true value
  abline(h = sigma2_real, col = sigma2_cols[2], lty = 2, lwd = 1.5)

  # Add legend
  legend("topright",
         legend = c("Estimate", "True value"),
         col = sigma2_cols,
         lty = c(1, 2),
         lwd = c(2, 1.5),
         bty = "n",
         cex = 0.8)

  # 4. MSE and log-likelihood
  # Compute per-iteration MSE
  mse_history <- list()
  for(i in 1:length(history)) {
    if(!is.null(history[[i]]$gamma) && !is.null(history[[i]]$D) && !is.null(history[[i]]$sigma2)) {
      mse_history[[i]] <- get_mse(history[[i]]$gamma,
                                 history[[i]]$D,
                                 history[[i]]$sigma2,
                                 gamma_real,
                                 D_real,
                                 sigma2_real)
    }
  }

  # Extract MSE values
  mse_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse else NA)
  mse_gamma_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse_gamma else NA)
  mse_D_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse_D else NA)
  mse_sigma2_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse_sigma2 else NA)

  # Colours for MSE curves
  mse_cols <- get_nice_colors(5)  # 5 colours for 5 curves

  # Create dual-axis plot
  par(mar = c(4, 4, 2, 4))
  plot(0:(length(mse_values)-1), mse_values, type = "l", col = mse_cols[1], lwd = 2,
       xlab = "Iteration", ylab = "MSE", main = "MSE and log-likelihood",
       ylim = c(0, max(c(mse_values, mse_gamma_values, mse_D_values, mse_sigma2_values), na.rm = TRUE)))
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
  lines(0:(length(mse_gamma_values)-1), mse_gamma_values, col = mse_cols[2], lwd = 2, lty = 2)
  lines(0:(length(mse_D_values)-1), mse_D_values, col = mse_cols[3], lwd = 2, lty = 3)
  lines(0:(length(mse_sigma2_values)-1), mse_sigma2_values, col = mse_cols[4], lwd = 2, lty = 4)

  # Add log-likelihood curve on secondary axis
  par(new = TRUE)
  loglik_history <- sapply(history, function(x) x$loglik)
  loglik_min <- floor(min(loglik_history, na.rm = TRUE))
  loglik_max <- ceiling(max(loglik_history, na.rm = TRUE))
  loglik_breaks <- seq(loglik_min, loglik_max, length.out = 5)
  plot(0:(length(loglik_history)-1), loglik_history, type = "l", col = mse_cols[5], lwd = 2, lty = 5,
       xlab = "", ylab = "", axes = FALSE, ylim = c(loglik_min, loglik_max))
  axis(4, at = loglik_breaks, labels = format(loglik_breaks, digits = 2), col = mse_cols[5], col.axis = mse_cols[5])
  mtext("Log-likelihood", side = 4, line = 3, col = mse_cols[5])

  # Add legend
  legend("topright",
         legend = c("Overall MSE", expression(MSE[gamma]), expression(MSE[D]), expression(MSE[sigma^2]), "Log-likelihood"),
         col = mse_cols,
         lty = c(1, 2, 3, 4, 5),
         lwd = 2,
         bty = "n",
         cex = 0.7)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2, font = 2)

  # Reset graphics parameters
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot average iteration path across multiple simulations
#' @param result result list from run_multiple_em_or_mcem
#' @param title plot title
plot_multiple_simulation_iterations <- function(result, title = "Average iteration path across simulations") {
  # Get data dimensions
  p <- result$details[[1]]$data$p
  q <- result$details[[1]]$data$q
  gamma_dim <- (p + 1) * (q + 1)
  D_dim <- p + 1

  # True parameter values
  gamma_real <- result$gamma_real
  D_real <- result$D_real
  sigma2_real <- result$sigma2_real

  # Set plot layout
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

  # 1. gamma iteration path
  gamma_cols <- get_nice_colors(gamma_dim)
  plot(0:(ncol(result$gamma_iter_avg_history)-1), result$gamma_iter_avg_history[1,],
       type = "l", col = gamma_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(gamma),
       main = expression(gamma*" iteration path"),
       ylim = range(c(result$gamma_iter_avg_history, gamma_real), na.rm = TRUE))
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
  abline(h = gamma_real[1], col = gamma_cols[1], lty = 2, lwd = 1.5)
  for(i in 2:gamma_dim) {
    lines(0:(ncol(result$gamma_iter_avg_history)-1), result$gamma_iter_avg_history[i,],
          col = gamma_cols[i], lwd = 2)
    abline(h = gamma_real[i], col = gamma_cols[i], lty = 2, lwd = 1.5)
  }
  gamma_labels <- paste0("gamma[", 1:gamma_dim, "]")
  legend("topright", legend = gamma_labels, col = gamma_cols, lty = 1, lwd = 2,
         bty = "n", cex = min(0.7, 2.5/gamma_dim), ncol = ceiling(gamma_dim/10))

  # 2. D matrix iteration path
  D_cols <- get_nice_colors(D_dim^2)
  D_real_vec <- as.vector(D_real)
  n_iter <- dim(result$D_iter_avg_history)[3]

  # Build matrix of D element histories
  D_elements <- matrix(0, nrow = D_dim^2, ncol = n_iter)
  for(i in 1:D_dim) {
    for(j in 1:D_dim) {
      idx <- (i-1)*D_dim + j
      D_elements[idx,] <- result$D_iter_avg_history[i,j,]
    }
  }

  # Plot D matrix elements
  for(i in 1:(D_dim^2)) {
    if(i == 1) {
      plot(0:(n_iter-1), D_elements[i,], type = "l", col = D_cols[i], lwd = 2,
           xlab = "Iteration", ylab = expression(D),
           main = expression(D*" matrix iteration path"),
           ylim = range(c(D_elements, D_real_vec), na.rm = TRUE))
      grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
    } else {
      lines(0:(n_iter-1), D_elements[i,], col = D_cols[i], lwd = 2)
    }
    abline(h = D_real_vec[i], col = D_cols[i], lty = 2, lwd = 1.5)
  }
  D_labels <- paste0("D[", rep(1:D_dim, each=D_dim), ",", rep(1:D_dim, times=D_dim), "]")
  legend("topright", legend = D_labels, col = D_cols, lty = 1, lwd = 2,
         bty = "n", cex = min(0.7, 2.5/(D_dim^2)), ncol = ceiling((D_dim^2)/10))

  # 3. sigma2 iteration path
  # Compute y-axis range including 0 and the true value
  y_min <- 0
  y_max <- max(c(result$sigma2_iter_avg_history, sigma2_real), na.rm = TRUE)
  y_max <- ceiling(y_max * 1.1)  # add 10% headroom

  # Determine suitable tick spacing (no more than 5 ticks)
  y_breaks <- pretty(c(y_min, y_max), n = 5)

  # Colours
  sigma2_cols <- get_nice_colors(3)[1:2]  # take 3 colours but use only the first 2

  plot(0:(length(result$sigma2_iter_avg_history)-1), result$sigma2_iter_avg_history,
       type = "l", col = sigma2_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(sigma^2),
       main = expression(sigma^2*" iteration path"),
       ylim = c(y_min, y_max), yaxt = "n")
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

  # Add y-axis ticks using %g format
  axis(2, at = y_breaks, labels = sprintf("%g", y_breaks))

  # Add dashed line for the true value
  abline(h = sigma2_real, col = sigma2_cols[2], lty = 2, lwd = 1.5)

  # Add legend
  legend("topright",
         legend = c("Estimate", "True value"),
         col = sigma2_cols,
         lty = c(1, 2),
         lwd = c(2, 1.5),
         bty = "n",
         cex = 0.8)

  # 4. MSE and log-likelihood
  # Compute per-iteration MSE
  mse_history <- list()
  for(k in 1:length(result$sigma2_iter_avg_history)) {
    mse_history[[k]] <- get_mse(result$gamma_iter_avg_history[,k],
                               result$D_iter_avg_history[,,k],
                               result$sigma2_iter_avg_history[k],
                               result$gamma_real,
                               result$D_real,
                               result$sigma2_real)
  }

  # Extract MSE values
  mse_values <- sapply(mse_history, function(x) x$mse)
  mse_gamma_values <- sapply(mse_history, function(x) x$mse_gamma)
  mse_D_values <- sapply(mse_history, function(x) x$mse_D)
  mse_sigma2_values <- sapply(mse_history, function(x) x$mse_sigma2)

  # Colours for MSE curves
  mse_cols <- get_nice_colors(5)  # 5 colours for 5 curves

  # Create dual-axis plot
  par(mar = c(4, 4, 2, 4))
  plot(0:(length(mse_values)-1), mse_values, type = "l", col = mse_cols[1], lwd = 2,
       xlab = "Iteration", ylab = "MSE", main = "MSE and log-likelihood",
       ylim = c(0, max(c(mse_values, mse_gamma_values, mse_D_values, mse_sigma2_values), na.rm = TRUE)))
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
  lines(0:(length(mse_gamma_values)-1), mse_gamma_values, col = mse_cols[2], lwd = 2, lty = 2)
  lines(0:(length(mse_D_values)-1), mse_D_values, col = mse_cols[3], lwd = 2, lty = 3)
  lines(0:(length(mse_sigma2_values)-1), mse_sigma2_values, col = mse_cols[4], lwd = 2, lty = 4)

  # Add average marginal log-likelihood curve (read directly from the already-aggregated values)
  par(new = TRUE)
  avg_loglik <- result$loglik_iter_avg_history
  # Filter out invalid iteration slots
  avg_loglik <- avg_loglik[is.finite(avg_loglik)]
  if (length(avg_loglik) == 0) avg_loglik <- c(0, 0)

  loglik_min <- floor(min(avg_loglik, na.rm = TRUE))
  loglik_max <- ceiling(max(avg_loglik, na.rm = TRUE))
  loglik_breaks <- seq(loglik_min, loglik_max, length.out = 5)
  plot(0:(length(avg_loglik)-1), avg_loglik, type = "l", col = mse_cols[5], lwd = 2, lty = 5,
       xlab = "", ylab = "", axes = FALSE, ylim = c(loglik_min, loglik_max))
  axis(4, at = loglik_breaks, labels = format(loglik_breaks, digits = 2), col = mse_cols[5], col.axis = mse_cols[5])
  mtext("Log-likelihood", side = 4, line = 3, col = mse_cols[5])

  # Add legend
  legend("topright",
         legend = c("Overall MSE", expression(MSE[gamma]), expression(MSE[D]), expression(MSE[sigma^2]), "Log-likelihood"),
         col = mse_cols,
         lty = c(1, 2, 3, 4, 5),
         lwd = 2,
         bty = "n",
         cex = 0.7)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2, font = 2)

  # Reset graphics parameters
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot MSE evolution across multiple simulations
#' @param result result list from run_multiple_em_or_mcem
#' @param title plot title
plot_multiple_simulation_mse <- function(result, title = "MSE evolution across simulations") {
  # Get data dimensions
  p <- result$details[[1]]$data$p
  q <- result$details[[1]]$data$q

  # Compute per-iteration average MSE
  n_iter <- length(result$sigma2_iter_avg_history)
  avg_mse <- numeric(n_iter)
  avg_mse_gamma <- numeric(n_iter)
  avg_mse_D <- numeric(n_iter)
  avg_mse_sigma2 <- numeric(n_iter)

  for(k in 1:n_iter) {
    mse <- get_mse(result$gamma_iter_avg_history[,k],
                   result$D_iter_avg_history[,,k],
                   result$sigma2_iter_avg_history[k],
                   result$gamma_real,
                   result$D_real,
                   result$sigma2_real)
    avg_mse[k] <- mse$mse
    avg_mse_gamma[k] <- mse$mse_gamma
    avg_mse_D[k] <- mse$mse_D
    avg_mse_sigma2[k] <- mse$mse_sigma2
  }

  # Set graphics parameters
  par(mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))

  # Main plot
  plot(0:(n_iter-1), avg_mse, type = "l", col = "#1f77b4", lwd = 2,
       xlab = "Iteration", ylab = "MSE", main = "Average MSE evolution",
       ylim = c(0, max(avg_mse) * 1.1))
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

  # Add remaining MSE curves
  lines(0:(n_iter-1), avg_mse_gamma, col = "#d62728", lwd = 2, lty = 2)
  lines(0:(n_iter-1), avg_mse_D, col = "#2ca02c", lwd = 2, lty = 3)
  lines(0:(n_iter-1), avg_mse_sigma2, col = "#ff7f0e", lwd = 2, lty = 4)

  # Add legend
  legend("topright",
         legend = c("Overall MSE", expression(MSE[gamma]), expression(MSE[D]), expression(MSE[sigma^2])),
         col = c("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"),
         lty = c(1, 2, 3, 4),
         lwd = 2,
         bty = "n",
         cex = 0.8)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2, font = 2)

  # Reset graphics parameters
  par(mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot a zoomed-in view of the parameter iteration path
#' @param result result list from run_multiple_em_or_mcem
#' @param title plot title
#' @param zoom_factor zoom factor (default 0.1 means true-value +/-10% of range)
plot_zoomed_iterations <- function(result, title = "Zoomed parameter iteration path", zoom_factor = 0.1) {
  # Get data dimensions
  p <- result$details[[1]]$data$p
  q <- result$details[[1]]$data$q
  gamma_dim <- (p + 1) * (q + 1)
  D_dim <- p + 1

  # True parameter values
  gamma_real <- result$gamma_real
  D_real <- result$D_real
  sigma2_real <- result$sigma2_real

  # Set plot layout
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1), oma = c(0, 0, 3, 0))

  # 1. gamma iteration path (zoomed)
  gamma_cols <- get_nice_colors(gamma_dim)
  y_range <- range(c(result$gamma_iter_avg_history, gamma_real), na.rm = TRUE)
  y_center <- mean(y_range)
  y_span <- diff(y_range) * zoom_factor
  y_lim <- c(y_center - y_span, y_center + y_span)

  plot(0:(ncol(result$gamma_iter_avg_history)-1), result$gamma_iter_avg_history[1,],
       type = "l", col = gamma_cols[1], lwd = 2,
       xlab = "Iteration", ylab = expression(gamma),
       main = expression(gamma*" iteration path (zoomed)"),
       ylim = y_lim)
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
  abline(h = gamma_real[1], col = gamma_cols[1], lty = 2, lwd = 1.5)
  for(i in 2:gamma_dim) {
    lines(0:(ncol(result$gamma_iter_avg_history)-1), result$gamma_iter_avg_history[i,],
          col = gamma_cols[i], lwd = 2)
    abline(h = gamma_real[i], col = gamma_cols[i], lty = 2, lwd = 1.5)
  }
  gamma_labels <- paste0("gamma[", 1:gamma_dim, "]")
  legend("topright", legend = gamma_labels, col = gamma_cols, lty = 1, lwd = 2,
         bty = "n", cex = min(0.7, 2.5/gamma_dim), ncol = ceiling(gamma_dim/10))

  # 2. D matrix iteration path (zoomed)
  D_cols <- get_nice_colors(D_dim^2)
  D_real_vec <- as.vector(D_real)
  n_iter <- dim(result$D_iter_avg_history)[3]

  # Build matrix of D element histories
  D_elements <- matrix(0, nrow = D_dim^2, ncol = n_iter)
  for(i in 1:D_dim) {
    for(j in 1:D_dim) {
      idx <- (i-1)*D_dim + j
      D_elements[idx,] <- result$D_iter_avg_history[i,j,]
    }
  }

  # Compute zoomed y-axis range for D matrix
  y_range <- range(c(D_elements, D_real_vec), na.rm = TRUE)
  y_center <- mean(y_range)
  y_span <- diff(y_range) * zoom_factor
  y_lim <- c(y_center - y_span, y_center + y_span)

  # Plot D matrix elements
  for(i in 1:(D_dim^2)) {
    if(i == 1) {
      plot(0:(n_iter-1), D_elements[i,], type = "l", col = D_cols[i], lwd = 2,
           xlab = "Iteration", ylab = expression(D),
           main = expression(D*" matrix iteration path (zoomed)"),
           ylim = y_lim)
      grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
    } else {
      lines(0:(n_iter-1), D_elements[i,], col = D_cols[i], lwd = 2)
    }
    abline(h = D_real_vec[i], col = D_cols[i], lty = 2, lwd = 1.5)
  }
  D_labels <- paste0("D[", rep(1:D_dim, each=D_dim), ",", rep(1:D_dim, times=D_dim), "]")
  legend("topright", legend = D_labels, col = D_cols, lty = 1, lwd = 2,
         bty = "n", cex = min(0.7, 2.5/(D_dim^2)), ncol = ceiling((D_dim^2)/10))

  # 3. sigma2 iteration path (zoomed)
  y_range <- range(c(result$sigma2_iter_avg_history, sigma2_real), na.rm = TRUE)
  y_center <- mean(y_range)
  y_span <- diff(y_range) * zoom_factor
  y_lim <- c(y_center - y_span, y_center + y_span)

  # Determine suitable tick spacing
  y_breaks <- pretty(y_lim, n = 5)

  plot(0:(length(result$sigma2_iter_avg_history)-1), result$sigma2_iter_avg_history,
       type = "l", col = "#1f77b4", lwd = 2,
       xlab = "Iteration", ylab = expression(sigma^2),
       main = expression(sigma^2*" iteration path (zoomed)"),
       ylim = y_lim, yaxt = "n")
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

  # Add y-axis ticks using %g format
  axis(2, at = y_breaks, labels = sprintf("%g", y_breaks))

  # Add dashed line for the true value
  abline(h = sigma2_real, col = "#1f77b4", lty = 2, lwd = 1.5)

  # 4. MSE evolution (zoomed)
  # Compute per-iteration MSE
  mse_history <- list()
  for(k in 1:length(result$sigma2_iter_avg_history)) {
    mse_history[[k]] <- get_mse(result$gamma_iter_avg_history[,k],
                               result$D_iter_avg_history[,,k],
                               result$sigma2_iter_avg_history[k],
                               result$gamma_real,
                               result$D_real,
                               result$sigma2_real)
  }

  # Extract MSE values
  mse_values <- sapply(mse_history, function(x) x$mse)
  mse_gamma_values <- sapply(mse_history, function(x) x$mse_gamma)
  mse_D_values <- sapply(mse_history, function(x) x$mse_D)
  mse_sigma2_values <- sapply(mse_history, function(x) x$mse_sigma2)

  # Compute zoomed y-axis range for MSE
  y_range <- range(c(mse_values, mse_gamma_values, mse_D_values, mse_sigma2_values), na.rm = TRUE)
  y_center <- mean(y_range)
  y_span <- diff(y_range) * zoom_factor
  y_lim <- c(y_center - y_span, y_center + y_span)

  # Create dual-axis plot
  par(mar = c(4, 4, 2, 4))
  plot(0:(length(mse_values)-1), mse_values, type = "l", col = "#1f77b4", lwd = 2,
       xlab = "Iteration", ylab = "MSE", main = "MSE evolution (zoomed)",
       ylim = y_lim)
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
  lines(0:(length(mse_gamma_values)-1), mse_gamma_values, col = "#d62728", lwd = 2, lty = 2)
  lines(0:(length(mse_D_values)-1), mse_D_values, col = "#2ca02c", lwd = 2, lty = 3)
  lines(0:(length(mse_sigma2_values)-1), mse_sigma2_values, col = "#ff7f0e", lwd = 2, lty = 4)

  # Add legend
  legend("topright",
         legend = c("Overall MSE", expression(MSE[gamma]), expression(MSE[D]), expression(MSE[sigma^2])),
         col = c("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"),
         lty = c(1, 2, 3, 4),
         lwd = 2,
         bty = "n",
         cex = 0.8)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2, font = 2)

  # Reset graphics parameters
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}

#' Plot MSE evolution for a single simulation run
#' @param result result list from run_single_em_or_mcem
#' @param title plot title
plot_single_mse <- function(result, title = "MSE evolution (single run)") {
  # Check input
  if (is.null(result$history)) {
    stop("Input result is missing the required iteration history")
  }

  # Set graphics parameters
  par(mar = c(4, 4, 2, 1), oma = c(0, 0, 3, 0))

  # Compute per-iteration MSE
  mse_history <- list()
  for(i in 1:length(result$history)) {
    if(!is.null(result$history[[i]]$gamma) && !is.null(result$history[[i]]$D) && !is.null(result$history[[i]]$sigma2)) {
      mse_history[[i]] <- get_mse(result$history[[i]]$gamma,
                                 result$history[[i]]$D,
                                 result$history[[i]]$sigma2,
                                 result$data$gamma_real,
                                 result$data$D_real,
                                 result$data$sigma2_real)
    }
  }

  # Extract MSE values
  mse_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse else NA)
  mse_gamma_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse_gamma else NA)
  mse_D_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse_D else NA)
  mse_sigma2_values <- sapply(mse_history, function(x) if(!is.null(x)) x$mse_sigma2 else NA)

  # Colours
  colors <- c("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e")

  # Base plot
  plot(0:(length(mse_values)-1), mse_values, type = "l", col = colors[1], lwd = 2,
       xlab = "Iteration", ylab = "MSE", main = "Per-parameter MSE evolution",
       ylim = c(0, max(c(mse_values, mse_gamma_values, mse_D_values, mse_sigma2_values), na.rm = TRUE)))
  grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")

  # Add remaining MSE curves
  lines(0:(length(mse_gamma_values)-1), mse_gamma_values, col = colors[2], lwd = 2, lty = 2)
  lines(0:(length(mse_D_values)-1), mse_D_values, col = colors[3], lwd = 2, lty = 3)
  lines(0:(length(mse_sigma2_values)-1), mse_sigma2_values, col = colors[4], lwd = 2, lty = 4)

  # Add legend
  legend("topright",
         legend = c("Overall MSE", expression(MSE[gamma]), expression(MSE[D]), expression(MSE[sigma^2])),
         col = colors,
         lty = c(1, 2, 3, 4),
         lwd = 2,
         bty = "n",
         cex = 0.8)

  # Add overall title
  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2, font = 2)

  # Reset graphics parameters
  par(mar = c(5, 4, 4, 2) + 0.3, oma = c(0, 0, 0, 0))
}
