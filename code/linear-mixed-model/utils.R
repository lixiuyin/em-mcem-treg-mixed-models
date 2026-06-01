library(this.path)
source(file.path(this.dir(), "data_generation.R"), encoding = "UTF-8")

# ============================================================================
# General Linear Mixed Model (LMM): EM / MCEM Estimation
#   y_i = X_i beta + Z_i u_i + epsilon_i,  u_i ~ N(0, G0),  epsilon_i ~ N(0, sigma^2 I)
#   Latent variables = random effects u = (u_1, ..., u_m)
#
# Prior: flat non-informative prior pi(beta, G0, sigma^2) proportional to 1.
# ML:   G0 M-step denominator = m (number of groups); sigma^2 denominator = n.
# REML: enable options(lmm_reml=TRUE) to propagate beta-hat uncertainty into
#       the G0 and sigma^2 updates and evaluate the restricted log-likelihood.
#       This removes the small-sample ML downward bias in variance components
#       and matches lme4(REML=TRUE) in the library comparison driver.
# ============================================================================

# Robust matrix inversion: prefer solve(), fall back to regularised ginv on failure
safe_solve <- function(A) {
  tryCatch(solve(A), error = function(e) ginv(A + 1e-8 * diag(nrow(A))))
}

# Extract the fixed-effects design matrix X_i (n_i x p_fixed) for group i
get_X_i <- function(data, i) {
  gs <- data$group_sizes
  row_start <- if (i == 1) 1 else sum(gs[1:(i - 1)]) + 1
  row_end <- sum(gs[1:i])
  data$X[row_start:row_end, , drop = FALSE]
}

# Random-effects design matrix Z_i = X_i[, 1:k] for group i
get_Z_i <- function(data, i) {
  get_X_i(data, i)[, 1:data$k, drop = FALSE]
}

# Extract the response vector y_i for group i
get_y_i <- function(data, i) {
  gs <- data$group_sizes
  row_start <- if (i == 1) 1 else sum(gs[1:(i - 1)]) + 1
  row_end <- sum(gs[1:i])
  matrix(data$y[row_start:row_end], ncol = 1)
}

# E-step: posterior of random effects for group i: u_i | y_i ~ N(u_star, V_star)
#   u_star = (Z_i'Z_i + sigma^2 G0^{-1})^{-1} Z_i' (y_i - X_i beta)
#   V_star = sigma^2 (Z_i'Z_i + sigma^2 G0^{-1})^{-1}
get_post_u <- function(data, i, beta, G0, sigma2) {
  sigma2 <- as.numeric(sigma2)[1]
  Z_i <- get_Z_i(data, i)
  X_i <- get_X_i(data, i)
  y_i <- get_y_i(data, i)
  M_i <- safe_solve(t(Z_i) %*% Z_i + sigma2 * safe_solve(G0))
  u_star <- M_i %*% t(Z_i) %*% (y_i - X_i %*% beta)
  V_star <- sigma2 * M_i
  list(u_star = u_star, V_star = V_star)
}

# Marginal covariance for group i: V_i = Z_i G0 Z_i' + sigma^2 I_{n_i}
get_V_i <- function(data, i, G0, sigma2) {
  sigma2 <- as.numeric(sigma2)[1]
  Z_i <- get_Z_i(data, i)
  Z_i %*% G0 %*% t(Z_i) + sigma2 * diag(nrow(Z_i))
}

# ---- EM M-step ----

# beta update: marginal GLS given (G0, sigma^2) -- equivalent to marginal ML / ECME, closed form
#   beta = (sum X_i' V_i^{-1} X_i)^{-1} sum X_i' V_i^{-1} y_i
em_update_beta <- function(data, G0, sigma2) {
  p <- data$p_fixed
  A <- matrix(0, p, p)
  b <- matrix(0, p, 1)
  for (i in 1:data$m) {
    X_i <- get_X_i(data, i)
    y_i <- get_y_i(data, i)
    Vinv <- safe_solve(get_V_i(data, i, G0, sigma2))
    A <- A + t(X_i) %*% Vinv %*% X_i
    b <- b + t(X_i) %*% Vinv %*% y_i
  }
  safe_solve(A) %*% b
}

# REML helper: covariance of beta-hat under marginal GLS, evaluated at (G0, sigma^2),
# and the group-wise V_i^{-1}. C = (sum X_i' V_i^{-1} X_i)^{-1}.
get_reml_pieces <- function(data, G0, sigma2) {
  p <- data$p_fixed
  Cinv <- matrix(0, p, p)
  Vinv_list <- vector("list", data$m)
  for (i in 1:data$m) {
    X_i <- get_X_i(data, i)
    Vinv <- safe_solve(get_V_i(data, i, G0, sigma2))
    Vinv_list[[i]] <- Vinv
    Cinv <- Cinv + t(X_i) %*% Vinv %*% X_i
  }
  list(C = safe_solve(Cinv), Cinv = Cinv, Vinv = Vinv_list)
}

# G0 update. ML uses G0 = (1/m) sum (u* u*' + V*).
# REML adds B_i C B_i', where B_i = G0 Z_i' V_i^{-1} X_i, to account for the
# uncertainty in beta-hat that ML treats as fixed.
em_update_G0 <- function(data, beta, G0, sigma2) {
  k <- data$k
  reml <- isTRUE(getOption("lmm_reml", FALSE))
  rp <- if (reml) get_reml_pieces(data, G0, sigma2) else NULL
  G0_new <- matrix(0, k, k)
  for (i in 1:data$m) {
    post <- get_post_u(data, i, beta, G0, sigma2)
    term <- post$u_star %*% t(post$u_star) + post$V_star
    if (reml) {
      X_i <- get_X_i(data, i)
      Z_i <- get_Z_i(data, i)
      B_i <- G0 %*% t(Z_i) %*% rp$Vinv[[i]] %*% X_i
      term <- term + B_i %*% rp$C %*% t(B_i)
    }
    G0_new <- G0_new + term
  }
  G0_new <- G0_new / data$m
  (G0_new + t(G0_new)) / 2   # enforce symmetry
}

# sigma^2 update: sigma^2 = (1/n) sum [ ||y_i - X_i beta - Z_i u*||^2 + tr(Z_i V* Z_i') ].
# Under REML, add tr(A_i C A_i') with A_i = X_i - Z_i G0 Z_i' V_i^{-1} X_i.
# For the no-random-effects special case this fixed point reduces to RSS/(n-p),
# so the divisor remains n while the correction supplies the lost fixed-effect df.
em_update_sigma2 <- function(data, beta, G0, sigma2) {
  total <- 0
  reml <- isTRUE(getOption("lmm_reml", FALSE))
  rp <- if (reml) get_reml_pieces(data, G0, sigma2) else NULL
  for (i in 1:data$m) {
    X_i <- get_X_i(data, i)
    Z_i <- get_Z_i(data, i)
    y_i <- get_y_i(data, i)
    post <- get_post_u(data, i, beta, G0, sigma2)
    residual <- y_i - X_i %*% beta - Z_i %*% post$u_star
    total <- total + sum(residual^2) + sum(diag(Z_i %*% post$V_star %*% t(Z_i)))
    if (reml) {
      A_i <- X_i - Z_i %*% G0 %*% t(Z_i) %*% rp$Vinv[[i]] %*% X_i
      total <- total + sum(diag(A_i %*% rp$C %*% t(A_i)))
    }
  }
  as.numeric(total / data$n)
}

# ---- MCEM M-step (beta still uses closed-form GLS; G0 and sigma^2 use Monte Carlo approximation) ----

# Draw M posterior samples of random effects for each group; return list of k x M matrices (one per group).
# Under REML, inflate the sampling covariance so the Monte Carlo second moment matches
# u*u' + V* + B C B' in the G0 update.
get_u_samples <- function(data, beta, G0, sigma2, M, reml = FALSE) {
  samples <- vector("list", data$m)
  rp <- if (reml) get_reml_pieces(data, G0, sigma2) else NULL
  for (i in 1:data$m) {
    post <- get_post_u(data, i, beta, G0, sigma2)
    Sigma_i <- post$V_star
    if (reml) {
      X_i <- get_X_i(data, i)
      Z_i <- get_Z_i(data, i)
      B_i <- G0 %*% t(Z_i) %*% rp$Vinv[[i]] %*% X_i
      Sigma_i <- Sigma_i + B_i %*% rp$C %*% t(B_i)
    }
    s <- mvrnorm(n = M, mu = as.numeric(post$u_star), Sigma = Sigma_i)
    samples[[i]] <- t(matrix(s, nrow = M))   # k x M
  }
  samples
}

mcem_update_G0 <- function(data, M, u_samples) {
  k <- data$k
  G0_new <- matrix(0, k, k)
  for (mc in 1:M) {
    for (i in 1:data$m) {
      u_im <- u_samples[[i]][, mc, drop = FALSE]
      G0_new <- G0_new + u_im %*% t(u_im)
    }
  }
  G0_new <- G0_new / (M * data$m)
  G0_new <- (G0_new + t(G0_new)) / 2
  # enforce positive definiteness
  ev <- eigen(G0_new, symmetric = TRUE)
  if (min(ev$values) < 1e-8) {
    ev$values[ev$values < 1e-8] <- 1e-8
    G0_new <- ev$vectors %*% diag(ev$values, nrow = k) %*% t(ev$vectors)
  }
  G0_new
}

mcem_update_sigma2 <- function(data, beta, M, u_samples, G0 = NULL, sigma2 = NULL, reml = FALSE) {
  total <- 0
  fixed_effect_total <- 0
  rp <- NULL
  if (reml) {
    if (is.null(G0) || is.null(sigma2)) stop("G0 and sigma2 are required for REML MCEM sigma2 update")
    rp <- get_reml_pieces(data, G0, sigma2)
    for (i in 1:data$m) {
      X_i <- get_X_i(data, i)
      Z_i <- get_Z_i(data, i)
      A_i <- X_i - Z_i %*% G0 %*% t(Z_i) %*% rp$Vinv[[i]] %*% X_i
      fixed_effect_total <- fixed_effect_total + sum(diag(A_i %*% rp$C %*% t(A_i)))
    }
  }
  for (mc in 1:M) {
    for (i in 1:data$m) {
      X_i <- get_X_i(data, i)
      Z_i <- get_Z_i(data, i)
      y_i <- get_y_i(data, i)
      u_im <- u_samples[[i]][, mc, drop = FALSE]
      residual <- y_i - X_i %*% beta - Z_i %*% u_im
      total <- total + sum(residual^2)
    }
  }
  if (reml) total <- total + M * fixed_effect_total
  as.numeric(total / (M * data$n))
}

mcem_update_all <- function(data, beta, G0, sigma2, M) {
  # Aligned with the ECM structure of EM: beta is closed-form; resample using updated params before each step
  beta_new <- em_update_beta(data, G0, sigma2)
  reml <- isTRUE(getOption("lmm_reml", FALSE))
  u_for_G0 <- get_u_samples(data, beta_new, G0, sigma2, M, reml = reml)
  G0_new <- mcem_update_G0(data, M, u_for_G0)
  u_for_sigma <- get_u_samples(data, beta_new, G0_new, sigma2, M)
  sigma2_new <- mcem_update_sigma2(data, beta_new, M, u_for_sigma,
                                   G0 = G0_new, sigma2 = sigma2, reml = reml)
  list(beta_new = beta_new, G0_new = G0_new, sigma2_new = sigma2_new)
}

# Observed-data marginal log-likelihood: ln p(y | beta, G0, sigma^2) = sum_i ln N(y_i; X_i beta, V_i)
get_loglik <- function(data, beta, G0, sigma2) {
  sigma2 <- as.numeric(sigma2)[1]
  # enforce G0 positive definiteness
  ev <- eigen(G0, symmetric = TRUE)
  if (min(ev$values) < 1e-8) {
    ev$values[ev$values < 1e-8] <- 1e-8
    G0 <- ev$vectors %*% diag(ev$values, nrow = data$k) %*% t(ev$vectors)
  }
  loglik <- 0
  for (i in 1:data$m) {
    X_i <- get_X_i(data, i)
    y_i <- get_y_i(data, i)
    n_i <- nrow(X_i)
    V_i <- get_V_i(data, i, G0, sigma2)
    residual <- y_i - X_i %*% beta
    chol_V <- tryCatch(chol(V_i), error = function(e) chol(V_i + 1e-8 * diag(n_i)))
    log_det <- 2 * sum(log(diag(chol_V)))
    quad <- sum(backsolve(chol_V, residual, transpose = TRUE)^2)
    loglik <- loglik - 0.5 * n_i * log(2 * pi) - 0.5 * log_det - 0.5 * quad
  }
  if (isTRUE(getOption("lmm_reml", FALSE))) {
    p_beta <- data$p_fixed
    Cinv <- get_reml_pieces(data, G0, sigma2)$Cinv
    loglik <- loglik - 0.5 * as.numeric(determinant(Cinv, logarithm = TRUE)$modulus) +
      0.5 * p_beta * log(2 * pi)
  }
  as.numeric(loglik)
}

# Evaluation metrics: beta uses L2 norm, G0 uses Frobenius norm, sigma^2 uses absolute value;
# the sum is the total metric (following the convention in Chapter 4 of the paper).
# Note: when used for a single estimate this is the estimation error norm; when used for the
# "average estimate across multiple simulations" it yields the bias norm (not the average
# per-run squared error) -- interpret accordingly.
get_mse <- function(beta_hat, G0_hat, sigma2_hat, beta_real, G0_real, sigma2_real) {
  beta_hat <- as.matrix(beta_hat)
  beta_real <- as.matrix(beta_real)
  mse_beta <- norm(beta_hat - beta_real, "2")
  mse_G0 <- norm(G0_hat - G0_real, "F")
  mse_sigma2 <- abs(as.numeric(sigma2_hat)[1] - as.numeric(sigma2_real)[1])
  list(mse_beta = mse_beta, mse_G0 = mse_G0, mse_sigma2 = mse_sigma2,
       mse = mse_beta + mse_G0 + mse_sigma2)
}

# ---- Single run of EM / MCEM ----
run_single_em_or_mcem <- function(data, beta_0 = NULL, G0_0 = NULL, sigma2_0 = NULL,
                                   max_iter = 100, tolerance = 1e-6,
                                   bold_random = FALSE, is_mcem = FALSE, mc_times = 100) {
  if (max_iter <= 0 || tolerance <= 0 || mc_times <= 0) stop("Iteration parameters must be positive")
  p_fixed <- data$p_fixed
  k <- data$k

  if (is.null(beta_0) || is.null(G0_0) || is.null(sigma2_0)) {
    init <- if (bold_random) bold_random_init(p_fixed, k) else random_init(p_fixed, k)
    if (is.null(beta_0))   beta_0   <- init$beta_init
    if (is.null(G0_0))     G0_0     <- init$G0_init
    if (is.null(sigma2_0)) sigma2_0 <- init$sigma2_init
  }

  history <- vector("list", max_iter + 1)
  beta_hat <- matrix(beta_0, ncol = 1)
  G0_hat <- G0_0
  sigma2_hat <- as.numeric(sigma2_0)[1]
  history[[1]] <- list(beta = beta_hat, G0 = G0_hat, sigma2 = sigma2_hat,
                       loglik = get_loglik(data, beta_hat, G0_hat, sigma2_hat))

  iter_count <- 0
  for (idx in 2:(max_iter + 1)) {
    if (is_mcem) {
      upd <- mcem_update_all(data, beta_hat, G0_hat, sigma2_hat, mc_times)
      beta_new <- upd$beta_new; G0_new <- upd$G0_new; sigma2_new <- upd$sigma2_new
    } else {
      beta_new <- em_update_beta(data, G0_hat, sigma2_hat)
      G0_new <- em_update_G0(data, beta_new, G0_hat, sigma2_hat)
      sigma2_new <- em_update_sigma2(data, beta_new, G0_new, sigma2_hat)
    }
    max_diff <- max(norm(beta_new - beta_hat, "2"),
                    norm(G0_new - G0_hat, "F"),
                    abs(sigma2_new - sigma2_hat))
    beta_hat <- beta_new; G0_hat <- G0_new; sigma2_hat <- sigma2_new
    history[[idx]] <- list(beta = beta_hat, G0 = G0_hat, sigma2 = sigma2_hat,
                          loglik = get_loglik(data, beta_hat, G0_hat, sigma2_hat))
    iter_count <- iter_count + 1
    if (max_diff < tolerance) break
  }

  list(final_iter_count = iter_count,
       final_beta_hat = beta_hat, final_G0_hat = G0_hat, final_sigma2_hat = sigma2_hat,
       history = history[1:(iter_count + 1)], data = data)
}

# ---- Multiple simulation runs, averaged ----
run_multiple_em_or_mcem <- function(data, n_sim = 50, max_iter = 100, tolerance = 1e-6,
                                     is_mcem = FALSE, mc_times = 100, reset_dataset = FALSE,
                                     bold_random = FALSE, beta_0 = NULL, G0_0 = NULL, sigma2_0 = NULL) {
  if (n_sim <= 0) stop("n_sim must be a positive integer")
  m <- data$m; k <- data$k; p_fixed <- data$p_fixed
  group_sizes <- data$group_sizes
  beta_real <- data$beta_real; G0_real <- data$G0_real; sigma2_real <- data$sigma2_real

  final_betas <- matrix(0, p_fixed, n_sim)
  final_G0s <- array(0, dim = c(k, k, n_sim))
  final_sigma2s <- numeric(n_sim)
  final_iters <- integer(n_sim)
  details <- vector("list", n_sim)

  for (s in 1:n_sim) {
    if (reset_dataset) {
      data <- generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real)
    }
    # Independent initial values per simulation run (randomly drawn only when missing; not written back to function params)
    init <- if (bold_random) bold_random_init(p_fixed, k) else random_init(p_fixed, k)
    beta_i   <- if (is.null(beta_0))   init$beta_init   else beta_0
    G0_i     <- if (is.null(G0_0))     init$G0_init     else G0_0
    sigma2_i <- if (is.null(sigma2_0)) init$sigma2_init else sigma2_0

    res <- run_single_em_or_mcem(data, beta_0 = beta_i, G0_0 = G0_i, sigma2_0 = sigma2_i,
                                 max_iter = max_iter, tolerance = tolerance,
                                 is_mcem = is_mcem, mc_times = mc_times)
    final_betas[, s] <- res$final_beta_hat
    final_G0s[, , s] <- res$final_G0_hat
    final_sigma2s[s] <- res$final_sigma2_hat
    final_iters[s] <- res$final_iter_count
    details[[s]] <- res
  }

  list(n_sim = n_sim,
       final_iter_count = mean(final_iters),
       final_beta_hat = rowMeans(final_betas),
       final_G0_hat = apply(final_G0s, c(1, 2), mean),
       final_sigma2_hat = mean(final_sigma2s),
       beta_real = beta_real, G0_real = G0_real, sigma2_real = sigma2_real,
       details = details)
}

# ---- Plot: single-run iteration trajectory + log-likelihood ----
plot_single_iteration <- function(result, title = "LMM EM Iteration Process") {
  history <- result$history
  data <- result$data
  beta_real <- as.numeric(data$beta_real)
  G0_real <- data$G0_real
  sigma2_real <- data$sigma2_real
  p_fixed <- data$p_fixed; k <- data$k
  n_iter <- length(history)
  idx <- 0:(n_iter - 1)

  beta_hist <- sapply(history, function(h) as.numeric(h$beta))    # p x n_iter
  G0_hist <- sapply(history, function(h) as.numeric(h$G0))        # k^2 x n_iter
  s2_hist <- sapply(history, function(h) h$sigma2)
  ll_hist <- sapply(history, function(h) h$loglik)

  par(mfrow = c(2, 2), mar = c(4, 4, 2.5, 1), oma = c(0, 0, 3, 0))

  cols_b <- get_nice_colors(p_fixed)
  matplot(idx, t(beta_hist), type = "l", lty = 1, lwd = 2, col = cols_b,
          xlab = "Iterations", ylab = expression(beta), main = expression(beta * " parameter iteration"),
          ylim = range(c(beta_hist, beta_real)))
  abline(h = beta_real, col = cols_b, lty = 2)
  legend("right", legend = paste0("beta[", 0:(p_fixed - 1), "]"), col = cols_b, lty = 1, lwd = 2, bty = "n", cex = 0.7)

  cols_g <- get_nice_colors(k * k)
  matplot(idx, t(G0_hist), type = "l", lty = 1, lwd = 2, col = cols_g,
          xlab = "Iterations", ylab = expression(G[0]), main = expression(G[0] * " matrix iteration"),
          ylim = range(c(G0_hist, as.numeric(G0_real))))
  abline(h = as.numeric(G0_real), col = cols_g, lty = 2)

  plot(idx, s2_hist, type = "l", lwd = 2, col = "#1f77b4",
       xlab = "Iterations", ylab = expression(sigma^2), main = expression(sigma^2 * " parameter iteration"),
       ylim = range(c(0, s2_hist, sigma2_real)))
  abline(h = sigma2_real, col = "#d62728", lty = 2, lwd = 1.5)
  legend("right", legend = c("Estimate", "True value"), col = c("#1f77b4", "#d62728"), lty = c(1, 2), lwd = 2, bty = "n", cex = 0.8)

  plot(idx, ll_hist, type = "l", lwd = 2, col = "#2ca02c",
       xlab = "Iterations", ylab = "Log-likelihood", main = "Observed-data marginal log-likelihood")
  grid()

  mtext(title, side = 3, line = 1, outer = TRUE, cex = 1.2, font = 2)
  par(mfrow = c(1, 1), mar = c(5, 4, 4, 2) + 0.1, oma = c(0, 0, 0, 0))
}

# ---- Average the per-iteration trajectory across simulations (for the 500-sim averaged figures) ----
# Each run is forward-filled to the longest run (its converged value is carried forward), then every
# iteration slot is averaged over all runs -- so the tail is not dominated by the few runs that
# iterate longest. The result is a history in the same format as a single run, so plot_single_iteration
# renders it directly. Averaging over many datasets cancels single-dataset sampling noise, so these
# curves track the true parameters (and the table means), unlike a single-run trajectory.
average_iteration_history <- function(result) {
  details <- result$details
  n_sim <- length(details)
  d0 <- details[[1]]$data
  p_fixed <- d0$p_fixed; k <- d0$k
  max_len <- max(vapply(details, function(d) length(d$history), integer(1)))

  beta_acc <- matrix(0, p_fixed, max_len)
  G0_acc <- array(0, c(k, k, max_len))
  s2_acc <- numeric(max_len); cnt <- integer(max_len)
  ll_acc <- numeric(max_len); ll_cnt <- integer(max_len)

  for (s in 1:n_sim) {
    h <- details[[s]]$history; H <- length(h)
    last_beta <- last_G0 <- last_s2 <- last_ll <- NULL
    for (idx in 1:max_len) {
      if (idx <= H) {
        last_beta <- as.numeric(h[[idx]]$beta)
        last_G0   <- h[[idx]]$G0
        last_s2   <- h[[idx]]$sigma2
        if (is.finite(h[[idx]]$loglik)) last_ll <- h[[idx]]$loglik
      }
      beta_acc[, idx] <- beta_acc[, idx] + last_beta
      G0_acc[, , idx] <- G0_acc[, , idx] + last_G0
      s2_acc[idx] <- s2_acc[idx] + last_s2
      cnt[idx] <- cnt[idx] + 1
      if (!is.null(last_ll)) { ll_acc[idx] <- ll_acc[idx] + last_ll; ll_cnt[idx] <- ll_cnt[idx] + 1 }
    }
  }

  lapply(1:max_len, function(idx) list(
    beta   = matrix(beta_acc[, idx] / cnt[idx], ncol = 1),
    G0     = G0_acc[, , idx] / cnt[idx],
    sigma2 = s2_acc[idx] / cnt[idx],
    loglik = if (ll_cnt[idx] > 0) ll_acc[idx] / ll_cnt[idx] else NA_real_
  ))
}

# ---- Plot: 500-simulation averaged iteration trajectory (reuses the single-run 2x2 layout) ----
plot_multiple_iteration <- function(result, title = "LMM: average iteration trajectory over simulations") {
  hist_avg <- average_iteration_history(result)
  plot_single_iteration(list(history = hist_avg, data = result$details[[1]]$data), title = title)
}

get_nice_colors <- function(n) {
  if (requireNamespace("RColorBrewer", quietly = TRUE)) {
    if (n <= 8) return(RColorBrewer::brewer.pal(max(3, n), "Set1")[1:n])
    if (n <= 12) return(RColorBrewer::brewer.pal(n, "Paired"))
    return(colorRampPalette(RColorBrewer::brewer.pal(12, "Paired"))(n))
  }
  grDevices::rainbow(n)
}
