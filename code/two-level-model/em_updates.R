# ============================================================================
# EM Update Functions
#   E-step and M-step implementations for the EM algorithm.
#   Model: y_j = X_j W_j gamma + X_j mu_j + epsilon_j,
#          mu_j ~ N(0, D),  epsilon_j ~ N(0, sigma^2 I)
# ============================================================================

# EM update for gamma (GLS form corresponding to paper eq. 4-25b; converges to the same MLE as
# the strict EM eq. 4-25 but is a one-step closed-form solution, avoiding the extremely slow
# convergence of 4-25 when variance components are large)
# gamma_{k+1} = (sum_j W_j^T Lambda_j^{-1} W_j)^{-1} sum_j W_j^T Lambda_j^{-1} beta_j^{OLS}
# where Lambda_j = D_k + sigma^2_k (X_j^T X_j)^{-1}, beta_j^{OLS} = (X_j^T X_j)^{-1} X_j^T y_j
em_update_gamma <- function(data, D_k, sigma2_k){
  sigma2_k <- as.numeric(sigma2_k)[1]
  shape <- (data$p + 1) * (data$q + 1)
  part_1 <- matrix(0, nrow = shape, ncol = shape)
  part_2 <- matrix(0, nrow = shape, ncol = 1)
  for (j in 1:data$J){
    X_j <- get_X_j(data, j)
    y_j <- get_y_j(data, j)
    W_j <- get_W_j(data, j)
    XtX_inv <- ginv(t(X_j) %*% X_j)
    big_gamma_j <- D_k + sigma2_k * XtX_inv
    inv_big_gamma_j <- ginv(big_gamma_j)
    beta_j <- XtX_inv %*% t(X_j) %*% y_j
    part_1 <- part_1 + t(W_j) %*% inv_big_gamma_j %*% W_j
    part_2 <- part_2 + t(W_j) %*% inv_big_gamma_j %*% beta_j
  }
  return(ginv(part_1) %*% part_2)
}

# REML helper: evaluated at (D_k, sigma^2_k), returns the covariance of gamma,
# C = (sum_j W_j' Lambda_j^{-1} W_j)^{-1}, and the group-wise Lambda_j^{-1},
# where Lambda_j = D_k + sigma^2_k (X_j'X_j)^{-1} is the marginal covariance of the
# group-level OLS estimate (identity X_j' V_j^{-1} X_j = Lambda_j^{-1}).
# C is the covariance of gamma-hat under GLS / marginal MLE.
get_reml_pieces <- function(data, D_k, sigma2_k) {
  sigma2_k <- as.numeric(sigma2_k)[1]
  shape <- (data$p + 1) * (data$q + 1)
  Cinv <- matrix(0, shape, shape)
  Linv_list <- vector("list", data$J)
  for (j in 1:data$J) {
    X_j <- get_X_j(data, j); W_j <- get_W_j(data, j)
    XtX_inv <- ginv(t(X_j) %*% X_j)
    Linv <- ginv(D_k + sigma2_k * XtX_inv)
    Linv_list[[j]] <- Linv
    Cinv <- Cinv + t(W_j) %*% Linv %*% W_j
  }
  list(C = ginv(Cinv), Cinv = Cinv, Linv = Linv_list)
}

# EM update for D (REML correction). The ML update D=(1/J)sum(m_j m_j'+V_j) suffers from a
# small-sample downward bias of approximately (J-(q+1))/J after the mean structure W_j*gamma
# has been estimated (q+1 group-level coefficients per component). REML corrects this by
# propagating the uncertainty of gamma-hat into the random effects: the random-effects predictor
# mu-hat_j = D X_j' V_j^{-1} (y_j - X_j W_j gamma-hat) satisfies
# d(mu-hat_j)/d(gamma-hat) = -D Lambda_j^{-1} W_j =: -B_j, so E[mu_j mu_j'] gains an
# additional term B_j Cov(gamma-hat) B_j' = B_j C B_j'.
# D_REML = (1/J) sum_j ( m_j m_j' + V_j + B_j C B_j' ), which is approximately unbiased for
# any number of groups J.
# REML is enabled via options(hlm_reml=TRUE) (simulation drivers only; empirical pipelines
# default to ML for element-wise comparison with lme4 ML results, see section 7.2).
# Under ML this reduces to D=(1/J)sum(m_j m_j'+V_j).
em_update_D <- function(data, gamma_k, D_k, sigma2_k){
  sigma2_k <- as.numeric(sigma2_k)[1]
  J <- data$J
  reml <- isTRUE(getOption("hlm_reml", FALSE))
  rp <- if (reml) get_reml_pieces(data, D_k, sigma2_k) else NULL   # evaluated at D_k, sigma^2_k, consistent with E-step
  D_k_new <- matrix(0, nrow = data$p + 1, ncol = data$p + 1)
  for (j in 1:data$J){
    mu_jk_star_prime <- get_mu_jk_star(data, j, gamma_k, D_k, sigma2_k)
    V_jk_star <- get_V_jk_star(data, j, D_k, sigma2_k)
    term <- mu_jk_star_prime %*% t(mu_jk_star_prime) + V_jk_star
    if (reml) {
      W_j <- get_W_j(data, j)
      B_j <- D_k %*% rp$Linv[[j]] %*% W_j          # (p+1) x dim(gamma), d(mu-hat_j)/d(gamma-hat) = -B_j
      term <- term + B_j %*% rp$C %*% t(B_j)       # gamma-hat uncertainty propagation term B_j Cov(gamma-hat) B_j'
    }
    D_k_new <- D_k_new + term
  }
  D_k_new <- D_k_new / J
  return(D_k_new)
}

# Implementation of the matrix trace function
trace <- function(X) {
  sum(diag(X))
}

# Under REML, also add tr(A_j C A_j') with A_j = X_j W_j - X_j D Lambda_j^{-1} W_j (= -d(residual_j)/d(gamma-hat)),
# propagating the gamma-hat uncertainty into the residual sum of squares. This is the special case (fixed design
# X_j W_j, random design X_j) of the general LMM sigma^2 REML correction and matches lme4(REML=TRUE). Under ML the
# term is absent, leaving the standard divisor-n update.
em_update_sigma2 <- function(data, gamma_kp1, D_kp1, sigma2_k) {
  sigma2_k <- as.numeric(sigma2_k)[1]
  sample_size <- data$sample_size
  reml <- isTRUE(getOption("hlm_reml", FALSE))
  rp <- if (reml) get_reml_pieces(data, D_kp1, sigma2_k) else NULL   # evaluated at (D_kp1, sigma^2_k)
  sigma2_k_new <- 0
  for (j in 1:data$J){
    X_j <- get_X_j(data, j)
    y_j <- get_y_j(data, j)
    W_j <- get_W_j(data, j)
    mu_jk_star_2prime <- get_mu_jk_star(data, j, gamma_kp1, D_kp1, sigma2_k)
    V_jk_star_prime <- get_V_jk_star(data, j, D_kp1, sigma2_k)
    residual <- y_j - X_j %*% W_j %*% gamma_kp1 - X_j %*% mu_jk_star_2prime
    part_1 <- t(residual) %*% residual
    part_2 <- trace(X_j %*% V_jk_star_prime %*% t(X_j))
    sigma2_k_new <- sigma2_k_new + part_1 + part_2
    if (reml) {
      A_j <- X_j %*% W_j - X_j %*% D_kp1 %*% rp$Linv[[j]] %*% W_j   # gamma-hat uncertainty entering the residual
      sigma2_k_new <- sigma2_k_new + trace(A_j %*% rp$C %*% t(A_j))
    }
  }
  sigma2_k_new <- sigma2_k_new / sample_size
  return(sigma2_k_new)
}
