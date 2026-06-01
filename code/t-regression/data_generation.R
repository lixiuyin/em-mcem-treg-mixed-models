library(scatterplot3d)
library(ggplot2)
library(tidyr)
library(dplyr)
library(gridExtra)
library(RColorBrewer)
library(grid)
library(viridis)

generate_t_simulated_data <- function(n, p, t_df, t_mu,
                                      t_sigma2_real, beta_real) {
  if (length(beta_real) != p + 1) {
    stop("Length of beta_real must equal the number of features plus 1")
  }
  sigma <- sqrt(t_sigma2_real) # scale parameter of the error term
  # Generate error term
  epsilon <- t_mu + sigma * rt(n, df = t_df)
  # Generate design matrix
  X <- cbind(1, matrix(rnorm(n * p), nrow = n))  # design matrix including intercept
  X_data <- data.frame(X)
  colnames(X_data) <- c("Intercept", paste0("X", 1:p))  # set column names
  # Generate response variable
  y_imag <- X %*% beta_real # response without error term
  y_real  <- y_imag + epsilon  # simulated response with error term
  list(
    y_imag = y_imag,
    y_real = y_real,
    X = X,
    X_df = X_data,
    beta_real = beta_real,
    epsilon = epsilon,
    t_df = t_df,
    t_mu = t_mu,
    t_sigma2_real = t_sigma2_real
  )
}

random_init <- function(p) {
  beta_init <- runif(p + 1, -1, 1)
  sigma2_init <- runif(1, 0.1, 1)
  init_result <- list(
    beta_init = beta_init,
    sigma2_init = sigma2_init
  )
  return(init_result)
}

bold_random_init <- function(p) {
  # Initialize beta and sigma^2 with a wider range
  beta_init <- runif(p + 1, -10, 10)
  sigma2_init <- runif(1, 0.01, 10)
  return(list(
    beta_init = beta_init,
    sigma2_init = sigma2_init
  ))
}
