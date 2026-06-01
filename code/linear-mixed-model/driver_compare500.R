# General Linear Mixed Model (LMM): EM vs MCEM (M sweep) over 500 simulations.
#
# PAIRED comparison: the 500 datasets are generated ONCE and every configuration (EM and MCEM at
# each M) is run on the SAME datasets, so differences across the MCEM columns reflect the Monte
# Carlo sample size M alone -- not dataset variation. (Running each config on freshly redrawn data
# would confound M with the data and can even make MCEM look better than EM, which is impossible on
# identical data, since MCEM only approximates EM's exact E-step.)
#
# Figures show the 500-simulation AVERAGED iteration trajectory, so they track the true values and
# the table means (a single-run trajectory would carry one dataset's sampling noise instead).
#
# Note: MCEM is slow; the full M sweep takes ~100 min sequentially. Set LMM_NSIM / LMM_MLIST for a
# quicker smoke run. Each config reseeds to 2025 so its Monte Carlo draws are reproducible.
library(this.path)
source(file.path(this.dir(), "utils.R"), encoding = "UTF-8")

set.seed(2025)

m <- 50
group_sizes <- sample(c(15, 20, 25), m, replace = TRUE)
beta_real <- c(1, 2, -1, 0.5)
G0_real <- matrix(c(4, 1,
                    1, 2), nrow = 2, byrow = TRUE)
sigma2_real <- 1

beta_0 <- matrix(rep(0, length(beta_real)), ncol = 1)
G0_0 <- diag(nrow(G0_real))
sigma2_0 <- 0.5

out_dir <- file.path(this.dir(), "figures")
if (!dir.exists(out_dir)) dir.create(out_dir)

save_plot <- function(name, expr, w = 1400, h = 1000) {
  png(file.path(out_dir, paste0(name, ".png")), width = w, height = h, res = 150)
  force(expr)
  dev.off()
  cat("Saved:", name, ".png\n")
}

N_SIM    <- as.integer(Sys.getenv("LMM_NSIM", "500"))
MAX_ITER <- as.integer(Sys.getenv("LMM_MAXITER", "50"))
M_LIST   <- as.integer(strsplit(Sys.getenv("LMM_MLIST", "20,50,100,200"), ",")[[1]])
M_TOP    <- max(M_LIST)

# Generate the shared datasets once (all configs see these same datasets)
datasets <- lapply(1:N_SIM, function(s) generate_lmm_data(m, group_sizes, beta_real, G0_real, sigma2_real))

# Run one configuration over the shared datasets; returns the same structure as
# run_multiple_em_or_mcem (so get_mse and plot_multiple_iteration work directly).
run_config <- function(is_mcem, M, tol) {
  set.seed(2025)   # reproducible Monte Carlo draws per config (the data is already fixed and shared)
  fb <- matrix(0, length(beta_real), N_SIM); fG <- array(0, c(2, 2, N_SIM))
  fs <- numeric(N_SIM); fi <- integer(N_SIM); details <- vector("list", N_SIM)
  t0 <- Sys.time()
  for (s in 1:N_SIM) {
    res <- run_single_em_or_mcem(datasets[[s]], beta_0 = beta_0, G0_0 = G0_0, sigma2_0 = sigma2_0,
                                 max_iter = MAX_ITER, tolerance = tol, is_mcem = is_mcem, mc_times = M)
    fb[, s] <- res$final_beta_hat; fG[, , s] <- res$final_G0_hat
    fs[s] <- res$final_sigma2_hat; fi[s] <- res$final_iter_count; details[[s]] <- res
  }
  list(n_sim = N_SIM, final_iter_count = mean(fi), final_beta_hat = rowMeans(fb),
       final_G0_hat = apply(fG, c(1, 2), mean), final_sigma2_hat = mean(fs),
       beta_real = beta_real, G0_real = G0_real, sigma2_real = sigma2_real, details = details,
       elapsed = as.numeric(Sys.time() - t0, units = "secs"))
}

# ----- EM -----
cat("\n>>>>> EM:", N_SIM, "simulation runs <<<<<\n")
em_500 <- run_config(is_mcem = FALSE, M = 100, tol = 1e-7)
save_plot("41_lmm_em_500sim_iter",
          plot_multiple_iteration(em_500, title = "LMM EM: average iteration trajectory over 500 simulations"))

# ----- MCEM at each M -----
mcem_list <- list()
for (M in M_LIST) {
  cat(sprintf("\n>>>>> MCEM(M=%d): %d simulation runs <<<<<\n", M, N_SIM))
  mcem_list[[as.character(M)]] <- run_config(is_mcem = TRUE, M = M, tol = 1e-6)
}
save_plot(sprintf("42_lmm_mcem_M%d_500sim_iter", M_TOP),
          plot_multiple_iteration(mcem_list[[as.character(M_TOP)]],
                                  title = sprintf("LMM MCEM(M=%d): average iteration trajectory over 500 simulations", M_TOP)))

# ----- Summary table -----
fmt_vec <- function(v, d = 3) paste0("(", paste(sprintf(paste0("%.", d, "f"), v), collapse = ", "), ")")
summarize_row <- function(label, res) {
  mse <- get_mse(res$final_beta_hat, res$final_G0_hat, res$final_sigma2_hat, beta_real, G0_real, sigma2_real)
  cat(sprintf("%-12s beta=%-32s G0diag=%-16s sigma2=%.4f  totalMSE=%.5f  iters=%.2f  time=%.1fs\n",
              label, fmt_vec(res$final_beta_hat), fmt_vec(diag(res$final_G0_hat)),
              res$final_sigma2_hat, mse$mse, res$final_iter_count, res$elapsed))
}
cat("\n========== LMM: 500-run PAIRED comparison (same datasets across all configs) ==========\n")
cat("True: beta=(1, 2, -1, 0.5)  G0 diag=(4, 2)  sigma^2=1\n")
summarize_row("EM", em_500)
for (M in M_LIST) summarize_row(sprintf("MCEM(M=%d)", M), mcem_list[[as.character(M)]])
