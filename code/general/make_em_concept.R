# EM concept illustration (thesis 图 2.1): the observed-data log-likelihood
# ln L(theta | Y) and a lower bound B(theta | theta_k) that touches it at theta_k.
# Maximizing B (the M-step) moves theta_k -> theta_{k+1} and necessarily increases
# ln L, since ln L(theta) >= B(theta | theta_k) with equality at theta_k.
library(this.path)

out_dir <- file.path(this.dir(), "figures")
dir.create(out_dir, showWarnings = FALSE)

# Observed-data log-likelihood (a smooth, slightly wavy concave curve)
lnL  <- function(t) -0.10 * (t - 5.5)^2 + 0.6 * sin(0.6 * t)
dlnL <- function(t) -0.20 * (t - 5.5) + 0.36 * cos(0.6 * t)

theta_k <- 2.5                       # current iterate
slope   <- dlnL(theta_k)             # B shares ln L's gradient at theta_k
curv    <- 0.7                       # downward curvature; > |lnL''| so B stays a lower bound
# Tangent quadratic minorizer: equal value and slope at theta_k, strictly below elsewhere
B <- function(t) lnL(theta_k) + slope * (t - theta_k) - 0.5 * curv * (t - theta_k)^2
theta_k1 <- theta_k + slope / curv   # argmax of B  ->  next EM iterate

grid <- seq(0, 9, length.out = 800)
stopifnot(max(B(grid) - lnL(grid)) < 1e-8)   # verify B is a genuine lower bound

png(file.path(out_dir, "em_concept.png"), width = 1500, height = 1050, res = 170)
par(mar = c(4, 4, 3, 1), family = "sans")
plot(grid, lnL(grid), type = "l", lwd = 3, col = "#1f4e79",
     xlab = expression(theta), ylab = "log-likelihood",
     main = expression("EM: maximizing the lower bound " * B(theta * "|" * theta[k]) *
                       " increases " * lnL(theta * "|" * Y)),
     ylim = range(c(lnL(grid), B(grid))) + c(-0.2, 0.4), bty = "l", xaxt = "n")
lines(grid, B(grid), lwd = 2.5, lty = 2, col = "#c0392b")

# Mark theta_k and theta_{k+1}: vertical guides + ascent on ln L
for (tt in c(theta_k, theta_k1))
  segments(tt, par("usr")[3], tt, lnL(tt), col = "gray60", lty = 3)
points(c(theta_k, theta_k1), c(lnL(theta_k), lnL(theta_k1)), pch = 19, col = "#1f4e79", cex = 1.3)
points(theta_k1, B(theta_k1), pch = 19, col = "#c0392b", cex = 1.3)
axis(1, at = c(theta_k, theta_k1),
     labels = c(expression(theta[k]), expression(theta[k + 1])))
text(theta_k1, B(theta_k1) - 0.35, "argmax B", col = "#c0392b", cex = 0.9)

legend("topright", bty = "n", lwd = c(3, 2.5), lty = c(1, 2),
       col = c("#1f4e79", "#c0392b"),
       legend = c(expression(lnL(theta * "|" * Y)), expression(B(theta * "|" * theta[k]))))
dev.off()
cat("Saved:", file.path(out_dir, "em_concept.png"), "\n")
