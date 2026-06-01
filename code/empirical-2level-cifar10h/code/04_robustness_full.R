# -*- coding: utf-8 -*-
# Robustness check: re-run deterministic EM on the full annotator set (~2570 eligible,
# J=2567 after dropping rank-deficient annotators) and verify that fixed effects and ICC
# are consistent with the J=300 main-analysis subset.
suppressMessages({library(MASS); library(Matrix)})
ROOT <- dirname(dirname(this.path::this.dir()))   # derived from script location, points to code/
source(file.path(ROOT, "two-level-model/utils.R"), encoding = "UTF-8")
get_X_j <- function(data, j) data$X_list[[j]]
get_y_j <- function(data, j) data$y_list[[j]]
get_W_j <- function(data, j) data$W_list[[j]]
OUT <- file.path(ROOT, "empirical-2level-cifar10h/output")
full_csv <- file.path(ROOT, "empirical-2level-cifar10h/data/cifar10h_lmm_full.csv")
if (!file.exists(full_csv)) {
  stop("cifar10h_lmm_full.csv not found (it is git-ignored). Run 01_build_lmm_data.py first to regenerate it.")
}
df <- read.csv(full_csv)
# Drop rank-deficient annotators
gs <- sort(unique(df$group))
bad <- gs[vapply(gs, function(g){d<-df[df$group==g,]; qr(cbind(1,d$difficulty_z,d$correct,d$trial_z))$rank<4L}, logical(1))]
if(length(bad)) df <- df[!df$group %in% bad, ]
df$group <- as.integer(factor(df$group))
J <- length(unique(df$group)); p <- 3L; I4 <- diag(p+1)
cat(sprintf("Full dataset: J=%d N=%d (dropped %d rank-deficient annotators)\n", J, nrow(df), length(bad)))
Xl<-yl<-Wl<-vector("list",J)
for(j in 1:J){d<-df[df$group==j,]; Xl[[j]]<-cbind(1,d$difficulty_z,d$correct,d$trial_z); yl[[j]]<-d$logRT; Wl[[j]]<-I4}
data <- list(J=J,p=p,q=0L,sample_size=nrow(df),X_list=Xl,y_list=yl,W_list=Wl)
Xall<-cbind(1,df$difficulty_z,df$correct,df$trial_z); ols<-lm.fit(Xall,df$logRT)
t_em <- system.time(em <- run_single_em_or_mcem(data, matrix(ols$coefficients,ncol=1),
        diag(c(0.10,0.03,0.03,0.01)), as.numeric(var(ols$residuals)),
        is_mcem=FALSE, max_iter=300, tolerance=1e-6))["elapsed"]
D<-em$final_D_hat; icc<-D[1,1]/(D[1,1]+em$final_sigma2_hat)
cat(sprintf("[Full EM] elapsed %.0fs, iterations %d\n", t_em, em$final_iter_count))
cat("gamma (intercept/difficulty/correct/trial) =", round(as.numeric(em$final_gamma_hat),4),"\n")
cat(sprintf("sigma2=%.4f  random-effect SD=%s  intercept ICC=%.4f\n",
    em$final_sigma2_hat, paste(round(sqrt(diag(D)),4),collapse="/"), icc))
saveRDS(list(J=J,N=nrow(df),gamma=em$final_gamma_hat,D=D,sigma2=em$final_sigma2_hat,icc=icc,
        iter=em$final_iter_count), file.path(OUT,"lmm_full_robustness.rds"))
cat("[done] full-dataset robustness results saved\n")
