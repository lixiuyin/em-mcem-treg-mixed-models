#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10H real data -> two-level linear model (LMM) dataset.
Level-1: logRT_ij = b0_j + b1_j*difficulty_i + b2_j*correct_ij + b3_j*trial_ij + eps
Level-2: beta_j ~ N(gamma, D)   (annotator random coefficients)
difficulty = entropy of the human soft-label distribution for each image
             (human uncertainty on the deep-learning benchmark CIFAR-10).
"""
import os, numpy as np, pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "data")
# Read the raw trials directly from the committed zip (single-member archive);
# pandas infers the compression from the .zip extension, so no manual unzip is needed.
raw = pd.read_csv(os.path.join(D, "cifar10h-raw.zip"))
print("raw rows:", len(raw), "annotators:", raw.annotator_id.nunique())

# 1) Per-image difficulty: entropy of human soft labels (probs: 10000 x 10)
probs = np.load(os.path.join(D, "cifar10h-probs.npy"))
p = np.clip(probs, 1e-12, 1.0)
ent = -(p * np.log(p)).sum(axis=1)          # nats, 0..ln(10)
print(f"image entropy: min={ent.min():.3f} med={np.median(ent):.3f} max={ent.max():.3f} (ln10={np.log(10):.3f})")

# 2) Clean trial-level data
df = raw[raw.is_attn_check == 0].copy()      # remove attention-check trials
df = df[(df.reaction_time > 200) & (df.reaction_time < 20000)]  # plausible RT window (ms)
df["logRT"]      = np.log(df.reaction_time.astype(float))
df["difficulty"] = ent[df.cifar10_test_test_idx.values]
df["correct"]    = df.correct_guess.astype(float)
df["trial"]      = df.trial_index.astype(float)
print("after clean rows:", len(df))

# 3) Standardize continuous covariates (aids numerical stability and interpretation)
for c in ["difficulty", "trial"]:
    df[c + "_z"] = (df[c] - df[c].mean()) / df[c].std()

# 4) Trial count distribution per annotator
sz = df.groupby("annotator_id").size()
print(f"trials/annotator: min={sz.min()} med={int(sz.median())} max={sz.max()} mean={sz.mean():.0f}")

# 5) Main-analysis subset: keep annotators with sufficient trials, sample J (reproducible)
MIN_TRIALS = 150
SEED = 20250529
J_MAIN = 300
elig = sz[sz >= MIN_TRIALS].index.to_numpy()
print(f"annotators with >= {MIN_TRIALS} trials: {len(elig)}")
rng = np.random.default_rng(SEED)
sel = np.sort(rng.choice(elig, size=min(J_MAIN, len(elig)), replace=False))
main = df[df.annotator_id.isin(sel)].copy()
# Re-index groups 1..J
remap = {a: i + 1 for i, a in enumerate(sorted(main.annotator_id.unique()))}
main["group"] = main.annotator_id.map(remap)
main = main.sort_values(["group", "trial"]).reset_index(drop=True)
cols = ["group", "annotator_id", "logRT", "difficulty_z", "correct", "trial_z",
        "reaction_time", "difficulty", "trial", "cifar10_test_test_idx", "correct_guess"]
out_main = os.path.join(D, "cifar10h_lmm_main.csv")
main[cols].to_csv(out_main, index=False)
gs = main.groupby("group").size()
print(f"[main] J={main.group.nunique()} N={len(main)} n_j[min/med/max]={gs.min()}/{int(gs.median())}/{gs.max()}")
print(f"[done] wrote {out_main}")

# 6) Also export full dataset (for scalability/robustness checks): all >=MIN_TRIALS annotators
full = df[df.annotator_id.isin(elig)].copy()
remapF = {a: i + 1 for i, a in enumerate(sorted(full.annotator_id.unique()))}
full["group"] = full.annotator_id.map(remapF)
full = full.sort_values(["group", "trial"]).reset_index(drop=True)
out_full = os.path.join(D, "cifar10h_lmm_full.csv")
full[cols].to_csv(out_full, index=False)
print(f"[full] J={full.group.nunique()} N={len(full)} -> {out_full}")

# 7) Naive correlation sanity check (expected: higher difficulty -> slower; correct -> faster; later trial -> faster)
import numpy as np
print("\n[sanity] pooled OLS logRT ~ difficulty_z + correct + trial_z")
X = np.column_stack([np.ones(len(main)), main.difficulty_z, main.correct, main.trial_z])
b, *_ = np.linalg.lstsq(X, main.logRT.values, rcond=None)
print("  intercept=%.4f  difficulty=%.4f  correct=%.4f  trial=%.4f" % tuple(b))
