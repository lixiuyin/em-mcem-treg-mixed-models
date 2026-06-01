#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTKFace real data: extract deep embeddings from pretrained ResNet-50
-> PCA dimensionality reduction -> export CSV ready for R.
Used in Chapter 5 of the thesis for Student-t regression empirical study:
age ~ deep embeddings, heavy-tailed errors (t distribution).
Reproducible: fixed random seed; subsampling uses fixed seed.
"""
import os, re, sys, time
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from datasets import load_dataset

SEED = 20250529
N_SUB = 8000          # subsample size (CPU/MPS friendly; not needed for LMM; sufficient for t regression)
N_PCS = 30            # number of PCA components
BATCH = 64
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(SEED); torch.manual_seed(SEED)

def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEV = device()
print(f"[info] device = {DEV}", flush=True)

# ---- Load dataset (non-streaming, cached locally) ----
print("[info] loading UTKFace-Cropped from HF ...", flush=True)
ds = load_dataset("py97/UTKFace-Cropped", split="train")
n_total = len(ds)
print(f"[info] total images = {n_total}", flush=True)

# ---- Parse key -> age/gender/race, filter invalid filenames ----
key_re = re.compile(r"(\d+)_(\d)_(\d)_")
def parse_key(k):
    base = k.split("/")[-1]
    m = key_re.match(base)
    if not m:
        return None
    age, gender, race = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if age < 1 or age > 116:
        return None
    return age, gender, race

# Scan all metadata first, then do reproducible subsampling
meta = []
for i in range(n_total):
    k = ds[i]["__key__"]
    p = parse_key(k)
    if p is not None:
        meta.append((i, *p))
print(f"[info] valid labeled = {len(meta)}", flush=True)

rng = np.random.default_rng(SEED)
# Slight oversampling to compensate for occasional decode failures (None)
order = rng.permutation(len(meta))[:min(N_SUB + 400, len(meta))]
sel = sorted(order.tolist())
sel_meta = [meta[j] for j in sel]
sel_ds_idx = [m[0] for m in sel_meta]
print(f"[info] candidate subsample = {len(sel_ds_idx)}", flush=True)

# ---- Pretrained ResNet-50, remove classification head -> 2048-dim features ----
weights = models.ResNet50_Weights.IMAGENET1K_V2
net = models.resnet50(weights=weights)
net.fc = nn.Identity()
net.eval().to(DEV)
tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

feat_chunks=[]; keep_meta=[]
t0=time.time(); done=0
with torch.no_grad():
    buf=[]; bufmeta=[]
    def run_batch(buf):
        x=torch.stack(buf).to(DEV)
        return net(x).cpu().numpy()
    for di, m in zip(sel_ds_idx, sel_meta):
        if done >= N_SUB:
            break
        try:
            img = ds[di]["jpg.chip.jpg"]
            if img is None:
                continue
            img = img.convert("RGB")
        except Exception:
            continue
        buf.append(tf(img)); bufmeta.append(m)
        if len(buf)==BATCH:
            feat_chunks.append(run_batch(buf)); keep_meta.extend(bufmeta)
            done += len(buf); buf=[]; bufmeta=[]
            if done % (BATCH*10)==0:
                print(f"  [{done}/{N_SUB}]  {time.time()-t0:.0f}s", flush=True)
    if buf and done < N_SUB:
        feat_chunks.append(run_batch(buf)); keep_meta.extend(bufmeta); done += len(buf)
feats = np.concatenate(feat_chunks, axis=0)
ages   = np.array([m[1] for m in keep_meta], dtype=np.float64)
genders= np.array([m[2] for m in keep_meta], dtype=np.int64)
races  = np.array([m[3] for m in keep_meta], dtype=np.int64)
sel_ds_idx = [m[0] for m in keep_meta]
print(f"[info] embeddings done in {time.time()-t0:.0f}s, shape={feats.shape}  "
      f"age[min/med/max]={ages.min():.0f}/{np.median(ages):.0f}/{ages.max():.0f}", flush=True)

# ---- PCA (after standardization) ----
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(feats)
pca = PCA(n_components=N_PCS, random_state=SEED)
PCs = pca.fit_transform(Xs)
# Standardize each PC to unit variance to improve numerical conditioning on the R side
PCs = StandardScaler().fit_transform(PCs)
evr = pca.explained_variance_ratio_
print(f"[info] PCA {N_PCS} comps explain {evr.sum()*100:.1f}% var", flush=True)

# ---- Export ----
np.savez(os.path.join(OUT_DIR,"utkface_embeddings.npz"),
         feats=feats, age=ages, gender=genders, race=races,
         pcs=PCs, evr=evr, ds_idx=np.array(sel_ds_idx))
import pandas as pd
# Main modelling CSV contains only response (age) and predictors (PCA components of deep embeddings).
# Protected attributes gender/race are excluded from all modelling; retained in npz for data
# provenance only and never used in analysis.
df = pd.DataFrame(PCs, columns=[f"PC{i+1}" for i in range(N_PCS)])
df.insert(0,"age",ages)
csv_path=os.path.join(OUT_DIR,"utkface_t_regression.csv")
df.to_csv(csv_path, index=False)
print(f"[done] wrote {csv_path}  ({df.shape[0]} x {df.shape[1]})", flush=True)
