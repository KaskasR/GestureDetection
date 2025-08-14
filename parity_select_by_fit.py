# parity_select_by_fit.py (robust)
# Brute-force 6-of-8 selection by fitting softmax to device probabilities, with data cleaning.
# pip install numpy pandas

import json, itertools, numpy as np, pandas as pd

FRAMES_CSV = "frames_raw.csv"
PROBS_CSV  = "probs_raw.csv"
WINDOW = 15
SEED = 42

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    s = ez.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return ez / s

def xent(P, Q, eps=1e-8):
    Q = np.clip(Q, eps, 1 - eps)
    return -np.mean(np.sum(P * np.log(Q), axis=1))

def finite_rows(a):  # a is 2D
    return np.all(np.isfinite(a), axis=1)

def train_softmax_regression(X, Y, Xv, Yv, lr=0.02, l2=1e-4, epochs=400, clip=5.0):
    N, D = X.shape
    K = Y.shape[1]
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.01, size=(D, K)).astype(np.float32)
    b = np.zeros((1, K), dtype=np.float32)

    best_loss = np.inf
    best_W, best_b = W.copy(), b.copy()
    bad = False

    for ep in range(epochs):
        Z = X @ W + b
        Q = softmax(Z)
        if not np.isfinite(Q).all():
            bad = True; break
        G = (Q - Y) / N
        dW = X.T @ G + l2 * W
        db = G.sum(axis=0, keepdims=True)
        # gradient clip
        gnorm = np.sqrt((dW**2).sum() + (db**2).sum())
        if gnorm > clip:
            dW *= clip/gnorm; db *= clip/gnorm
        W -= lr * dW; b -= lr * db

        # validate
        Qv = softmax(Xv @ W + b)
        if not np.isfinite(Qv).all():
            bad = True; break
        vloss = xent(Yv, Qv)
        if not np.isfinite(vloss):
            bad = True; break
        if vloss < best_loss:
            best_loss = vloss; best_W = W.copy(); best_b = b.copy()

    if bad or not np.isfinite(best_loss):
        return None, None, np.nan
    return best_W, best_b, float(best_loss)

def build_windows(frames_df):
    n_features = int(frames_df["n_features"].iloc[0])
    frames_df = frames_df.sort_values("frame").reset_index(drop=True)
    feats  = frames_df[[f"f{i}" for i in range(n_features)]].astype(np.float32).values
    frames = frames_df["frame"].values.astype(int)
    labels = frames_df["label"].values.astype(int)
    wins, win_frames, win_labels = [], [], []
    for i in range(len(frames) - WINDOW + 1):
        wins.append(feats[i:i+WINDOW, :].copy())
        win_frames.append(frames[i+WINDOW-1])
        win_labels.append(labels[i+WINDOW-1])
    return np.array(wins), np.array(win_frames), np.array(win_labels), n_features

def align_probs(win_frames, probs_df):
    prob_cols = [c for c in probs_df.columns if c.startswith("p")]
    K = len(prob_cols)
    P = probs_df[prob_cols].astype(np.float32).values
    F = probs_df["frame"].astype(int).values
    # Keep only rows with finite values and softmax-ish sum
    sums = P.sum(axis=1)
    ok = finite_rows(P) & np.isfinite(F) & (sums > 0.8) & (sums < 1.2) & (P.min(axis=1) >= -1e-4)
    F, P = F[ok], P[ok]
    pmap = {int(f): P[i] for i, f in enumerate(F)}
    idx = [i for i, f in enumerate(win_frames) if f in pmap]
    Y = np.stack([pmap[int(win_frames[i])] for i in idx], axis=0)
    return np.array(idx, dtype=int), Y, K

def main():
    frames_df = pd.read_csv(FRAMES_CSV)
    probs_df  = pd.read_csv(PROBS_CSV)

    windows, win_frames, _, n_features = build_windows(frames_df)
    sel_idx, Y, K = align_probs(win_frames, probs_df)
    windows = windows[sel_idx]

    # Drop any windows that contain non-finite features
    Wflat = windows.reshape(len(windows), -1)
    okw = finite_rows(Wflat)
    windows, Y = windows[okw], Y[okw]

    print(f"[INFO] Windows aligned & cleaned: {len(windows)}   n_features={n_features}   K={K}")
    if len(windows) < 200:
        print("[WARN] Very small dataset; results may be noisy.")

    # train/val split
    rng = np.random.default_rng(SEED)
    N = len(windows)
    perm = rng.permutation(N)
    Nv = max(100, int(0.2*N)) if N >= 500 else max(20, int(0.2*N))
    val_idx, tr_idx = perm[:Nv], perm[Nv:]
    print(f"[INFO] Split: train={len(tr_idx)}  val={len(val_idx)}")

    combos = list(itertools.combinations(range(n_features), 6))
    print(f"[INFO] Evaluating {len(combos)} subsets...")

    best = None
    skipped = 0
    for idx6 in combos:
        sel = np.array(idx6)
        # firmware-like normalization: per-feature (6) mean/std across all frames in train
        train6 = windows[tr_idx][:,:,sel].reshape(-1, 6)
        mu = train6.mean(axis=0); sd = train6.std(axis=0) + 1e-8

        def make_X(wins):
            X6 = (wins[:,:,sel] - mu.reshape(1,1,6)) / sd.reshape(1,1,6)
            return X6.reshape(len(wins), 15*6)

        Xtr = make_X(windows[tr_idx]); Xv = make_X(windows[val_idx])
        # Clean any rows that went non-finite (paranoia)
        oktr = finite_rows(Xtr); okv = finite_rows(Xv)
        Xtr, Ytr = Xtr[oktr], Y[tr_idx][oktr]
        Xv,  Yv  = Xv[okv],  Y[val_idx][okv]
        if len(Xtr) < 50 or len(Xv) < 10:
            skipped += 1; continue

        _, _, vloss = train_softmax_regression(Xtr, Ytr, Xv, Yv)
        if not np.isfinite(vloss):
            skipped += 1; continue
        if (best is None) or (vloss < best["loss"]):
            best = dict(idx6=tuple(idx6), loss=vloss, mu=mu.tolist(), sd=sd.tolist())

    if best is None:
        print(f"[FAIL] All subsets invalid (skipped={skipped}). Check CSVs for NaNs and try another short capture.")
        return

    print(f"[RESULT] Best subset: {best['idx6']}   val xent={best['loss']:.4f}   (skipped {skipped})")

    # Save 6-feature mean/std for firmware
    out = dict(
        selected_feature_indices=list(map(int, best["idx6"])),
        feature_mean=best["mu"],
        feature_std=best["sd"],
        K=int(K)
    )
    with open("feature_mapping.json","w") as f:
        json.dump(out, f, indent=2)
    print("[INFO] Wrote feature_mapping.json")

if __name__ == "__main__":
    main()
