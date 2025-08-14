# parity_check.py — auto-detect K from probs_raw.csv and find the 6-of-8 feature indices.
# pip install numpy pandas

import os, re, json, itertools
import numpy as np
import pandas as pd

HEADERS_DIR = r"C:\Users\zadar\ccs_workspace_gestures\gesture_ML_6443_AOP\include\neuralnet"
FRAMES_CSV  = "frames_raw.csv"
PROBS_CSV   = "probs_raw.csv"

WINDOW = 15
SHAPE_W0 = (30, 90)
SHAPE_W1 = (60, 30)

def parse_float_array(path):
    txt = open(path, 'r', encoding='utf-8', errors='ignore').read()
    m = re.search(r'\{([^}]*)\}', txt, flags=re.S)
    if not m: return None
    vals = []
    for tok in m.group(1).replace('\n',' ').split(','):
        t = tok.strip().rstrip('fF')
        if t:
            try: vals.append(float(t))
            except: pass
    return np.array(vals, dtype=np.float32)

def load_headers(headers_dir, K):
    pools = []
    for fn in os.listdir(headers_dir):
        if fn.endswith(".h"):
            arr = parse_float_array(os.path.join(headers_dir, fn))
            if arr is not None and arr.size > 0:
                pools.append((fn, arr))

    # Helpers: pop by exact size
    def pop_by_size(n, name):
        for i,(fn,arr) in enumerate(pools):
            if arr.size == n:
                return pools.pop(i)
        raise RuntimeError(f"Missing header array of size {n} for {name}")

    # Fixed layers
    fn_w0, w0 = pop_by_size(SHAPE_W0[0]*SHAPE_W0[1], "W0")
    fn_b0, b0 = pop_by_size(SHAPE_W0[0], "b0")
    fn_w1, w1 = pop_by_size(SHAPE_W1[0]*SHAPE_W1[1], "W1")
    fn_b1, b1 = pop_by_size(SHAPE_W1[0], "b1")
    # Output layer (K x 60) and bias (K)
    fn_w2, w2 = pop_by_size(K*SHAPE_W1[0], "W2")
    fn_b2, b2 = pop_by_size(K, "b2")

    # Means/stds: pick two 6-length arrays; std must be positive
    sixes = [(fn,arr) for (fn,arr) in pools if arr.size == 6]
    if len(sixes) < 2:
        raise RuntimeError("Need two 6-float arrays for mean/std.")
    # Choose std as the 6-array with all positive entries and not all ones
    (fn_m, mean), (fn_s, std) = sixes[0], sixes[1]
    if not np.all(std > 0):  # swap if needed
        mean, std = std, mean
        fn_m, fn_s = fn_s, fn_m

    W0 = w0.reshape(SHAPE_W0)   # row-major W[out,in]
    W1 = w1.reshape(SHAPE_W1)
    W2 = w2.reshape((K, SHAPE_W1[0]))

    return dict(W0=W0,b0=b0,W1=W1,b1=b1,W2=W2,b2=b2,mean=mean,std=std,
                meta=dict(w0=fn_w0,b0=fn_b0,w1=fn_w1,b1=fn_b1,w2=fn_w2,b2=fn_b2,mean=fn_m,std=fn_s))

def relu(x): return np.maximum(x, 0.0)
def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)

def forward(W0,b0,W1,b1,W2,b2,x):
    y = relu(x @ W0.T + b0)
    y = relu(y @ W1.T + b1)
    z = y @ W2.T + b2
    return softmax(z)

def build_windows(frames_df):
    n_features = int(frames_df['n_features'].iloc[0])
    frames_df = frames_df.sort_values('frame').reset_index(drop=True)
    feats  = frames_df[[f"f{i}" for i in range(n_features)]].astype(np.float32).values
    frames = frames_df['frame'].values.astype(int)
    labels = frames_df['label'].values.astype(int)
    wins, win_frames, win_labels = [], [], []
    for i in range(len(frames) - WINDOW + 1):
        wins.append(feats[i:i+WINDOW, :].copy())
        win_frames.append(frames[i+WINDOW-1])
        win_labels.append(labels[i+WINDOW-1])
    return np.array(wins), np.array(win_frames), np.array(win_labels), n_features

def load_probs(csv_path):
    P = pd.read_csv(csv_path)
    prob_cols = [c for c in P.columns if c.startswith('p')]
    F = P['frame'].values.astype(int)
    return F, P[prob_cols].astype(np.float32).values, len(prob_cols)

def main():
    # Infer K from probs file
    F_prob, P_dev, K = load_probs(PROBS_CSV)
    print(f"[INFO] Detected K={K} outputs from probs_raw.csv")

    nets = load_headers(HEADERS_DIR, K)
    W0,b0,W1,b1,W2,b2 = nets['W0'],nets['b0'],nets['W1'],nets['b1'],nets['W2'],nets['b2']
    mean,std = nets['mean'],nets['std']
    print("[INFO] Headers:", nets['meta'])
    print(f"[INFO] mean={mean}, std={std}")

    frames_df = pd.read_csv(FRAMES_CSV)
    windows, win_frames, win_labels, n_features = build_windows(frames_df)
    print(f"[INFO] Built {len(windows)} windows; n_features={n_features}")

    # Align windows to device probs by frame number
    prob_map = {int(f): P_dev[i] for i,f in enumerate(F_prob)}
    idx = [i for i,f in enumerate(win_frames) if f in prob_map]
    if not idx:
        raise RuntimeError("No overlapping frames. Capture a short sequence with the new script.")
    windows = windows[idx]
    win_frames = win_frames[idx]
    Y_dev = np.stack([prob_map[int(f)] for f in win_frames], axis=0)
    print(f"[INFO] Aligned {len(windows)} windows to device probabilities.")

    # Try all 8-choose-6 subsets
    if n_features < 6:
        raise RuntimeError("Need at least 6 features.")
    combos = list(itertools.combinations(range(n_features), 6))
    print(f"[INFO] Evaluating {len(combos)} feature subsets...")

    def prep_x(wins, idx6):
        sel = wins[:, :, idx6]                  # (N,15,6)
        sel = (sel - mean.reshape(1,1,6)) / std.reshape(1,1,6)
        return sel.reshape(sel.shape[0], 15*6)  # (N,90)

    best = None
    for idx6 in combos:
        x = prep_x(windows, idx6)
        Y_hat = forward(W0,b0,W1,b1,W2,b2,x)
        err = np.mean(np.abs(Y_hat - Y_dev))
        if best is None or err < best['err']:
            best = dict(idx6=idx6, err=err)

    print(f"[RESULT] Best 6-of-{n_features} feature indices: {best['idx6']}   mean |Δ| = {best['err']:.6f}")
    with open("feature_mapping.json","w") as f:
        json.dump(dict(selected_feature_indices=list(map(int,best['idx6'])),
                       mean=list(map(float,mean)), std=list(map(float,std)), K=int(K)), f, indent=2)
    print("[INFO] Wrote feature_mapping.json")

if __name__ == "__main__":
    main()
