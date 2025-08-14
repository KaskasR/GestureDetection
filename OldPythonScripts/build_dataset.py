# build_dataset.py
import os, glob, argparse
import numpy as np

def load_clip(path):
    return np.load(path)  # [num_frames, feat_dim]

def to_fixed_length(seq, T=32):
    n, d = seq.shape if seq.ndim == 2 else (0, 0)
    if n == 0:
        return np.zeros((T, d), dtype=np.float32)
    if n == T:
        return seq.astype(np.float32)
    if n > T:
        # uniform downsample to T
        idx = np.linspace(0, n-1, T).astype(int)
        return seq[idx].astype(np.float32)
    # pad by repeating last frame
    out = np.zeros((T, seq.shape[1]), dtype=np.float32)
    out[:n] = seq
    out[n:] = seq[-1]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--featdir', default='training/features', help='Folder with .npy features')
    ap.add_argument('--out', default='training/dataset.npz', help='Output NPZ (X,y,labels)')
    ap.add_argument('--T', type=int, default=32, help='Frames per sample')
    args = ap.parse_args()

    # Infer labels from filenames: assume dataset/<label>/<file>.bin → features/<label_...>.npy
    # Simpler: read all .npy and decide label from prefix before first underscore
    files = sorted(glob.glob(os.path.join(args.featdir, '*.npy')))
    if not files:
        print('No .npy files found.'); return

    # Map label strings to integers
    def label_of(fp):
        base = os.path.basename(fp)
        # Expect something like wave_YYYY... or idle_YYYY...
        return base.split('_')[0]

    labels = sorted({label_of(f) for f in files})
    label_to_id = {l:i for i,l in enumerate(labels)}
    print('Labels:', labels)

    Xs, ys = [], []
    for f in files:
        seq = load_clip(f)
        if seq.ndim != 2 or seq.shape[0] == 0:
            print(f'[WARN] skip {f}: empty or bad shape {seq.shape}')
            continue
        x = to_fixed_length(seq, T=args.T)  # [T, feat_dim]
        y = label_to_id[label_of(f)]
        Xs.append(x)
        ys.append(y)

    if not Xs:
        print('No valid samples.'); return

    X = np.stack(Xs, axis=0).astype(np.float32)  # [N,T,D]
    y = np.array(ys, dtype=np.int64)
    np.savez_compressed(args.out, X=X, y=y, labels=np.array(labels))
    print(f'[DONE] Saved {args.out} with X={X.shape}, y={y.shape}, classes={labels}')

if __name__ == '__main__':
    main()
