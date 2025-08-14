#!/usr/bin/env python3
"""
Export a Keras 3-layer MLP to TI mmWave gesture demo headers.

Outputs eight files in --outdir (usually <ccs project>/include/neuralnet):
  mean.h, std.h, b0.h, w0.h, b1.h, w1.h, b2.h, w2.h

Expected shapes (from gesture.h):
  INP_DIM = 90  (15 frames * 6 features)
  L1_OUT  = 30
  L2_OUT  = 60
  L3_OUT  = 10

Keras Dense kernels are (in, out). TI wants W_n as [out][in] with each row being one neuron.
We therefore transpose kernels when writing.

Usage examples:
  python export_ti_gesture_ann.py --model model.h5 --outdir ./include/neuralnet \
         --mean 0,0,0,0,0,0 --std 1,1,1,1,1,1

  python export_ti_gesture_ann.py --model model.keras --outdir ./include/neuralnet \
         --stats-csv train_feats.csv --csv-cols 0,1,2,3,4,5
"""
import argparse, os, sys, textwrap
import numpy as np

# Optional TF import only when loading the model
def load_keras_model(path):
    try:
        import tensorflow as tf
    except Exception as e:
        print("ERROR: TensorFlow is required to load the Keras model.\n"
              "Install with: pip install tensorflow", file=sys.stderr)
        raise
    return tf.keras.models.load_model(path, compile=False)

# --- TI shapes from gesture.h (defaults; override with CLI if you ever change gesture.h)
DEFAULT_INP_DIM = 90
DEFAULT_L1_OUT  = 30
DEFAULT_L2_OUT  = 60
DEFAULT_L3_OUT  = 10

def fmt_float(x: float) -> str:
    # Compact, C-friendly scientific notation; no 'f' suffix (TI headers use plain literals)
    if not np.isfinite(x):
        # replace NaN/Inf defensively
        x = 0.0
    s = f"{float(x):.8e}"
    # Clean up exponent formatting a bit
    s = s.replace("e+0", "e+").replace("e-0", "e-")
    return s

def write_list_1d(path, arr):
    # Writes "a, b, c" — no braces; ann_params.h wraps include in braces already
    with open(path, "w", newline="\n") as f:
        f.write(", ".join(fmt_float(x) for x in arr) + "\n")

def write_matrix_rows(path, mat_out_in):
    """
    Writes:
      { r0c0, r0c1, ... r0cN },
      { r1c0, r1c1, ... r1cN },
      ...
    No outer braces; ann_params.h has a single block-level brace.
    """
    with open(path, "w", newline="\n") as f:
        rows = []
        for r in mat_out_in:
            row = "{ " + ", ".join(fmt_float(x) for x in r) + " }"
            rows.append(row)
        f.write(",\n".join(rows) + "\n")

def get_dense_layers_keras(model):
    # Collect Dense layers in order of appearance; ignore activations/normalizations
    try:
        import tensorflow as tf
    except:
        raise
    dense_layers = [ly for ly in model.layers if isinstance(ly, tf.keras.layers.Dense)]
    if len(dense_layers) < 3:
        raise ValueError(f"Model must have at least 3 Dense layers; found {len(dense_layers)}")
    return dense_layers[:3]

def parse_csv_cols(arg):
    # "0,1,2,3,4,5" or "f0,f1,f2,f3,f4,f5"
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    # try integer cols first
    idx = []
    for p in parts:
        try:
            idx.append(int(p))
        except ValueError:
            idx.append(p)  # keep as string (header name)
    return idx

def compute_mean_std_from_csv(csv_path, cols):
    import csv
    xs = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f) if not all(isinstance(c,int) for c in cols) else None
        if reader:
            for row in reader:
                xs.append([float(row[str(c)]) for c in cols])
        else:
            # No headers: load raw and pick columns by index
            arr = np.loadtxt(csv_path, delimiter=",")
            xs = arr[:, cols].tolist()
    X = np.asarray(xs, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 6:
        raise ValueError(f"Expected 6 feature columns for mean/std, got shape {X.shape}")
    mean = np.mean(X, axis=0)
    std  = np.std(X, axis=0, ddof=0)
    # Prevent divide-by-zero in runtime
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Export Keras MLP to TI gesture demo headers",
        epilog=textwrap.dedent("""
        Notes:
          * Keras Dense kernels are (in, out). TI headers want [out][in].
          * mean/std must be exactly 6 values (per-feature), not 90.
          * The output files are *fragments* (no type names or braces) included by ann_params.h.
        """)
    )
    ap.add_argument("--model", required=True, help="Path to Keras model (.h5/.keras)")
    ap.add_argument("--outdir", required=True, help="Output dir for headers (…/include/neuralnet)")
    ap.add_argument("--inp-dim", type=int, default=DEFAULT_INP_DIM, help="INP_DIM (default 90)")
    ap.add_argument("--l1-out", type=int, default=DEFAULT_L1_OUT,  help="L1 out size (default 30)")
    ap.add_argument("--l2-out", type=int, default=DEFAULT_L2_OUT,  help="L2 out size (default 60)")
    ap.add_argument("--l3-out", type=int, default=DEFAULT_L3_OUT,  help="L3 out size (default 10)")

    # mean/std options (choose one)
    ap.add_argument("--mean", type=str, help="Comma-separated 6 floats for mean")
    ap.add_argument("--std",  type=str, help="Comma-separated 6 floats for std")
    ap.add_argument("--stats-csv", type=str, help="CSV file to compute mean/std (6 feature cols)")
    ap.add_argument("--csv-cols", type=str, help="Column indices or names for the 6 features, e.g. 0,1,2,3,4,5")

    args = ap.parse_args()

    INP_DIM = args.inp_dim
    L1_OUT  = args.l1_out
    L2_OUT  = args.l2_out
    L3_OUT  = args.l3_out

    os.makedirs(args.outdir, exist_ok=True)

    # Load model and pull first 3 Dense layers
    model = load_keras_model(args.model)
    d0, d1, d2 = get_dense_layers_keras(model)

    # Extract weights (kernel, bias) and sanity-check shapes
    def k_b(layer):
        W, b = layer.get_weights()  # W: (in, out)
        return np.asarray(W, np.float32), np.asarray(b, np.float32)

    W0_in_out, b0 = k_b(d0)
    W1_in_out, b1 = k_b(d1)
    W2_in_out, b2 = k_b(d2)

    # Validate shapes
    if W0_in_out.shape[0] != INP_DIM or W0_in_out.shape[1] != L1_OUT:
        raise ValueError(f"L1 kernel shape mismatch: got {W0_in_out.shape}, expected ({INP_DIM}, {L1_OUT})")
    if b0.shape[0] != L1_OUT:
        raise ValueError(f"L1 bias shape mismatch: got {b0.shape}, expected ({L1_OUT},)")

    if W1_in_out.shape[0] != L1_OUT or W1_in_out.shape[1] != L2_OUT:
        raise ValueError(f"L2 kernel shape mismatch: got {W1_in_out.shape}, expected ({L1_OUT}, {L2_OUT})")
    if b1.shape[0] != L2_OUT:
        raise ValueError(f"L2 bias shape mismatch: got {b1.shape}, expected ({L2_OUT},)")

    if W2_in_out.shape[0] != L2_OUT or W2_in_out.shape[1] != L3_OUT:
        raise ValueError(f"L3 kernel shape mismatch: got {W2_in_out.shape}, expected ({L2_OUT}, {L3_OUT})")
    if b2.shape[0] != L3_OUT:
        raise ValueError(f"L3 bias shape mismatch: got {b2.shape}, expected ({L3_OUT},)")

    # Transpose to [out][in] rows for TI
    W0_out_in = W0_in_out.T.copy(order="C")  # (L1_OUT, INP_DIM)
    W1_out_in = W1_in_out.T.copy(order="C")  # (L2_OUT, L1_OUT)
    W2_out_in = W2_in_out.T.copy(order="C")  # (L3_OUT, L2_OUT)

    # mean/std (6 floats)
    if args.stats_csv:
        if not args.csv_cols:
            print("ERROR: --stats-csv requires --csv-cols with 6 columns", file=sys.stderr)
            sys.exit(2)
        cols = parse_csv_cols(args.csv_cols)
        if len(cols) != 6:
            print("ERROR: --csv-cols must specify exactly 6 columns", file=sys.stderr)
            sys.exit(2)
        mean, std = compute_mean_std_from_csv(args.stats_csv, cols)
    else:
        if not args.mean or not args.std:
            print("WARNING: mean/std not provided; defaulting to zeros/ones (OK only if you normalized elsewhere).",
                  file=sys.stderr)
            mean = np.zeros((6,), dtype=np.float32)
            std  = np.ones((6,), dtype=np.float32)
        else:
            try:
                mean = np.array([float(x) for x in args.mean.split(",")], dtype=np.float32)
                std  = np.array([float(x) for x in args.std.split(",")], dtype=np.float32)
            except Exception as e:
                print("ERROR parsing --mean/--std: ", e, file=sys.stderr)
                sys.exit(2)
            if mean.shape[0] != 6 or std.shape[0] != 6:
                print("ERROR: --mean and --std must each have exactly 6 values", file=sys.stderr)
                sys.exit(2)
            std[std == 0] = 1.0

    # Write headers
    def p(name): return os.path.join(args.outdir, name)

    write_list_1d(p("mean.h"), mean)
    write_list_1d(p("std.h"),  std)

    write_list_1d(p("b0.h"), b0)
    write_matrix_rows(p("w0.h"), W0_out_in)

    write_list_1d(p("b1.h"), b1)
    write_matrix_rows(p("w1.h"), W1_out_in)

    write_list_1d(p("b2.h"), b2)
    write_matrix_rows(p("w2.h"), W2_out_in)

    print("Wrote headers to:", os.path.abspath(args.outdir))
    for fn in ["mean.h","std.h","b0.h","w0.h","b1.h","w1.h","b2.h","w2.h"]:
        print("  -", fn)

if __name__ == "__main__":
    main()
