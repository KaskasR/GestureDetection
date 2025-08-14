# emit_ti_includes_v2.py
# Robustly extract arrays from nn_export/neuralnet_params.h and write TI-style include fragments.
# Usage:
#   python emit_ti_includes_v2.py --src "nn_export/neuralnet_params.h" --dst "C:\path\to\include\neuralnet"
#
# It writes: mean.h, std.h, b0.h, b1.h, b2.h, w0.h, w1.h, w2.h

import argparse, pathlib, re

NAMES = [
    ("feature_mean", "mean.h"),
    ("feature_std",  "std.h"),
    ("b0", "b0.h"),
    ("b1", "b1.h"),
    ("b2", "b2.h"),
    ("w0", "w0.h"),
    ("w1", "w1.h"),
    ("w2", "w2.h"),
]

def read_text(p):
    return pathlib.Path(p).read_text(encoding="utf-8", errors="ignore")

def find_array_block(text, name):
    """
    Find the initializer block { ... } (handles 1-D or 2-D) for `name`.
    Accepts lines like:
      static const float name[NN_DIM] = { ... };
      static const float name[OUT][IN] = { { ... }, { ... }, ... };
    """
    # Make a forgiving pattern: name [optional brackets] = { ... };
    # Use a non-greedy match for the braces, DOTALL enabled.
    pat = rf"{name}\s*(?:\[[^\]]*\]\s*)*(?:=\s*)\{{(.*?)\}};"
    m = re.search(pat, text, flags=re.S)
    if not m:
        return None
    block = m.group(1)
    # Remove nested braces for 2-D cases and compress whitespace
    block = re.sub(r"[{}]", " ", block)
    block = re.sub(r"\s+", " ", block)
    return block.strip()

def to_float_list(block):
    vals = []
    for tok in block.split(","):
        t = tok.strip()
        if not t:
            continue
        # Strip any trailing 'f'/'F'
        if t[-1:] in ("f","F"):
            t = t[:-1]
        try:
            vals.append(float(t))
        except:
            # ignore tokens that aren't numeric
            pass
    return vals

def wrap_floats(vals, per_line=8):
    out = []
    for i, v in enumerate(vals):
        s = f"{v:.8f}f"
        out.append(s)
        if (i+1) % per_line == 0:
            out[-1] += "\n"
    return ", ".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to nn_export/neuralnet_params.h")
    ap.add_argument("--dst", required=True, help="Destination TI include folder (…\\include\\neuralnet)")
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    if not src.exists():
        raise SystemExit(f"[ERROR] Source header not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    text = read_text(src)
    # Quick sanity: show what names exist in the file
    present = [name for name,_ in NAMES if re.search(rf"\b{name}\b", text)]
    print("[INFO] Arrays mentioned in header:", ", ".join(present) or "(none)")

    for name, outfn in NAMES:
        block = find_array_block(text, name)
        if block is None:
            print(f"[WARN] Could not find initializer for {name}; skipping.")
            continue
        vals = to_float_list(block)
        if not vals:
            print(f"[WARN] No numeric values parsed for {name}; skipping.")
            continue
        out_text = wrap_floats(vals)
        (dst / outfn).write_text(out_text, encoding="utf-8")
        print(f"[OK] Wrote {dst/outfn}  ({len(vals)} floats)")

    print("[DONE] If any files were skipped, open the header and confirm the array names match the expected ones above.")
    print("       If names differ, tell me what you see and I’ll adjust the emitter pattern.")

if __name__ == "__main__":
    main()
