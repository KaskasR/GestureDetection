# scan_headers.py
# Usage: python scan_headers.py "C:\path\to\project\root"
import sys, os, re

NEEDED_SIZES = {2700:"W0 (30x90)", 30:"b0 (30)",
                1800:"W1 (60x30)", 60:"b1 (60)",
                720:"W2 (Kx60) for K=12", 12:"b2 (12)",
                6:"mean/std (6)"}

def extract_all_arrays(text):
    # Find ALL { ... } float lists in a file (not just the first)
    out = []
    for m in re.finditer(r'\{([^}]*)\}', text, flags=re.S):
        body = m.group(1)
        vals = []
        for tok in body.replace('\n',' ').split(','):
            t = tok.strip().rstrip('fF')
            if not t: continue
            try:
                vals.append(float(t))
            except:
                pass
        if vals:
            out.append(len(vals))
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_headers.py <project_root>")
        sys.exit(1)
    root = sys.argv[1]
    found = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(('.h', '.c')): continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except:
                continue
            sizes = extract_all_arrays(text)
            if sizes:
                found.setdefault(dirpath, []).extend(sizes)

    # Summarize by directory
    print(f"[INFO] Scanned under: {root}\n")
    best_dirs = []
    for d, sizes in found.items():
        score = sum(1 for s in set(sizes) if s in NEEDED_SIZES)
        if score:
            best_dirs.append((score, d, sizes))
    best_dirs.sort(reverse=True)

    for score, d, sizes in best_dirs[:30]:
        counts = {}
        for s in sizes:
            counts[s] = counts.get(s, 0) + 1
        tagged = ", ".join(f"{s}x ({NEEDED_SIZES.get(s,'?')})" for s in sorted(counts.keys()))
        print(f"{d}\n  -> has: {tagged}\n")

    if not best_dirs:
        print("No candidate directories found. Double-check the project root.")

if __name__ == "__main__":
    main()
