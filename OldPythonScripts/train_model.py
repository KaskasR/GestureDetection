# train_model.py (fixed)
import os, json, argparse, numpy as np
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class TinyMLP(nn.Module):
    def __init__(self, input_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):  # x: [B, T, D]
        b, T, D = x.shape
        x = x.reshape(b, T*D)
        return self.net(x)

def export_to_c_arrays(model, mean, std, labels, out_h_path):
    os.makedirs(os.path.dirname(out_h_path), exist_ok=True)
    with open(out_h_path, "w") as f:
        f.write("// Auto-generated weights for TinyMLP\n")
        f.write("#pragma once\n#include <stdint.h>\n\n")

        # Metadata
        f.write("static const int ANN_INPUT_T = 32;\n")
        f.write("static const int ANN_INPUT_D = 8;\n")
        f.write("static const int ANN_INPUT_DIM = 256;\n")
        f.write(f"static const int ANN_NUM_CLASSES = {len(labels)};\n\n")

        # Helper to dump a float array
        def dump_float_array(name, arr_np):
            flat = arr_np.reshape(-1).astype(np.float32)
            f.write(f"static const float {name}[{flat.size}] = {{" )
            f.write(",".join(f"{v:.8e}" for v in flat))
            f.write("};\n\n")

        # Normalization stats (mean/std are length D=8)
        dump_float_array("ann_input_mean", mean.astype(np.float32))
        dump_float_array("ann_input_std", np.where(std==0, 1.0, std).astype(np.float32))

        # Layer weights/biases (ensure CPU numpy)
        state = {k: v.detach().cpu().numpy().astype(np.float32) for k,v in model.state_dict().items()}

        # Shapes (optional but handy on firmware side)
        w1 = state["net.0.weight"]; b1 = state["net.0.bias"]
        w2 = state["net.2.weight"]; b2 = state["net.2.bias"]
        w3 = state["net.4.weight"]; b3 = state["net.4.bias"]

        f.write(f"static const int L1_IN = {w1.shape[1]}; static const int L1_OUT = {w1.shape[0]};\n")
        f.write(f"static const int L2_IN = {w2.shape[1]}; static const int L2_OUT = {w2.shape[0]};\n")
        f.write(f"static const int L3_IN = {w3.shape[1]}; static const int L3_OUT = {w3.shape[0]};\n\n")

        dump_float_array("l1_weight", w1)
        dump_float_array("l1_bias",   b1)
        dump_float_array("l2_weight", w2)
        dump_float_array("l2_bias",   b2)
        dump_float_array("l3_weight", w3)
        dump_float_array("l3_bias",   b3)

        # Labels
        f.write(f"static const char* ann_labels[{len(labels)}] = {{")
        f.write(",".join([f'\"{str(lbl)}\"' for lbl in labels]))
        f.write("};\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/dataset.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", default="training")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pack = np.load(args.data, allow_pickle=True)
    X = pack["X"].astype(np.float32)  # [N,T,D]
    y = pack["y"].astype(np.int64)    # [N]
    labels_arr = pack["labels"]
    labels = [str(x) for x in labels_arr.tolist()]
    N, T, D = X.shape
    assert (T, D) == (32, 8), f"Expected (32,8) got {(T,D)}"

    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mean = Xtr.mean(axis=(0,1), keepdims=True)  # [1,1,D]
    std  = Xtr.std(axis=(0,1), keepdims=True) + 1e-6
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std

    tr_ds = TensorDataset(torch.from_numpy(Xtr_n), torch.from_numpy(ytr))
    va_ds = TensorDataset(torch.from_numpy(Xva_n), torch.from_numpy(yva))
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch, shuffle=False)

    model = TinyMLP(input_dim=T*D, num_classes=len(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    best_path = os.path.join(args.outdir, "tinymlp_best.pt")

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_ds)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                pred = logits.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        val_acc = correct/total if total else 0.0
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({"model": model.state_dict(),
                        "labels": labels,
                        "mean": mean,
                        "std": std}, best_path)

    # Export
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    mean = ckpt["mean"].astype(np.float32).reshape(-1)  # length D
    std  = ckpt["std"].astype(np.float32).reshape(-1)

    os.makedirs("firmware", exist_ok=True)
    export_to_c_arrays(model, mean, std, labels, "firmware/ann_weights.h")
    with open("firmware/labels.json","w") as fp:
        json.dump(labels, fp)
    np.savez("firmware/norm_stats.npz", mean=mean, std=std)

    print(f"[DONE] Best val_acc={best_val:.3f}")
    print(" - firmware/ann_weights.h")
    print(" - firmware/labels.json")
    print(" - firmware/norm_stats.npz")
    print(" - training/tinymlp_best.pt")

if __name__ == "__main__":
    main()
