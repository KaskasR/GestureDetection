# log_uart.py
import os, sys, time, argparse, datetime
import serial

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def record_clip(port, baud, out_dir, label, seconds):
    ensure_dir(os.path.join(out_dir, label))
    fname = os.path.join(out_dir, label, f"{label}_{timestamp()}.bin")
    ser = serial.Serial(port=port, baudrate=baud, bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE, timeout=0.1)
    print(f"[INFO] Recording {seconds}s from {port} @ {baud} to {fname}")
    start = time.time()
    with open(fname, "wb") as f:
        total = 0
        while time.time() - start < seconds:
            data = ser.read(4096)
            if data:
                f.write(data)
                total += len(data)
    ser.close()
    print(f"[DONE] Wrote {total} bytes")
    return fname, total

def quick_probe(port, baud, seconds=2):
    print(f"[PROBE] Reading {seconds}s from {port} @ {baud} ...")
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
    start = time.time()
    total = 0
    while time.time() - start < seconds:
        data = ser.read(4096)
        total += len(data)
    ser.close()
    print(f"[PROBE] Received {total} bytes")
    if total < 1000:
        print("[WARN] Very little data. Make sure radar is running and this is the DATA COM port.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="DATA COM port (e.g., COM8 on Windows)")
    ap.add_argument("--baud", type=int, default=921600, help="DATA port baudrate (default 921600)")
    ap.add_argument("--out", default="dataset", help="Output base folder")
    ap.add_argument("--label", required=True, help="Gesture label (folder name)")
    ap.add_argument("--seconds", type=float, default=2.0, help="Seconds to record per clip")
    ap.add_argument("--probe", action="store_true", help="Quickly test if data is flowing")
    args = ap.parse_args()

    if args.probe:
        quick_probe(args.port, args.baud, seconds=2)
        sys.exit(0)

    record_clip(args.port, args.baud, args.out, args.label, args.seconds)
