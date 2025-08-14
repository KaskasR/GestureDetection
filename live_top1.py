# live_top1.py — NoGesture-first live monitor with per-frame confidences
# Prints confidences for [NoGesture, Pinch, Fist, Wave] every frame.
# Works with 4/8/12-length outputs (softmax or one-hot). Auto-calibrates which index is idle.
# Requires: pip install pyserial

import serial, struct, time, math
from collections import deque

# ---------------- user settings ----------------
DATA_PORT  = 'COM8'
DATA_BAUD  = 921600

# Decisioning knobs (NoGesture-first)
SMOOTH_FRAMES = 7
ACCEPT_N       = 4
FIRE_THRESH    = 0.65
MARGIN_OVER_NO = 0.15
COOLDOWN_SEC   = 0.35
CAL_IDLE_SEC   = 2.0   # hold still at start

# Optional: map device indices -> [No,Pinch,Fist,Wave]
# Example if your wire is 8-wide: MAP8_TO_4 = {3:0, 5:1, 6:2, 7:3}
MAP8_TO_4 = None

# Optional: ignore classes (by our indices 0..3)
IGNORED_CLASSES = set()

# Optional: activity gate
USE_ACTIVITY_GATE = True
ACTIVITY_THRESH   = 0.35

# ---- printing controls ----
# 'smoothed' (default), 'raw', or 'both'
PRINT_CONFIDENCES = 'smoothed'
DECIMALS = 2
HEADER_EVERY = 25  # repeat header every N lines for readability
# ------------------------------------------------

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
FRAME_HDR_LEN = 40

LABELS = ["NoGesture", "Pinch", "Fist", "Wave"]

def open_port(name, baud):
    return serial.Serial(name, baudrate=baud, bytesize=8, parity='N', stopbits=1,
                         timeout=0.05, write_timeout=0.1)

def find_magic(buf): return buf.find(MAGIC)

def parse_frame_header(buf):
    # <Q 8I> = magic, version, total_len, platform, frame, ts, num_obj, num_tlvs, subframe
    _, ver, total_len, plat, frame, ts, num_obj, num_tlvs, sub = struct.unpack_from('<Q8I', buf, 0)
    return dict(total_len=total_len, frame=frame)

def find_prob_vector(pkt):
    """Scan payload for a float block that looks like probabilities."""
    start = FRAME_HDR_LEN
    end   = len(pkt)
    best = None
    for off in range(start, max(start, end - 4*4 + 1), 4):
        maxN = min(16, (end - off)//4)
        for n in range(4, maxN+1):
            try:
                vals = struct.unpack_from('<' + 'f'*n, pkt, off)
            except struct.error:
                break
            if any(math.isnan(v) or math.isinf(v) or v < -1e-3 for v in vals):
                continue
            s = sum(vals)
            if 0.9 <= s <= 1.1 and all(v <= 1.2 for v in vals):
                score = abs(s - 1.0)
                if best is None or score < best[0]:
                    best = (score, list(vals))
    return (best[1] if best else None)

def fmt_probs(p):
    return " ".join(f"{x:.{DECIMALS}f}" for x in p)

def maybe_print_header(line_no):
    if line_no % HEADER_EVERY == 0:
        print(f"{'frame':>8} | {'No':>6} {'Pinch':>6} {'Fist':>6} {'Wave':>6}"
              + (" || " + "raw".center(28) if PRINT_CONFIDENCES == 'both' else ""))

def main():
    try:
        data = open_port(DATA_PORT, DATA_BAUD)
    except Exception as e:
        print(f"[ERROR] Opening DATA port {DATA_PORT}@{DATA_BAUD}: {e}")
        return

    buf = bytearray()
    prob_hist = deque(maxlen=SMOOTH_FRAMES)

    seen_k = None
    cal_start = None
    cal_samples = []
    NO_IDX = 0
    last_cand = None
    stable_count = 0
    last_fire = 0.0
    line_no = 0

    print(f"[INFO] Live monitor on {DATA_PORT}@{DATA_BAUD}. Ctrl+C to quit.")
    print(f"[INFO] Hold still for ~{CAL_IDLE_SEC:.1f}s at start for idle calibration…")

    try:
        while True:
            chunk = data.read(4096)
            if chunk:
                buf.extend(chunk)

            # frame sync loop
            while True:
                idx = find_magic(buf)
                if idx < 0:
                    if len(buf) > 65536: del buf[:-7]
                    break
                if idx > 0: del buf[:idx]
                if len(buf) < FRAME_HDR_LEN: break

                hdr = parse_frame_header(buf)
                total_len = hdr['total_len']
                if total_len < FRAME_HDR_LEN or total_len > 128*1024:
                    del buf[:1]; continue
                if len(buf) < total_len: break

                pkt = bytes(buf[:total_len]); del buf[:total_len]

                probs = find_prob_vector(pkt)
                if probs is None: continue

                k = len(probs)
                if seen_k is None:
                    seen_k = k
                    print(f"[INFO] Detected {k} outputs on wire.")

                # Collapse to 4-way [No, Pinch, Fist, Wave]
                if MAP8_TO_4 and k >= max(MAP8_TO_4.keys())+1:
                    p4 = [0.0, 0.0, 0.0, 0.0]
                    for src, dst in MAP8_TO_4.items():
                        p4[dst] += probs[src]
                    probs = p4
                elif k > 4:
                    probs = probs[:4]
                elif k < 4:
                    probs = probs + [0.0]*(4-k)

                # Calibration for idle (NoGesture) index
                now = time.perf_counter()
                if cal_start is None:
                    cal_start = now
                if (now - cal_start) <= CAL_IDLE_SEC:
                    cal_samples.append(probs[:4])
                    continue
                elif cal_samples:
                    avg = [sum(col)/len(cal_samples) for col in zip(*cal_samples)]
                    NO_IDX = max(range(4), key=lambda i: avg[i])
                    print(f"[INFO] Calibrated NoGesture index = {NO_IDX}  avg={['%.2f'%v for v in avg]}")
                    cal_samples.clear()

                # Ignore some classes if asked
                allowed = [i for i in range(4) if i not in IGNORED_CLASSES]
                for i in IGNORED_CLASSES:
                    probs[i] = 0.0
                s = sum(probs[i] for i in allowed)
                if s > 0:
                    probs = [(probs[i]/s if i in allowed else 0.0) for i in range(4)]

                # Smoothing
                prob_hist.append(probs[:4])
                sm = [sum(col)/len(prob_hist) for col in zip(*prob_hist)]

                # ----- printing confidences -----
                line_no += 1
                maybe_print_header(line_no)
                if PRINT_CONFIDENCES == 'raw':
                    print(f"{hdr['frame']:8d} | {fmt_probs(probs)}")
                elif PRINT_CONFIDENCES == 'both':
                    print(f"{hdr['frame']:8d} | {fmt_probs(sm)} || {fmt_probs(probs)}")
                else:  # 'smoothed'
                    print(f"{hdr['frame']:8d} | {fmt_probs(sm)}")

                # NoGesture-first decision
                gesture_indices = [i for i in allowed if i != NO_IDX]
                if not gesture_indices: continue
                cand = max(gesture_indices, key=lambda i: sm[i])
                p_cand = sm[cand]; p_no = sm[NO_IDX]

                if cand == last_cand:
                    stable_count += 1
                else:
                    last_cand = cand; stable_count = 1

                active_ok = True
                if USE_ACTIVITY_GATE:
                    active_ok = (1.0 - p_no) >= ACTIVITY_THRESH

                ready = (
                    active_ok and
                    (p_cand >= FIRE_THRESH) and
                    ((p_cand - p_no) >= MARGIN_OVER_NO) and
                    (stable_count >= ACCEPT_N) and
                    (now - last_fire >= COOLDOWN_SEC)
                )
                if ready:
                    print(f"[GESTURE] {LABELS[cand]:<10}  p_gest={p_cand:.2f}  p_no={p_no:.2f}  frame={hdr['frame']}")
                    last_fire = now

    except KeyboardInterrupt:
        print("\n[INFO] Stopping.")
    finally:
        try: data.close()
        except: pass

if __name__ == '__main__':
    main()
