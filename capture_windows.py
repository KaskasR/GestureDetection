# capture_windows.py  — robust: logs features (TLV 1050) AND scans frame bytes for ANN probs
# SAME windows.csv format as before. Adds probs_raw.csv reliably even if TLV headers vary.
#
# Controls:
#   0=GUIDED NoGesture  1=Pinch  2=Fist  3=Wave
#   x=Cancel guided     r=Manual record  s=Snapshot  v=Verbose
#   m=Manual label cycle (0–3)   q=Quit
#
# Notes:
# - Features TLV type 1050 is still used for features.
# - ANN probs are found by scanning the frame payload for N floats (8<=N<=16) with sum≈1.

import serial, struct, time, sys, csv, random, math
from collections import deque

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
FRAME_HDR_LEN = 40
TLV_HDR_LEN = 8

CLI_PORT, CLI_BAUD  = 'COM7', 115200
DATA_PORT, DATA_BAUD = 'COM8', 921600

WINDOW = 15
OUT_WINDOWS_CSV = 'windows.csv'
OUT_FRAMES_CSV  = 'frames_raw.csv'
OUT_PROBS_CSV   = 'probs_raw.csv'

LABELS = {0: "NoGesture", 1: "Pinch", 2: "Fist", 3: "Wave"}

GUIDED = {
    "NoGesture": {"total_seconds": 60},
    "Pinch":     {"reps": 16, "action_s": (0.8,1.4), "rest_s": (0.8,1.3)},
    "Fist":      {"reps": 16, "action_s": (0.8,1.4), "rest_s": (0.8,1.3)},
    "Wave":      {"reps": 16, "action_s": (0.8,1.4), "rest_s": (0.8,1.3)},
    "lead_in":   3.0,
    "jitter":    0.25
}

ON_WINDOWS = sys.platform.startswith('win')
if ON_WINDOWS:
    import msvcrt
    def get_key_nonblock():
        if msvcrt.kbhit():
            try: return msvcrt.getch().decode('utf-8')
            except: return ''
        return None
else:
    import select
    def get_key_nonblock():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr: return sys.stdin.read(1)
        return None

def open_port(name, baud):
    return serial.Serial(name, baudrate=baud, bytesize=8, parity='N', stopbits=1,
                         timeout=0.05, write_timeout=0.1)

def find_magic(buf): return buf.find(MAGIC)

def parse_frame_header(buf):
    _, ver, total_len, plat, frame_num, ts, num_obj, num_tlvs, sub = struct.unpack_from('<Q8I', buf, 0)
    return dict(version=ver, total_len=total_len, platform=plat, frame=frame_num,
                ts=ts, num_obj=num_obj, num_tlvs=num_tlvs, sub=sub)

def iter_tlvs(pkt, num_tlvs):
    off = FRAME_HDR_LEN
    for _ in range(num_tlvs):
        if off + TLV_HDR_LEN > len(pkt): return
        t, tlv_len = struct.unpack_from('<II', pkt, off)
        if tlv_len < TLV_HDR_LEN: return
        end = off + tlv_len
        if end > len(pkt): return
        payload = pkt[off + TLV_HDR_LEN:end]
        yield t, payload
        off = end

def floats_from_payload(payload):
    if len(payload) % 4: return None
    n = len(payload)//4
    return list(struct.unpack('<' + 'f'*n, payload))

def ensure_files():
    for path in (OUT_WINDOWS_CSV, OUT_FRAMES_CSV, OUT_PROBS_CSV):
        try:
            with open(path, 'x', newline='') as f: pass
        except FileExistsError:
            pass

# --- guided session helper ---
class GuidedSession:
    def __init__(self):
        self.active=False; self.phases=[]; self.idx=0; self._announced=set(); self._final=""
    def start_nogesture(self, now):
        total=GUIDED["NoGesture"]["total_seconds"]; lead=GUIDED["lead_in"]
        self.phases=[
            {"t_start":now,"t_end":now+lead,"label":0,"recording":False,
             "message":f"Get ready: NoGesture in {int(lead)}s. Hold hand ~30–60 cm, centered."},
            {"t_start":now+lead,"t_end":now+lead+total,"label":0,"recording":True,
             "message":f"NO GESTURE: Hold still now for {total}s. Natural background OK."}
        ]
        self._final="NoGesture session complete."; self._reset()
    def start_action(self, now, label):
        name=LABELS[label]; cfg=GUIDED[name]; reps=cfg["reps"]; lead=GUIDED["lead_in"]; jitter=GUIDED["jitter"]
        self.phases=[{"t_start":now,"t_end":now+lead,"label":label,"recording":False,
                      "message":f"Get ready: {name.upper()} in {int(lead)}s. Distance 30–60 cm; vary position."}]
        t=now+lead
        for rep in range(1,reps+1):
            act=max(0.5, random.uniform(*cfg["action_s"])+random.uniform(-jitter,jitter))
            rest=max(0.5, random.uniform(*cfg["rest_s"])+random.uniform(-jitter,jitter))
            action_msg={1:f"PINCH #{rep}: pinch-in & release. GO!",
                        2:f"FIST  #{rep}: make fist + tiny wag, release. GO!",
                        3:f"WAVE  #{rep}: small left↔right wave. GO!"}[label]
            self.phases.append({"t_start":t,"t_end":t+act,"label":label,"recording":True,"message":action_msg}); t+=act
            self.phases.append({"t_start":t,"t_end":t+rest,"label":label,"recording":False,"message":f"Rest {rest:.1f}s…"}); t+=rest
        self._final=f"{name} session complete."; self._reset()
    def _reset(self): self.active=True; self.idx=0; self._announced.clear()
    def cancel(self): self.active=False; self.phases=[]; self.idx=0; self._announced.clear()
    def current(self, now):
        if not self.active or not self.phases: return None,None,None
        if self.idx>=len(self.phases):
            msg=self._final; self.cancel(); return None,None,msg
        ph=self.phases[self.idx]
        while self.idx<len(self.phases) and now>=self.phases[self.idx]["t_end"]:
            self.idx+=1
            if self.idx>=len(self.phases):
                msg=self._final; self.cancel(); return None,None,msg
            ph=self.phases[self.idx]
        msg=None
        if self.idx not in self._announced:
            msg=ph["message"]; self._announced.add(self.idx)
        return ph["label"], ph["recording"], msg

# ---- softmax block finder (bytescan) ----
def find_softmax_block(pkt):
    """Search frame payload for N floats (8<=N<=16) that sum ~1 and are all finite/non-negative."""
    start = FRAME_HDR_LEN
    end   = len(pkt)
    best = None
    for off in range(start, end - 4*8 + 1, 4):
        maxN = min(16, (end - off)//4)
        for n in range(8, maxN+1):
            try:
                vals = struct.unpack_from('<' + 'f'*n, pkt, off)
            except struct.error:
                break
            if any((math.isnan(v) or math.isinf(v) or v < -1e-3) for v in vals):
                continue
            s = sum(vals)
            # score closeness to 1 and probability-like shape
            score = abs(s - 1.0)
            if score <= 0.12:  # within ~12% of 1.0
                # prefer vectors with values in [0,1.2]
                if all(v <= 1.2 for v in vals):
                    if best is None or score < best[0]:
                        best = (score, off, n, vals)
    if best:
        return best[2], list(best[3])  # (length, values)
    return None, None

def main():
    try:
        cli = open_port(CLI_PORT, CLI_BAUD); data = open_port(DATA_PORT, DATA_BAUD)
    except Exception as e:
        print(f"[ERROR] Opening ports: {e}"); sys.exit(1)
    data.reset_input_buffer(); time.sleep(0.1); ensure_files()

    window = deque(maxlen=WINDOW)
    label=0; recording=False; verbose=False
    n_features=None; frames=0; saved=0

    win_f=open(OUT_WINDOWS_CSV,'a',newline=''); win_writer=csv.writer(win_f)
    frm_f=open(OUT_FRAMES_CSV,'a',newline='');  frm_writer=csv.writer(frm_f)
    prb_f=open(OUT_PROBS_CSV,'a',newline='');  prb_writer=csv.writer(prb_f)

    print("[INFO] Robust capture: features via TLV 1050, ANN probs via softmax scan.")
    print("[INFO] Controls: 0=NoGesture, 1=Pinch, 2=Fist, 3=Wave, x=cancel, r=record, s=snapshot, v=verbose, m=cycle label, q=quit")

    guided=GuidedSession()
    buf=bytearray()

    try:
        while True:
            # guided update
            now=time.perf_counter()
            g_label,g_record,g_msg=guided.current(now)
            if g_msg: print(f"[CUE] {g_msg}")
            if g_label is not None: label=g_label
            if g_record is not None: recording=g_record

            # read
            chunk=data.read(4096)
            if chunk: buf.extend(chunk)

            # frame sync
            while True:
                idx=find_magic(buf)
                if idx<0:
                    if len(buf)>65536: del buf[:-7]
                    break
                if idx>0: del buf[:idx]
                if len(buf)<FRAME_HDR_LEN: break

                hdr=parse_frame_header(buf)
                total_len=hdr['total_len']
                if total_len<FRAME_HDR_LEN or total_len>128*1024:
                    del buf[:1]; continue
                if len(buf)<total_len: break

                pkt=bytes(buf[:total_len]); del buf[:total_len]
                frames+=1

                # 1) scan probs first (independent of TLVs)
                nprob, probs = find_softmax_block(pkt)
                if nprob:
                    # write header once
                    if prb_f.tell()==0:
                        prb_writer.writerow(['frame'] + [f'p{i}' for i in range(nprob)])
                        prb_f.flush()
                    prb_writer.writerow([hdr['frame']] + [f"{x:.6f}" for x in probs]); prb_f.flush()

                # 2) iterate TLVs to grab features 1050
                # (defensive — if TLVs mismatch we still keep probs)
                for t,payload in iter_tlvs(pkt, hdr['num_tlvs']):
                    if t!=1050: continue
                    vals=floats_from_payload(payload)
                    if vals is None: continue
                    if n_features is None:
                        n_features=len(vals)
                        print(f"[INFO] Features TLV detected: {n_features} floats/frame.")
                        frm_writer.writerow(['frame','label','n_features']+[f'f{i}' for i in range(n_features)])
                        frm_f.flush()
                    frm_writer.writerow([hdr['frame'],label,n_features]+[f"{x:.6f}" for x in vals]); frm_f.flush()
                    window.append(vals)
                    if len(window)==WINDOW and recording:
                        flat=[]; [flat.extend(tstep) for tstep in window]
                        win_writer.writerow([label,n_features,WINDOW]+[f"{x:.6f}" for x in flat]); win_f.flush()
                        saved+=1
                        if saved%10==0:
                            print(f"[INFO] Saved windows: {saved} (label {label} = {LABELS.get(label,'?')})")

                    if verbose and len(window)>0:
                        head=", ".join(f"{x:.3f}" for x in window[0][:min(5,n_features)])
                        print(f"[DBG] frame={hdr['frame']} label={label} rec={recording} f0..={head} ...")

            # keys
            k=get_key_nonblock()
            if k:
                if k.lower()=='q': raise KeyboardInterrupt
                elif k.lower()=='x': guided.cancel(); print("[STATE] Guided cancelled.")
                elif k.lower()=='r': recording=not recording; print(f"[STATE] Manual recording = {recording}")
                elif k.lower()=='s':
                    if len(window)==WINDOW and n_features is not None:
                        flat=[]; [flat.extend(t) for t in window]
                        win_writer.writerow([label,n_features,WINDOW]+[f"{x:.6f}" for x in flat]); win_f.flush()
                        saved+=1; print(f"[INFO] Snapshot saved. Total windows: {saved}")
                    else:
                        print("[WARN] Window not full yet.")
                elif k.lower()=='v': verbose=not verbose; print(f"[STATE] Verbose = {verbose}")
                elif k.lower()=='m': label=(label+1)%4; print(f"[STATE] Manual label = {label} ({LABELS[label]})")
                elif k=='0': guided.start_nogesture(time.perf_counter())
                elif k=='1': guided.start_action(time.perf_counter(),1)
                elif k=='2': guided.start_action(time.perf_counter(),2)
                elif k=='3': guided.start_action(time.perf_counter(),3)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        for f in (win_f,frm_f,prb_f):
            try: f.close()
            except: pass
        try: data.close(); cli.close()
        except: pass
        print(f"[INFO] Frames: {frames}, windows saved: {saved}")
        print(f"[INFO] Wrote {OUT_WINDOWS_CSV}, {OUT_FRAMES_CSV}, {OUT_PROBS_CSV}")

if __name__=='__main__':
    main()
