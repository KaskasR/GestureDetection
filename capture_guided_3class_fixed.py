# capture_guided_3class_fixed.py  (with SNIFF + better fallback)
import argparse, csv, sys, time, struct
import serial
from math import isfinite

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_U64 = int.from_bytes(MAGIC, 'little')

FRAME_HDR_FMT = '<QIIIIIIII'   # magic,version,totalLen,platform,frameNum,timeStamp,numObj,numTLVs,subframe
FRAME_HDR_SZ  = struct.calcsize(FRAME_HDR_FMT)
TLV_HDR_FMT   = '<II'
TLV_HDR_SZ    = struct.calcsize(TLV_HDR_FMT)

MAX_PACKET_LEN = 200_000
MAX_TLVS       = 128

def open_serial(port, baud, timeout=0.1):
    return serial.Serial(port=port, baudrate=baud, timeout=timeout, write_timeout=timeout)

def cli_send(cli, line, wait=0.5):
    if cli is None: return True, ""
    try:
        cli.reset_input_buffer()
        cli.write((line.strip() + '\r\n').encode('ascii'))
        cli.flush()
        t0 = time.time()
        buf = b''
        while time.time() - t0 < wait:
            chunk = cli.read(cli.in_waiting or 1)
            if chunk: buf += chunk
            else: time.sleep(0.01)
        txt = buf.decode('latin1', errors='ignore')
        ok = ('Done' in txt) or ('ok' in txt.lower()) or ('already' in txt.lower())
        return ok, txt
    except Exception as e:
        return False, str(e)

def start_sensor(cli):
    if cli is None: return False
    cli_send(cli, 'sensorStop', wait=0.25)
    ok, _ = cli_send(cli, 'sensorStart', wait=0.8)
    if not ok:
        ok, _ = cli_send(cli, 'sensorStart 0', wait=0.8)
    return ok

def stop_sensor(cli):
    ok, _ = cli_send(cli, 'sensorStop', wait=0.6)
    return ok

def sane_header_fields(total_len, num_tlvs):
    if not (FRAME_HDR_SZ + TLV_HDR_SZ <= total_len <= MAX_PACKET_LEN):
        return False
    if not (0 <= num_tlvs <= MAX_TLVS):
        return False
    return True

def find_full_frame(buf):
    i = buf.find(MAGIC)
    if i < 0:
        return None, buf[-7:] if len(buf) > 7 else buf
    if len(buf) - i < FRAME_HDR_SZ:
        return None, buf[i:]
    try:
        magic, ver, total_len, platform, frame_no, ts, num_obj, num_tlvs, subf = struct.unpack_from(FRAME_HDR_FMT, buf, i)
    except struct.error:
        return None, buf[i+1:]
    if magic != MAGIC_U64 or not sane_header_fields(total_len, num_tlvs):
        return None, buf[i+1:]
    if len(buf) - i < total_len:
        return None, buf[i:]
    return buf[i:i+total_len], buf[i+total_len:]

def is_probabilities_10f(arr):
    if len(arr) != 10: return False
    s = 0.0
    for v in arr:
        if not isfinite(v): return False
        if v < -0.05 or v > 1.2:  # probs should be ~[0,1]
            return False
        s += v
    return 0.75 <= s <= 1.25

def looks_like_features_10f(arr):
    # Feature heuristics: often include values > 1, integers like numPoints, negatives, etc.
    if len(arr) != 10: return False
    if is_probabilities_10f(arr): return False
    any_gt_1p5 = any((v > 1.5) for v in arr if isfinite(v))
    any_intish = any((abs(v - round(v)) < 1e-3 and v >= 3.0) for v in arr if isfinite(v))
    any_neg    = any((v < 0.0) for v in arr if isfinite(v))
    return any_gt_1p5 or any_intish or any_neg

def extract_features(frame_bytes):
    # numTLVs is u32 at byte offset 32 in the 40B frame header
    try:
        num_tlvs = struct.unpack_from('<I', frame_bytes, 32)[0]
    except struct.error:
        return None

    off = FRAME_HDR_SZ
    end = len(frame_bytes)
    for _ in range(num_tlvs):
        if off + TLV_HDR_SZ > end: break
        try:
            tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, frame_bytes, off)
        except struct.error:
            break
        off += TLV_HDR_SZ

        paylen = tlv_len - TLV_HDR_SZ
        if paylen < 0 or off + paylen > end:
            break

        payload = frame_bytes[off:off+paylen]
        off += paylen

        # Your stream: type=1050, payload=32 (8 floats)
        if tlv_type == 1050 and len(payload) == 32:
            try:
                return struct.unpack('<8f', payload)  # 8 features
            except struct.error:
                continue

        # Backward-compat: older build with 10-float (40B) features
        if tlv_type == 1050 and len(payload) == 40:
            try:
                return struct.unpack('<10f', payload)  # 10 features
            except struct.error:
                continue
    return None


def check_stream_header(ser, seconds=1.5):
    t0 = time.time()
    buf = b''
    while time.time() - t0 < seconds:
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buf += chunk
            if len(buf) > 65536: buf = buf[-65536:]
            frame, buf = find_full_frame(buf)
            if frame: return True
        else:
            time.sleep(0.005)
    return False

def sniff_frames(ser, frames=3):
    print(f"\n🔎 Sniffing {frames} frame(s) to see what TLVs are present...")
    got = 0
    buf = b''
    t0 = time.time()
    while got < frames and time.time() - t0 < 10.0:
        buf += ser.read(4096)
        frame, buf = find_full_frame(buf)
        if not frame:
            continue
        got += 1
        try:
            _, ver, total_len, platform, frame_no, ts, num_obj, num_tlvs, subf = struct.unpack_from(FRAME_HDR_FMT, frame, 0)
        except struct.error:
            print(f" Frame {got}: header parse error")
            continue
        print(f" Frame {got}: total_len={total_len} numTLVs={num_tlvs} frameNo={frame_no}")
        off = FRAME_HDR_SZ
        end = len(frame)
        for k in range(min(num_tlvs, MAX_TLVS)):
            if off + TLV_HDR_SZ > end: break
            try:
                tlv_type, tlv_len = struct.unpack_from(TLV_HDR_FMT, frame, off)
            except struct.error:
                print("   TLV parse error.")
                break
            off += TLV_HDR_SZ
            if tlv_len < TLV_HDR_SZ or off + (tlv_len - TLV_HDR_SZ) > end:
                print(f"   TLV[{k}] malformed (len={tlv_len})")
                break
            payload = frame[off : off + (tlv_len - TLV_HDR_SZ)]
            off += (tlv_len - TLV_HDR_SZ)
            tag = ""
            if len(payload) == 40:
                try:
                    vals = struct.unpack('<10f', payload)
                    if is_probabilities_10f(vals): tag = " (10f PROBS)"
                    elif looks_like_features_10f(vals): tag = " (10f FEATURES)"
                    else: tag = " (10f unknown)"
                except:
                    tag = " (10f unpack err)"
            print(f"   TLV[{k}] type={tlv_type} len={tlv_len} payload={len(payload)}{tag}")
    if got == 0:
        print("  …didn’t catch any full frames. Is the sensor streaming?")

def guided_capture(ser_data, writer, label_name, samples, frames_per_sample=15, timeout_per_sample=12.0, accept_any_40=False):
    buf = b''
    for s in range(1, samples+1):
        print(f"\nGO {s}/{samples}")
        ser_data.reset_input_buffer()
        collected = 0
        feat_window = []
        last_progress = -1
        t_start = time.time()

        while collected < frames_per_sample and (time.time() - t_start) < timeout_per_sample:
            chunk = ser_data.read(4096)
            if chunk:
                buf += chunk
            else:
                time.sleep(0.002)
                continue
            if len(buf) > 262144: buf = buf[-262144:]

            frame, buf = find_full_frame(buf)
            if not frame: continue

            fvals = extract_features(frame)
            if fvals is None:
                continue
            feat_window.append(fvals[:6])
            collected += 1
            if collected != last_progress:
                print(f"  frame {collected}/{frames_per_sample}", end=('\r' if collected < frames_per_sample else '\n'))
                last_progress = collected

        if collected < frames_per_sample:
            print("  ⚠️  Timed out waiting for enough feature frames; skipping this sample.")
            continue

        row = [label_name]
        for fr in feat_window: row.extend(fr)
        writer.writerow(row)
        print("  saved.")

def main():
    ap = argparse.ArgumentParser(description="Guided 3-class capture (no_gesture, pinch, flash) with robust COM handling.")
    ap.add_argument('--data-port', '--port', dest='data_port', required=True, help='DATA UART (e.g., COM8)')
    ap.add_argument('--baud', type=int, default=921600, help='DATA baud rate (default 921600)')
    ap.add_argument('--cli-port', dest='cli_port', default=None, help='CLI UART (optional, e.g., COM7)')
    ap.add_argument('--cli-baud', dest='cli_baud', type=int, default=115200, help='CLI baud (default 115200)')
    ap.add_argument('--samples-per-class', type=int, default=50, help='How many samples per class')
    ap.add_argument('--outfile', default='dataset_3class.csv', help='CSV output path')
    ap.add_argument('--autostart', action='store_true', help='Try to start sensor via CLI automatically')
    ap.add_argument('--sniff', type=int, default=0, help='Sniff N frames and print TLVs, then exit')
    ap.add_argument('--timeout-per-sample', type=float, default=12.0, help='Seconds allowed to collect 15 frames')
    ap.add_argument('--accept-any-40', action='store_true', help='If no clear features TLV exists, accept any 40B TLV (even probs)')
    args = ap.parse_args()

    print(f"Opening DATA {args.data_port} @ {args.baud} ...")
    try:
        ser_data = open_serial(args.data_port, args.baud, timeout=0.05)
    except Exception as e:
        print(f"❌ Could not open DATA port {args.data_port}: {e}")
        sys.exit(2)

    ser_cli = None
    if args.cli_port:
        print(f"Opening CLI  {args.cli_port} @ {args.cli_baud} ...")
        try:
            ser_cli = open_serial(args.cli_port, args.cli_baud, timeout=0.1)
        except Exception as e:
            print(f"⚠️ Could not open CLI port {args.cli_port}: {e}")

    if args.autostart and ser_cli:
        print("Starting sensor via CLI ...")
        if start_sensor(ser_cli):
            print("✅ Sensor start acknowledged.")
        else:
            print("⚠️ Start may have failed; continuing.")

    print(f"Checking DATA stream on {args.data_port} @ {args.baud} ...")
    if check_stream_header(ser_data, seconds=2.0):
        print("✅ DATA stream detected (sane header found).")
    else:
        print("⚠️ No sane frame header yet. If the lab is running, confirm COMs and that no other app holds the port.")

    if args.sniff > 0:
        sniff_frames(ser_data, frames=args.sniff)
        try:
            if ser_cli: ser_cli.close()
        except: pass
        try:
            ser_data.close()
        except: pass
        print("Port(s) closed. Bye!")
        return

    print(f"\nConnected. Writing to {args.outfile}\n")

    menu = """
------------------------------------------------------------
Press a key to start guided capture:
  0 → no_gesture   (stay still as instructed)
  1 → pinch        (do the gesture at each GO prompt)
  2 → flash        (do the gesture at each GO prompt)
  s → start sensor via CLI
  x → stop sensor via CLI
  r → re-check DATA stream
  q → quit
------------------------------------------------------------
Your choice: """
    try:
        with open(args.outfile, 'a', newline='') as f:
            writer = csv.writer(f)
            while True:
                choice = input(menu).strip().lower()
                if choice == 'q':
                    break
                elif choice == 'r':
                    print("Re-checking stream ...")
                    ok = check_stream_header(ser_data, seconds=1.8)
                    print("✅ DATA stream detected." if ok else "⚠️ Still no sane header.")
                elif choice == 's':
                    if ser_cli and start_sensor(ser_cli):
                        print("✅ Sensor start acknowledged.")
                    else:
                        print("⚠️ Could not start (missing CLI or CLI busy).")
                elif choice == 'x':
                    if ser_cli and stop_sensor(ser_cli):
                        print("✅ Sensor stopped.")
                    else:
                        print("⚠️ Could not stop (missing CLI or CLI busy).")
                elif choice in ('0','1','2'):
                    labels = {'0':'no_gesture','1':'pinch','2':'flash'}
                    label = labels[choice]
                    N = args.samples_per_class
                    print("\n============================================================")
                    if label == 'no_gesture':
                        print(f"[{label}] Guided capture for {N} samples (each = 15 frames).")
                        print("You’ll see GO prompts. **Stay still** until it finishes.")
                    else:
                        print(f"[{label}] Guided capture for {N} samples (each = 15 frames).")
                        print("You’ll see GO prompts. Perform the gesture at each GO.")
                        print("Keep doing it smoothly/repeatedly until it finishes.")
                    print("Starting in 3..."); time.sleep(1)
                    print("2..."); time.sleep(1)
                    print("1..."); time.sleep(1)
                    guided_capture(
                        ser_data, writer, label, N,
                        frames_per_sample=15,
                        timeout_per_sample=args.timeout_per_sample,
                        accept_any_40=args.accept_any_40
                    )
                    print("✅ Class capture complete.")
                else:
                    print("Enter 0/1/2, s, x, r, or q.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        try:
            if ser_cli: ser_cli.close()
        except: pass
        try:
            ser_data.close()
        except: pass
        print("Port(s) closed. Bye!")

if __name__ == '__main__':
    main()
