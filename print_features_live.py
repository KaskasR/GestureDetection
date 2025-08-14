# uart_peek.py
# Requires: pip install pyserial
# Usage:  python uart_peek.py
#
# What you should see for the features TLV when you're on the correct binary:
#   TLV 1050: len=48  payload=40  -> 10 floats
#
# If you see payload=32 -> 8 floats, you're likely running an older/different binary.

import serial, struct, time, sys

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'  # TI mmWave magic word (little endian)
FRAME_HDR_LEN = 40
TLV_HDR_LEN = 8

CLI_PORT   = 'COM7'     # your CFG/CLI port
CLI_BAUD   = 115200
DATA_PORT  = 'COM8'     # your DATA port
DATA_BAUD  = 921600     # typical for this lab; keep if your prior logs used 921600

def open_port(name, baud):
    try:
        ser = serial.Serial(
            name,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,      # non-blocking-ish
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            write_timeout=0.1
        )
        return ser
    except Exception as e:
        print(f"[ERROR] Could not open {name} @ {baud}: {e}")
        sys.exit(1)

def find_magic(buf):
    """Return index of MAGIC in buf, or -1."""
    return buf.find(MAGIC)

def parse_frame_header(buf):
    # <Q8I  => uint64 + 8x uint32  (8 + 32 = 40 bytes)
    magic, version, total_len, platform, frame_num, ts, num_obj, num_tlvs, subframe_num = struct.unpack_from('<Q8I', buf, 0)
    return {
        'version': version,
        'total_len': total_len,
        'platform': platform,
        'frame_num': frame_num,
        'timestamp': ts,
        'num_obj': num_obj,
        'num_tlvs': num_tlvs,
        'subframe_num': subframe_num
    }

def iter_tlvs(pkt, num_tlvs):
    off = FRAME_HDR_LEN
    for i in range(num_tlvs):
        if off + TLV_HDR_LEN > len(pkt):
            raise ValueError("Truncated TLV header")
        tlv_type, tlv_len = struct.unpack_from('<II', pkt, off)
        off += TLV_HDR_LEN
        if tlv_len < TLV_HDR_LEN or off + (tlv_len - TLV_HDR_LEN) > len(pkt):
            raise ValueError("Invalid TLV length")
        payload = pkt[off: off + (tlv_len - TLV_HDR_LEN)]
        off += (tlv_len - TLV_HDR_LEN)
        yield (tlv_type, tlv_len, payload)

def floats_from_payload(payload):
    n = len(payload) // 4
    if n <= 0 or len(payload) % 4 != 0:
        return None
    return list(struct.unpack('<' + 'f'*n, payload))

def main():
    print(f"[INFO] Opening CLI {CLI_PORT}@{CLI_BAUD} and DATA {DATA_PORT}@{DATA_BAUD} ...")
    cli = open_port(CLI_PORT, CLI_BAUD)
    data = open_port(DATA_PORT, DATA_BAUD)

    # Flush anything pending on DATA so we start at a clean boundary
    data.reset_input_buffer()
    time.sleep(0.1)

    buf = bytearray()
    frames = 0
    last_print = time.time()

    print("[INFO] Listening on DATA port. Press Ctrl+C to quit.\n")

    try:
        while True:
            chunk = data.read(4096)
            if chunk:
                buf.extend(chunk)

            # Try to sync to magic word
            while True:
                idx = find_magic(buf)
                if idx < 0:
                    # Keep buffer bounded
                    if len(buf) > 64 * 1024:
                        # retain last 7 bytes in case MAGIC straddles a read
                        del buf[:-7]
                    break
                if idx > 0:
                    # discard leading noise
                    del buf[:idx]

                # Need at least a full header
                if len(buf) < FRAME_HDR_LEN:
                    break

                hdr = parse_frame_header(buf)

                total_len = hdr['total_len']
                # Sanity check total_len
                if total_len < FRAME_HDR_LEN or total_len > 128 * 1024:
                    # Bad length; drop one byte and resync
                    del buf[0:1]
                    continue

                # Wait for full packet
                if len(buf) < total_len:
                    break

                # We have a full packet
                pkt = bytes(buf[:total_len])
                del buf[:total_len]

                frames += 1
                # Print a compact frame summary
                print(f"Frame {hdr['frame_num']:>6}  numTLVs={hdr['num_tlvs']:<2}  totalLen={hdr['total_len']}")

                # Walk TLVs
                try:
                    for (t, tlv_len, payload) in iter_tlvs(pkt, hdr['num_tlvs']):
                        # Compute payload bytes and float count
                        payload_len = tlv_len - TLV_HDR_LEN
                        float_count = payload_len // 4 if payload_len % 4 == 0 else 0

                        # TLV 1050 is Gesture Features in this lab
                        if t == 1050:
                            print(f"  TLV {t}: len={tlv_len}  payload={payload_len}  -> {float_count} floats  [Gesture Features]")
                            # Uncomment to dump the actual float values (debug)
                            # vals = floats_from_payload(payload)
                            # if vals is not None:
                            #     print('    ' + ', '.join(f'{v:.3f}' for v in vals))
                            if float_count == 10:
                                pass  # expected for this build (10 debug features output)
                            elif float_count == 8:
                                print("    [WARN] Only 8 floats found. This suggests an older/different binary than your current project.")
                            else:
                                print("    [INFO] Unusual feature count. Verify the project defines for NUM_GESTURE_FEATURES_OUTPUT.")
                        else:
                            # You may see another TLV with ANN output probabilities
                            # We don't assume a fixed type here; just summarize.
                            print(f"  TLV {t}: len={tlv_len}  payload={payload_len}  -> {float_count} floats")

                except Exception as e:
                    print(f"[WARN] TLV parse error: {e}")

            # Periodic heartbeat
            now = time.time()
            if now - last_print > 5 and frames == 0:
                print("[INFO] No frames yet. Ensure the board is in functional mode and the lab binary is running.")
                last_print = now

    except KeyboardInterrupt:
        print("\n[INFO] Exiting...")
    finally:
        try:
            data.close()
        except:
            pass
        try:
            cli.close()
        except:
            pass

if __name__ == '__main__':
    main()
