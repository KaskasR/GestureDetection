#!/usr/bin/env python3
import argparse, time, struct, sys
import serial

MAGIC = b'\x02\x01\x04\x03\x06\x05\x08\x07'  # 0x0201040306050807 little-endian
HDR_FMT = '<QIIIIIIII'                        # magic(8) + 8x uint32
HDR_LEN = struct.calcsize(HDR_FMT)            # 40 bytes

def open_serial(port, baud, timeout):
    return serial.Serial(port=port, baudrate=baud, timeout=timeout, write_timeout=timeout)

def cli_send(ser_cli, cmd, wait=1.0):
    """Send a CLI command and grab whatever comes back for a short time."""
    ser_cli.reset_input_buffer()
    ser_cli.write((cmd + '\r\n').encode('ascii'))
    ser_cli.flush()
    t0 = time.time()
    buf = b''
    while time.time() - t0 < wait:
        chunk = ser_cli.read(ser_cli.in_waiting or 1)
        if chunk:
            buf += chunk
        else:
            time.sleep(0.01)
    return buf

def cli_ping(ser_cli):
    out = cli_send(ser_cli, 'version', wait=1.0)
    if b'version' in out.lower() or b'sdk' in out.lower() or b'industrial' in out.lower() or len(out) > 0:
        return True, out.decode('latin1', errors='ignore')
    return False, out.decode('latin1', errors='ignore')

def start_sensor(ser_cli):
    out = cli_send(ser_cli, 'sensorStart', wait=1.0)
    txt = out.decode('latin1', errors='ignore')
    ok = ('Done' in txt) or ('already' in txt.lower()) or ('start' in txt.lower())
    return ok, txt

def stop_sensor(ser_cli):
    out = cli_send(ser_cli, 'sensorStop', wait=1.0)
    txt = out.decode('latin1', errors='ignore')
    ok = ('Done' in txt) or ('stopped' in txt.lower())
    return ok, txt

def find_header(ser_data, scan_ms=2000):
    """
    Look for a valid frame header (magic + 32 bytes) on the DATA port.
    Returns (header_dict, raw_header_bytes) or (None, b'') if not found.
    """
    deadline = time.time() + (scan_ms/1000.0)
    buf = bytearray()
    while time.time() < deadline:
        chunk = ser_data.read(ser_data.in_waiting or 1)
        if chunk:
            buf += chunk
            # Keep buffer from growing without bound
            if len(buf) > 65536:
                buf = buf[-65536:]
            # Search for magic
            idx = buf.find(MAGIC)
            if idx != -1:
                # If we don't have full header yet, keep reading a bit more
                while len(buf) - idx < HDR_LEN and time.time() < deadline:
                    more = ser_data.read(ser_data.in_waiting or 1)
                    if more:
                        buf += more
                    else:
                        time.sleep(0.005)
                if len(buf) - idx >= HDR_LEN:
                    hdr_bytes = bytes(buf[idx:idx+HDR_LEN])
                    fields = struct.unpack(HDR_FMT, hdr_bytes)
                    magic, version, total_len, platform, frame_num, ts, num_obj, num_tlvs, subframe = fields
                    # Sanity checks (tune as needed)
                    sane = (
                        magic == int.from_bytes(MAGIC, 'little') and
                        48 <= total_len <= 200000 and
                        num_tlvs <= 64
                    )
                    if sane:
                        hdr = {
                            'version': version,
                            'totalPacketLen': total_len,
                            'platform': platform,
                            'frameNumber': frame_num,
                            'timeStamp': ts,
                            'numDetectedObj': num_obj,
                            'numTLVs': num_tlvs,
                            'subFrameNumber': subframe,
                        }
                        return hdr, hdr_bytes
                    # If magic matched but sanity failed, drop this magic and keep scanning
                    buf = buf[idx+8:]
        else:
            time.sleep(0.005)
    return None, b''

def main():
    ap = argparse.ArgumentParser(description='Ping check for TI mmWave CLI + DATA ports.')
    ap.add_argument('--cli-port', help='CLI COM (e.g. COM7)')
    ap.add_argument('--data-port', required=True, help='DATA COM (e.g. COM8)')
    ap.add_argument('--baud', type=int, default=921600, help='DATA baud (default 921600)')
    ap.add_argument('--start', action='store_true', help='Send sensorStart over CLI before data check')
    ap.add_argument('--stop', action='store_true', help='Send sensorStop over CLI first')
    ap.add_argument('--scan-ms', type=int, default=2000, help='How long to scan for a frame (ms)')
    args = ap.parse_args()

    # --- CLI ping (optional) ---
    if args.cli_port:
        try:
            ser_cli = open_serial(args.cli_port, 115200, 1.0)
        except Exception as e:
            print(f'❌ Could not open CLI {args.cli_port}: {e}')
            ser_cli = None
        if ser_cli:
            ok, txt = cli_ping(ser_cli)
            if ok:
                print(f'✅ CLI responded to "version" on {args.cli_port}')
            else:
                print(f'⚠️  CLI opened but "version" did not respond clearly. Output:\n{txt.strip()}')
            if args.stop:
                ok, t = stop_sensor(ser_cli)
                print(('✅ sensorStop OK' if ok else '⚠️ sensorStop uncertain') + f': {t.strip()}')
            if args.start:
                ok, t = start_sensor(ser_cli)
                print(('✅ sensorStart OK' if ok else '⚠️ sensorStart uncertain') + f': {t.strip()}')
            ser_cli.close()
    else:
        print('ℹ️  No CLI port provided; skipping CLI ping.')

    # --- DATA ping ---
    try:
        ser_data = open_serial(args.data_port, args.baud, 0.05)
    except Exception as e:
        print(f'❌ Could not open DATA {args.data_port}: {e}')
        sys.exit(2)

    print(f'Checking DATA stream on {args.data_port} @ {args.baud} ...')
    hdr, raw = find_header(ser_data, scan_ms=args.scan_ms)
    ser_data.close()

    if hdr:
        print('✅ DATA stream detected. Frame header looks sane:')
        for k, v in hdr.items():
            print(f'   {k}: {v}')
        # Quick hint if the recorder might be incompatible
        if hdr['numTLVs'] == 0 or hdr['totalPacketLen'] < 48:
            print('⚠️ TLV count/length looks odd; your recorder may not match this profile.')
        sys.exit(0)
    else:
        print('❌ No valid frame header found. Either the sensor is not started, using a different profile, or another app has the DATA port open.')
        print('   Tips:')
        print('   • Make sure no other program (Visualizer, old Python script) has COM ports open.')
        print('   • Start the sensor with the gesture demo config, then rerun this ping.')
        print('   • Use --cli-port and --start to try starting it here.')
        sys.exit(3)

if __name__ == '__main__':
    main()
