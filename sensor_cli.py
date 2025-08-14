# sensor_cli.py
# Use: python sensor_cli.py --port COM4 --start
#      python sensor_cli.py --port COM4 --stop
#      python sensor_cli.py --port COM4 --cmd "version"
#      python sensor_cli.py --port COM4 --cfg "C:\path\to\gesture_aop.cfg"

import argparse, time, serial, sys, os

def send_line(ser, line, read_lines=6, delay=0.03):
    """Send one CLI line with CRLF and print a few reply lines."""
    msg = (line + "\r\n").encode("utf-8")
    print(f">>> {line}")
    ser.write(msg)
    ser.flush()
    time.sleep(delay)
    for _ in range(read_lines):
        resp = ser.readline().decode("utf-8", errors="ignore").strip()
        if resp:
            print(f"<<< {resp}")

def main():
    p = argparse.ArgumentParser(description="TI mmWave CLI helper")
    p.add_argument("--port", required=True, help="CLI/CFG COM port (e.g., COM4)")
    p.add_argument("--baud", type=int, default=115200, help="CLI baud (default 115200)")
    p.add_argument("--start", action="store_true", help="Send sensorStart")
    p.add_argument("--stop", action="store_true", help="Send sensorStop")
    p.add_argument("--cmd", action="append", help="Extra CLI command(s) to send (can repeat)")
    p.add_argument("--cfg", help="Path to .cfg file to send (optional)")
    args = p.parse_args()

    try:
        with serial.Serial(args.port, args.baud, timeout=1) as ser:
            # small settle delay
            time.sleep(0.2)
            # sanity ping (optional)
            if not args.cfg and not args.cmd and not args.start and not args.stop:
                args.cmd = ["version"]

            # Stop first if requested (safe even if already stopped)
            if args.stop:
                send_line(ser, "sensorStop")

            # Send a .cfg file if provided
            if args.cfg:
                if not os.path.isfile(args.cfg):
                    print(f"CFG not found: {args.cfg}", file=sys.stderr)
                    sys.exit(1)
                with open(args.cfg, "r") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith(("%", "#", "//")):
                            continue
                        send_line(ser, line)
                        # some lines benefit from a slightly longer pause
                        time.sleep(0.02)

            # Any ad-hoc commands
            if args.cmd:
                for c in args.cmd:
                    send_line(ser, c)

            # Start at the end if requested
            if args.start:
                send_line(ser, "sensorStart")

            print("Done.")
    except serial.SerialException as e:
        print(f"ERROR opening {args.port}: {e}\n"
              f"- Make sure no other program (CCS terminal, PuTTY, another script) is using {args.port}.\n"
              f"- Confirm this is the CLI/CFG port (not the Data port).", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
