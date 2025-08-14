import sys, time, serial

PORT = sys.argv[1] if len(sys.argv)>1 else "COM7"
BAUD = 115200

with serial.Serial(PORT, BAUD, timeout=0.5) as s:
    print(f"Opening {PORT}@{BAUD} …")
    def send(cmd):
        s.reset_input_buffer()
        s.write((cmd + "\r\n").encode("ascii"))  # CRLF is safest on TI CLI
        s.flush()
        time.sleep(0.15)
        out = s.read(2048)
        print(f">>> {cmd}")
        print(out.decode(errors="replace"))

    send("version")
    send("sensorStop")
    send("sensorStart 0")
