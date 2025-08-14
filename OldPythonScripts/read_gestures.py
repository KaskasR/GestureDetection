import serial

# CHANGE THIS to match your CLI COM port
CLI_PORT = "COM7"     # Example: COM5 on Windows
BAUD_RATE = 115200    # Common default for gesture lab CLI port

# Open the serial port
ser = serial.Serial(CLI_PORT, BAUD_RATE, timeout=1)

print(f"Listening for gestures on {CLI_PORT}... Press Ctrl+C to stop.")

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(line)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    ser.close()
