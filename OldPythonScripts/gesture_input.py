import serial
import threading
from pynput.keyboard import Controller, Key

keyboard = Controller()

class GestureReader:
    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.ser = serial.Serial(port, baud, timeout=1)
        self.running = False
        self.callbacks = {}

    def on(self, gesture_name, func):
        """Register a function to call when a gesture is detected."""
        self.callbacks[gesture_name.lower()] = func

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._read_loop, daemon=True)
        thread.start()

    def _read_loop(self):
        while self.running:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip().lower()
            if line:
                for gesture, func in self.callbacks.items():
                    if gesture in line:
                        func()

    def stop(self):
        self.running = False
        self.ser.close()
