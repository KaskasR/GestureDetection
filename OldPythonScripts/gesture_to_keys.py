from OldPythonScripts.gesture_input import GestureReader
from pynput.keyboard import Controller, Key

keyboard = Controller()

# Change this to your CLI COM port
PORT = "COM7"

# Define what each gesture should do
def go_left():
    keyboard.press('a')
    keyboard.release('a')

def go_right():
    keyboard.press('d')
    keyboard.release('d')

def go_up():
    keyboard.press('w')
    keyboard.release('w')

def go_down():
    keyboard.press('s')
    keyboard.release('s')

gr = GestureReader(port=PORT, baud=115200)
gr.on("twirl-ccw", go_left)
gr.on("twirl-cw", go_right)
gr.on("on-gesture", go_up)
gr.on("off-gesture", go_down)

gr.start()

print("Gesture to keyboard mapping active. Press Ctrl+C to stop.")
try:
    while True:
        pass
except KeyboardInterrupt:
    gr.stop()
    print("Stopped.")
