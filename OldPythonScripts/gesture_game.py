import serial
import pygame
import threading

# --- SERIAL SETUP ---
CLI_PORT = "COM7"    # Change to your CLI port
BAUD_RATE = 115200
ser = serial.Serial(CLI_PORT, BAUD_RATE, timeout=1)

# --- GAME SETUP ---
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Gesture Controlled UI")
clock = pygame.time.Clock()

player_pos = [400, 300]
running = True

# --- THREAD TO READ GESTURES ---
def read_gestures():
    global player_pos, running
    while running:
        line = ser.readline().decode('utf-8', errors='ignore').strip().lower()
        if line:
            print("Gesture detected:", line)
            if "up-to-down" in line:
                player_pos[0] -= 20
            elif "down-to-up" in line:
                player_pos[0] += 20
            elif "on-gesture" in line:
                player_pos[1] -= 20
            elif "off-gesture" in line:
                player_pos[1] += 20

gesture_thread = threading.Thread(target=read_gestures, daemon=True)
gesture_thread.start()

# --- MAIN GAME LOOP ---
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    screen.fill((30, 30, 30))
    pygame.draw.rect(screen, (0, 200, 0), (*player_pos, 50, 50))
    pygame.display.flip()
    clock.tick(60)

running = False
ser.close()
pygame.quit()
