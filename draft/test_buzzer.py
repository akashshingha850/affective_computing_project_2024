import time
import Jetson.GPIO as GPIO

# Set up GPIO
GPIO.setmode(GPIO.BCM)
BUZZER_PIN = 21  # Use BCM pin 21, which corresponds to BOARD pin 40
GPIO.setup(BUZZER_PIN, GPIO.OUT)

def buzz(pitch, duration):
    period = 1.0 / pitch
    delay = period / 2
    cycles = int(duration * pitch)
    
    for i in range(cycles):
        GPIO.output(BUZZER_PIN, True)
        time.sleep(delay * 0.8)  # Increase the on-time to 80% of the period
        GPIO.output(BUZZER_PIN, False)
        time.sleep(delay * 0.2)  # Decrease the off-time to 20% of the period

try:
    while True:
        pitch = float(input("Enter the pitch (Hz): "))
        duration = float(input("Enter the duration (seconds): "))
        buzz(pitch, duration)
except KeyboardInterrupt:
    print("Program stopped")
finally:
    GPIO.cleanup()