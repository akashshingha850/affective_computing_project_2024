import Jetson.GPIO as GPIO
import time

# Set the GPIO pin numbering mode
GPIO.setmode(GPIO.BOARD)

# Set up pin 40 as an output pin
buzzer_pin = 40
GPIO.setup(buzzer_pin, GPIO.OUT)

# Function to make sound with the buzzer
def buzz(frequency, duration):
    delay = 1.0 / frequency
    cycles = int(frequency * duration)

    for _ in range(cycles):
        GPIO.output(buzzer_pin, GPIO.HIGH)  # Turn on the buzzer
        time.sleep(delay / 2)                # Wait for half the delay
        GPIO.output(buzzer_pin, GPIO.LOW)   # Turn off the buzzer
        time.sleep(delay / 2)                # Wait for half the delay

def alert_tone():
    # Alert tone pattern
    high_frequency = 1200  # High frequency for alert
    low_frequency = 600     # Low frequency for alert
    duration = 0.3          # Duration of each beep (in seconds)
    repetitions = 3         # Number of repetitions for each frequency

    for _ in range(repetitions):
        buzz(high_frequency, duration)  # High tone
        time.sleep(0.1)                 # Short pause
        buzz(low_frequency, duration)    # Low tone
        time.sleep(0.1)                 # Short pause

try:
    while True:
        alert_tone()  # Play the alert tone in a loop
        time.sleep(1)  # Wait before repeating the alert tone

except KeyboardInterrupt:
    print("Program terminated")

finally:
    GPIO.cleanup()  # Clean up GPIO settings
