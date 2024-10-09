import Jetson.GPIO as GPIO
import time

# Pin Definitions
input_pin = 7  # BOARD pin number for the button

# Set up GPIO mode
GPIO.setmode(GPIO.BOARD)
GPIO.setup(input_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set pin 7 as input with pull-down resistor

try:
    print("Press the button...")
    while True:
        # Read the state of the input pin
        state = GPIO.input(input_pin)
        
        if state == GPIO.HIGH:  # Button pressed (connected to 3.3V)
            print("Button is pressed!")
        else:  # Button released (not pressed)
            print("Button is not pressed.")
        
        time.sleep(0.5)  # Delay to prevent rapid reading
except KeyboardInterrupt:
    print("Exiting...")

finally:
    # Cleanup GPIO settings before exiting
    GPIO.cleanup()
