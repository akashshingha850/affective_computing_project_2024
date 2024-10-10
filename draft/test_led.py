import time
import Jetson.GPIO as GPIO

# Pin Definitions
green_led_pin = 11  # BOARD pin 33
yellow_led_pin = 13  # BOARD pin 35
red_led_pin = 15  # BOARD pin 37

# Pin Setup
GPIO.setmode(GPIO.BOARD)  # Use physical pin-numbering scheme
GPIO.setup(green_led_pin, GPIO.OUT)  # Green LED pin set as output
GPIO.setup(yellow_led_pin, GPIO.OUT)  # Yellow LED pin set as output
GPIO.setup(red_led_pin, GPIO.OUT)  # Red LED pin set as output

# LED control functions
def turn_on_green():
    GPIO.output(green_led_pin, GPIO.HIGH)

def turn_off_green():
    GPIO.output(green_led_pin, GPIO.LOW)

def turn_on_yellow():
    GPIO.output(yellow_led_pin, GPIO.HIGH)

def turn_off_yellow():
    GPIO.output(yellow_led_pin, GPIO.LOW)

def turn_on_red():
    GPIO.output(red_led_pin, GPIO.HIGH)

def turn_off_red():
    GPIO.output(red_led_pin, GPIO.LOW)

print("Starting LED sequence. Press CTRL+C to exit.")
try:
    while True:
        # # Turn on  LED, turn off others
        # turn_off_green()
        # turn_on_yellow()
        # turn_off_red()
        # time.sleep(2)  # Keep Green on for 1 second

        # Turn on Green LED, turn off others
        turn_on_green()
        turn_off_yellow()
        turn_off_red()
        time.sleep(2)  # Keep Green on for 1 second

        # Turn on Yellow LED, turn off others
        turn_off_green()
        turn_on_yellow()
        turn_off_red()
        time.sleep(2)  # Keep Yellow on for 1 second

        # Turn on Red LED, turn off others
        turn_off_green()
        turn_off_yellow()
        turn_on_red()
        time.sleep(2)  # Keep Red on for 1 second

except KeyboardInterrupt:
    print("Exiting LED sequence.")
finally:
    GPIO.cleanup()  # Cleanup all GPIO

