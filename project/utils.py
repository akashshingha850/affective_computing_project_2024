# utils.py
import time
import Jetson.GPIO as GPIO

# Pin Definitions
green_led_pin = 11  # BOARD pin 33
yellow_led_pin = 13  # BOARD pin 35
red_led_pin = 15    # BOARD pin 37

# Setup GPIO pins
GPIO.setmode(GPIO.BOARD)
GPIO.setup(green_led_pin, GPIO.OUT)
GPIO.setup(yellow_led_pin, GPIO.OUT)
GPIO.setup(red_led_pin, GPIO.OUT)

# LED control functions
def turn_on_green():
    GPIO.output(green_led_pin, GPIO.HIGH)
    turn_off_yellow()
    turn_off_red()

def turn_on_yellow():
    GPIO.output(yellow_led_pin, GPIO.HIGH)
    turn_off_green()
    turn_off_red()

def turn_on_red():
    GPIO.output(red_led_pin, GPIO.HIGH)
    turn_off_green()
    turn_off_yellow()

def turn_off_green():
    GPIO.output(green_led_pin, GPIO.LOW)

def turn_off_yellow():
    GPIO.output(yellow_led_pin, GPIO.LOW)

def turn_off_red():
    GPIO.output(red_led_pin, GPIO.LOW)

def turn_off_all():
    turn_off_green()
    turn_off_yellow()
    turn_off_red()

# Cleanup GPIO on exit
def cleanup():
    GPIO.cleanup()
