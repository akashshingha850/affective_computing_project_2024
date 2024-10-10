import time
import os
import signal
import subprocess
import Jetson.GPIO as GPIO

# Pin Definitions
switch_pin = 7  # Use BOARD pin 7 for the switch

# Setup GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull-up resistor

def run_application():
    """Run the main application."""
    process = subprocess.Popen(["python3", "/home/jetson/affective_computing/project/main.py"])

    return process

def signal_handler(sig, frame):
    """Handle termination signal."""
    print("Exiting...")
    GPIO.cleanup()
    os._exit(0)

def main():
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    application_process = None
    while True:
        switch_state = GPIO.input(switch_pin)
        if switch_state == GPIO.LOW:  # Switch pressed
            if application_process is None or application_process.poll() is not None:
                print("Starting application...")
                application_process = run_application()
            else:
                print("Application is already running.")
        else:
            if application_process is not None:
                print("Stopping application...")
                application_process.terminate()  # Terminate the process
                application_process = None
            else:
                print("Application is not running.")

        time.sleep(1)  # Check the switch state every second

if __name__ == "__main__":
    main()
