#run.py

import time
import os
import signal
import subprocess
import Jetson.GPIO as GPIO
import utils

# Pin Definitions
switch_pin = 7  # Use BOARD pin 7 for the switch

# Setup GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Pull-up resistor

def run_application():
    """Run the main application and return the process."""
    return subprocess.Popen(["python3", "/home/jetson/affective_computing/project/main.py"])

def signal_handler(sig, frame):
    """Handle termination signal."""
    GPIO.cleanup()
    os._exit(0)

def main():
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)

    # Main loop
    application_process = None
    was_running = False  # Track if the application was previously running

    while True:
        switch_state = GPIO.input(switch_pin)
        if switch_state == GPIO.LOW:  # Switch pressed
            if application_process is None or application_process.poll() is not None:
                application_process = run_application()
                if not was_running:
                    print("Starting application...")
                    was_running = True  # Update the running state
        else:
            if application_process is not None:
                application_process.terminate()  # Terminate the process
                application_process = None
                if was_running:
                    print("Application Stopped")
                    utils.turn_off_all()  # Turn off all LEDs

                    was_running = False  # Update the running state

        # Check if the application crashed and restart if needed
        if application_process is not None and application_process.poll() is not None:
            application_process = None  # Reset process if it has exited
            if was_running:
                print("Application crashed. Restarting...")
                was_running = False  # Update the running state

        time.sleep(1)  # Check the switch state every second

if __name__ == "__main__":
    main()
