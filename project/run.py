import time
import os
import signal
import subprocess
import Jetson.GPIO as GPIO  # Only import for switch monitoring

# Pin Definitions
switch_pin = 7  # BOARD pin 7 for the switch

# Setup GPIO for the switch
GPIO.setmode(GPIO.BOARD)
GPIO.setup(switch_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Switch with pull-up resistor

def run_application():
    """
    Run the main application (main.py) and return the process.
    """
    return subprocess.Popen(["python3", "/home/jetson/affective_computing/project/main.py"])

def signal_handler(sig, frame):
    """
    Handle termination signals (like Ctrl+C) to ensure GPIO cleanup.
    """
    GPIO.cleanup()
    os._exit(0)

def main():
    # Register signal handler to ensure proper cleanup on exit
    signal.signal(signal.SIGINT, signal_handler)

    application_process = None
    was_running = False  # To track if the application was previously running

    try:
        while True:
            switch_state = GPIO.input(switch_pin)  # Get the current state of the switch

            if switch_state == GPIO.LOW:  # Switch is pressed (LOW state)
                if application_process is None or application_process.poll() is not None:
                    # Start the application if it's not running
                    application_process = run_application()
                    if not was_running:
                        print("Starting application...")
                        was_running = True
            else:
                # Switch is released, stop the application
                if application_process is not None:
                    application_process.terminate()  # Terminate the application
                    application_process = None
                    if was_running:
                        print("Application stopped")
                        was_running = False

            # Check if the application has crashed and restart it if necessary
            if application_process is not None and application_process.poll() is not None:
                application_process = None
                if was_running:
                    print("Application crashed. Restarting...")
                    was_running = False

            # Sleep briefly before checking again
            time.sleep(1)

    except KeyboardInterrupt:
        print("Exiting due to keyboard interrupt...")
    finally:
        # Ensure GPIO cleanup on exit
        GPIO.cleanup()

if __name__ == "__main__":
    main()
