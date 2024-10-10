import cv2
import time
import Jetson.GPIO as GPIO
from face_eye_detection import FaceEyeDetector
from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize

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
    GPIO.output(yellow_led_pin, GPIO.LOW)
    GPIO.output(red_led_pin, GPIO.LOW)

def turn_on_yellow():
    GPIO.output(green_led_pin, GPIO.LOW)
    GPIO.output(yellow_led_pin, GPIO.HIGH)
    GPIO.output(red_led_pin, GPIO.LOW)

def turn_on_red():
    GPIO.output(green_led_pin, GPIO.LOW)
    GPIO.output(yellow_led_pin, GPIO.LOW)
    GPIO.output(red_led_pin, GPIO.HIGH)

def turn_off_all():
    GPIO.output(green_led_pin, GPIO.LOW)
    GPIO.output(yellow_led_pin, GPIO.LOW)
    GPIO.output(red_led_pin, GPIO.LOW)

def main():
    # Initialize variables and settings that are adjustable
    eye_margin_x = 15
    eye_margin_y = 20
    face_model = "facedetect"
    eye_model = "/home/jetson/jetson-inference/data/networks/sleep/resnet18.onnx"
    labels = "/home/jetson/jetson-inference/data/networks/sleep/labels.txt"

    # Initialize video source (camera) and video output (display)
    camera = videoSource("/dev/video0")
    display = videoOutput("display://0")

    # Initialize face and eye detector
    face_eye_detector = FaceEyeDetector(
        eye_margin_x=eye_margin_x,
        eye_margin_y=eye_margin_y,
        face_model=face_model,
        eye_model=eye_model,
        labels=labels
    )

    closed_start_time = None  # Track when both eyes are closed
    close_duration_threshold = 3  # 3 seconds threshold for red LED

    while display.IsStreaming():
        img = camera.Capture()

        if img is None:
            continue

        # Detect and classify the state of the eyes
        left_eye_state, right_eye_state, cropped_face, left_eye_img, right_eye_img = face_eye_detector.detect_and_classify(img)

        # Synchronize to ensure GPU is finished before rendering
        cudaDeviceSynchronize()

        # If no face is detected, turn off all LEDs
        if cropped_face is None:
            turn_off_all()
        else:
            # Check if both eyes are open or closed
            if left_eye_state == "open" and right_eye_state == "open":
                turn_on_green()
                closed_start_time = None  # Reset closed timer
            elif left_eye_state == "close" and right_eye_state == "close":
                turn_on_yellow()
                
                if closed_start_time is None:
                    closed_start_time = time.time()  # Start timing how long eyes are closed
                elif time.time() - closed_start_time >= close_duration_threshold:
                    turn_on_red()  # Both eyes have been closed for 3 seconds
            else:
                closed_start_time = None  # Reset if eyes are not consistently closed

        # Display the cropped face and combined eyes
        face_eye_detector.display_face_and_eyes(cropped_face, left_eye_img, right_eye_img)

        # Render the original image
        display.Render(img)
        display.SetStatus(f"Face & Eye Detection | Network {face_eye_detector.face_detection.GetNetworkFPS():.0f} FPS")

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    GPIO.cleanup()  # Cleanup all GPIO

if __name__ == "__main__":
    main()
