#main.py
import cv2
import utils  # Handles LED and GPIO control
from face_eye_detection import FaceEyeDetector
from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize
import time

def main():
    # Initialize variables and settings
    eye_margin_x = 15
    eye_margin_y = 20
    face_model = "facedetect"
    eye_model = "/home/jetson/jetson-inference/python/training/classification/models/mrl_vgg16/vgg16.onnx"
    labels = "/home/jetson/jetson-inference/python/training/classification/models/mrl/labels.txt"

    # Initialize video source and output
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

    eyes_closed_duration = 0  # Track the duration both eyes are closed
    close_start_time = None   # Start time when both eyes are closed

    try:
        while display.IsStreaming():
            img = camera.Capture()
            if img is None:
                utils.turn_off_all()  # No face detected, turn off all LEDs
                continue

            # Detect and classify the state of the eyes
            left_eye_state, right_eye_state, cropped_face, left_eye_img, right_eye_img = face_eye_detector.detect_and_classify(img)

            # LED control logic
            if left_eye_state == "open" and right_eye_state == "open":
                utils.turn_on_green()  # Turn on green LED
                close_start_time = None  # Reset closed duration
            elif left_eye_state == "close" and right_eye_state == "close":
                if close_start_time is None:
                    close_start_time = time.time()  # Mark when eyes closed

                eyes_closed_duration = time.time() - close_start_time
                if eyes_closed_duration >= 2:
                    utils.turn_on_red()  # Turn on red LED if closed for 3 seconds
                else:
                    utils.turn_on_yellow()  # Turn on yellow LED otherwise
            else:
                utils.turn_off_all()  # Turn off all LEDs

            # Synchronize GPU and render image
            cudaDeviceSynchronize()
            display.Render(img)
            #display.SetStatus(f"Face & Eye Detection | Network {face_eye_detector.face_detection.GetNetworkFPS():.0f} FPS")
            display.SetStatus(f"Face & Eye Detection | Vgg16 {face_eye_detector.face_detection.GetNetworkFPS():.0f} FPS")

            # Display images using OpenCV
            face_eye_detector.display_face_and_eyes(cropped_face, left_eye_img, right_eye_img)

            if cv2.waitKey(1) == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        utils.turn_off_all()  # Turn off all LEDs

    finally:
        # Perform cleanup
        face_eye_detector.cleanup()  # Cleanup face and eye detector resources
        cv2.destroyAllWindows()  # Ensure any OpenCV windows are closed
        utils.turn_off_all() # Turn off all LEDs
        utils.cleanup()  # Clean up GPIO on exit

if __name__ == "__main__":
    main()
