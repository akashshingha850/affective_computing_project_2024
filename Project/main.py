import cv2
from face_eye_detection import FaceEyeDetector
from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize

def main():
    # Initialize variables and settings that are adjustable
    eye_margin_x = 15
    eye_margin_y = 20
    face_model = "facedetect"
    eye_model = "/home/jetson/jetson-inference/data/networks/sleep/resnet18.onnx"
    labels = "/home/jetson/jetson-inference/data/networks/sleep/labels.txt"

    # Initialize video source (camera) and video output (display)
    camera = videoSource("/dev/video1")
    display = videoOutput("display://0")

    # Initialize face and eye detector with adjustable parameters
    face_eye_detector = FaceEyeDetector(
        eye_margin_x=eye_margin_x,
        eye_margin_y=eye_margin_y,
        face_model=face_model,
        eye_model=eye_model,
        labels=labels
    )

    while display.IsStreaming():
        img = camera.Capture()

        if img is None:
            continue

        # Detect and classify the state of the eyes
        left_eye_state, right_eye_state, cropped_face, left_eye_img, right_eye_img = face_eye_detector.detect_and_classify(img)

        # Synchronize to ensure GPU is finished before rendering
        cudaDeviceSynchronize()

        # Display the cropped face and combined eyes
        face_eye_detector.display_face_and_eyes(cropped_face, left_eye_img, right_eye_img)

        # Render the original image
        display.Render(img)
        display.SetStatus(f"Face & Eye Detection | Network {face_eye_detector.face_detection.GetNetworkFPS():.0f} FPS")

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
