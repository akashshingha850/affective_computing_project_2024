import cv2
from face_eye_detection import FaceEyeDetector
from jetson_utils import videoSource, videoOutput, cudaDeviceSynchronize

def resize_eye_images(left_eye_img, right_eye_img):
    # Get the minimum height of the two images
    min_height = min(left_eye_img.shape[0], right_eye_img.shape[0])

    # Resize both images to have the same height (min_height)
    left_eye_img = cv2.resize(left_eye_img, (left_eye_img.shape[1], min_height))
    right_eye_img = cv2.resize(right_eye_img, (right_eye_img.shape[1], min_height))

    return left_eye_img, right_eye_img

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

        # Display the cropped face in a second window
        if cropped_face is not None:
            cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cv2.imshow("Face", cropped_face_rgb)

        # Combine the left and right eye images and display in a third window
        if left_eye_img is not None and right_eye_img is not None:
            left_eye_img_rgb = cv2.cvtColor(left_eye_img, cv2.COLOR_BGR2RGB)
            right_eye_img_rgb = cv2.cvtColor(right_eye_img, cv2.COLOR_BGR2RGB)
            # Ensure that the images have the same height and type before concatenating
            left_eye_img_rgb, right_eye_img_rgb = resize_eye_images(left_eye_img_rgb, right_eye_img_rgb)
            
            # Now concatenate the images
            eyes_combined = cv2.hconcat([left_eye_img_rgb, right_eye_img_rgb])  # Combine left and right eye images horizontally
            cv2.imshow("Eyes", eyes_combined)

        # Render the original image
        display.Render(img)
        display.SetStatus(f"Face & Eye Detection | Network {face_eye_detector.face_detection.GetNetworkFPS():.0f} FPS")

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
