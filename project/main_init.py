import cv2
import mediapipe as mp
from jetson_inference import detectNet, imageNet
from jetson_utils import videoSource, videoOutput, cudaAllocMapped, cudaCrop, cudaToNumpy

# Initialize face detection and eye classification networks
face_detection = detectNet("facedetect", threshold=0.5)
eye_classification = imageNet(model="/home/jetson/jetson-inference/data/networks/sleep/resnet18.onnx", 
                              labels="/home/jetson/jetson-inference/data/networks/sleep/labels.txt", 
                              input_blob="input_0", 
                              output_blob="output_0")

# Initialize MediaPipe face mesh for eye detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Initialize video source (camera) and video output (display)
camera = videoSource("/dev/video1")
display = videoOutput("display://0")

# Define margin for cropping eyes
eye_margin_x = 15
eye_margin_y = 20

while display.IsStreaming():
    img = camera.Capture()

    if img is None:
        continue

    # Convert the CUDA image to numpy for processing
    np_img = cudaToNumpy(img)
    rgb_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)

    # Detect faces using Jetson inference
    detections = face_detection.Detect(img, overlay="lines")

    for detection in detections:
        if face_detection.GetClassDesc(detection.ClassID) == "face":
            # Get bounding box coordinates for the detected face
            left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

            # Crop the face region for further processing with MediaPipe
            face_roi = rgb_img[top:bottom, left:right]
            results = face_mesh.process(face_roi)

            # Check if landmarks are detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye_coords = []
                    right_eye_coords = []

                    # Eye and eyebrow landmark indices
                    eye_eyebrow_landmarks = [33, 133, 362, 263, 55, 46, 285, 276]
                    for landmark_idx in eye_eyebrow_landmarks:
                        landmark = face_landmarks.landmark[landmark_idx]
                        x = int(landmark.x * face_roi.shape[1]) + left
                        y = int(landmark.y * face_roi.shape[0]) + top

                        # Store left and right eye coordinates
                        if landmark_idx in [362, 263, 285, 276]:  # Left eye and eyebrow
                            left_eye_coords.append((x, y))
                        elif landmark_idx in [33, 133, 55, 46]:  # Right eye and eyebrow
                            right_eye_coords.append((x, y))

                    # Define bounding boxes for the eyes
                    if left_eye_coords:
                        lx_min = max(min([coord[0] for coord in left_eye_coords]) - eye_margin_x, 0)
                        lx_max = min(max([coord[0] for coord in left_eye_coords]) + eye_margin_x, np_img.shape[1])
                        ly_min = max(min([coord[1] for coord in left_eye_coords]) - eye_margin_y, 0)
                        ly_max = min(max([coord[1] for coord in left_eye_coords]) + eye_margin_y, np_img.shape[0])

                        # Ensure valid dimensions before cropping
                        if lx_max > lx_min and ly_max > ly_min:
                            left_eye = np_img[ly_min:ly_max, lx_min:lx_max]
                            cropped_left_eye = cudaAllocMapped(width=left_eye.shape[1], height=left_eye.shape[0], format="rgb8")
                            cudaCrop(img, cropped_left_eye, (lx_min, ly_min, lx_max, ly_max))

                            # Classify the left eye
                            class_idx_left, confidence_left = eye_classification.Classify(cropped_left_eye)
                            class_desc_left = eye_classification.GetClassDesc(class_idx_left)
                            print(f"Classified left eye as '{class_desc_left}' with {confidence_left * 100:.2f}% confidence")

                            # Set font color to red if the eye is classified as 'closed', else green
                            font_color_left = (255, 0, 0) if class_desc_left.lower() == "close" else (0, 255, 0)

                            # Add classification result text to the left eye image
                            cv2.putText(left_eye, f"{class_desc_left}: {confidence_left * 100:.2f}%", (10, 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color_left, 2)

                    if right_eye_coords:
                        rx_min = max(min([coord[0] for coord in right_eye_coords]) - eye_margin_x, 0)
                        rx_max = min(max([coord[0] for coord in right_eye_coords]) + eye_margin_x, np_img.shape[1])
                        ry_min = max(min([coord[1] for coord in right_eye_coords]) - eye_margin_y, 0)
                        ry_max = min(max([coord[1] for coord in right_eye_coords]) + eye_margin_y, np_img.shape[0])

                        # Ensure valid dimensions before cropping
                        if rx_max > rx_min and ry_max > ry_min:
                            right_eye = np_img[ry_min:ry_max, rx_min:rx_max]
                            cropped_right_eye = cudaAllocMapped(width=right_eye.shape[1], height=right_eye.shape[0], format="rgb8")
                            cudaCrop(img, cropped_right_eye, (rx_min, ry_min, rx_max, ry_max))

                            # Classify the right eye
                            class_idx_right, confidence_right = eye_classification.Classify(cropped_right_eye)
                            class_desc_right = eye_classification.GetClassDesc(class_idx_right)
                            print(f"Classified right eye as '{class_desc_right}' with {confidence_right * 100:.2f}% confidence")

                            # Add classification result text to the right eye image
                            font_color_right = (255, 0, 0) if class_desc_right.lower() == "close" else (0, 255, 0)
                            cv2.putText(right_eye, f"{class_desc_right}: {confidence_right * 100:.2f}%", (10, 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color_right, 2)

    # Render the original image with bounding boxes
    display.Render(img)
    display.SetStatus("Face & Eye Detection | Network {:.0f} FPS".format(face_detection.GetNetworkFPS()))

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release OpenCV windows
cv2.destroyAllWindows()
