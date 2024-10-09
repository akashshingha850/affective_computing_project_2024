import cv2
import mediapipe as mp
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaToNumpy
import pycuda.driver as cuda

# Load the face detection network
net = detectNet("facedetect", threshold=0.5)

# Initialize MediaPipe face mesh for eye and eyebrow detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Initialize video source (camera) and video output (display)
camera = videoSource("/dev/video0")  # '/dev/video0' for V4L2, csi://0 for CSI camera
display = videoOutput("display://0")  # Full face display window

# Define a larger margin to include the eyebrows
eye_margin_x = 15  # Horizontal margin
eye_margin_y = 20  # Vertical margin to include eyebrow

# Loop until the display stops streaming
while display.IsStreaming():
    # Capture image frame from camera
    img = camera.Capture()

    if img is None:
        continue  # Skip the frame if it's not captured

    # Convert the image to numpy array for processing
    np_img = cudaToNumpy(img)  # Convert CUDA image to numpy
    rgb_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)  # Convert to RGB for MediaPipe

    # Run face detection using Jetson inference
    detections = net.Detect(img, overlay="none")  # We want to render the face without extra overlays

    # Process each detected face
    for detection in detections:
        # Get bounding box coordinates (left, top, right, bottom)
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

        # Crop the face region for MediaPipe
        face_roi = rgb_img[top:bottom, left:right]

        # Process the face region with MediaPipe to detect facial landmarks (including eyes and eyebrows)
        results = face_mesh.process(face_roi)

        # Check if landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Define empty lists to hold eye and eyebrow coordinates
                left_eye_coords = []
                right_eye_coords = []

                # Eye and eyebrow landmark indices for MediaPipe:
                # 33, 133 -> left eye, 362, 263 -> right eye, 55, 46 -> left eyebrow, 285, 276 -> right eyebrow
                eye_eyebrow_landmarks = [33, 133, 362, 263, 55, 46, 285, 276]
                for landmark_idx in eye_eyebrow_landmarks:
                    landmark = face_landmarks.landmark[landmark_idx]
                    x = int(landmark.x * face_roi.shape[1]) + left  # Scale and shift x-coordinates
                    y = int(landmark.y * face_roi.shape[0]) + top   # Scale and shift y-coordinates

                    # Draw red dot on eye and eyebrow landmarks
                    cv2.circle(np_img, (x, y), 3, (0, 0, 255), -1)  # Red dot on each landmark

                    # Store the coordinates of left and right eye landmarks
                    if landmark_idx in [33, 133, 55, 46]:  # Left eye + eyebrow
                        left_eye_coords.append((x, y))
                    elif landmark_idx in [362, 263, 285, 276]:  # Right eye + eyebrow
                        right_eye_coords.append((x, y))

                # Define bounding boxes for left and right eyes (including eyebrows) based on their coordinates
                if left_eye_coords and right_eye_coords:
                    # For left eye + eyebrow
                    lx_min = max(min([coord[0] for coord in left_eye_coords]) - eye_margin_x, 0)
                    lx_max = min(max([coord[0] for coord in left_eye_coords]) + eye_margin_x, np_img.shape[1])
                    ly_min = max(min([coord[1] for coord in left_eye_coords]) - eye_margin_y, 0)
                    ly_max = min(max([coord[1] for coord in left_eye_coords]) + eye_margin_y, np_img.shape[0])
                    left_eye = np_img[ly_min:ly_max, lx_min:lx_max]

                    # For right eye + eyebrow
                    rx_min = max(min([coord[0] for coord in right_eye_coords]) - eye_margin_x, 0)
                    rx_max = min(max([coord[0] for coord in right_eye_coords]) + eye_margin_x, np_img.shape[1])
                    ry_min = max(min([coord[1] for coord in right_eye_coords]) - eye_margin_y, 0)
                    ry_max = min(max([coord[1] for coord in right_eye_coords]) + eye_margin_y, np_img.shape[0])
                    right_eye = np_img[ry_min:ry_max, rx_min:rx_max]

                    # Draw bounding boxes around the eyes and eyebrows on the main image
                    cv2.rectangle(np_img, (lx_min, ly_min), (lx_max, ly_max), (255, 0, 0), 2)  # Left eye + eyebrow
                    cv2.rectangle(np_img, (rx_min, ry_min), (rx_max, ry_max), (255, 0, 0), 2)  # Right eye + eyebrow

                    # Ensure that both eyes are valid and not empty before displaying
                    if left_eye.size > 0:
                        # Display the left eye + eyebrow in a separate OpenCV window
                        cv2.imshow("Left Eye and Eyebrow", left_eye)

                    if right_eye.size > 0:
                        # Display the right eye + eyebrow in a separate OpenCV window
                        cv2.imshow("Right Eye and Eyebrow", right_eye)

    # Render the full face with eye and eyebrow bounding boxes and red dots on the main display
    display.Render(img)

    # Wait for the 'q' key to be pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release OpenCV windows
cv2.destroyAllWindows()
