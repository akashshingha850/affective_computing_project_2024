import dlib
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaToNumpy
import cv2

# Load the face detection network
net = detectNet("facedetect", threshold=0.5)

# Initialize video source (camera) and video output (display)
camera = videoSource("/dev/video0")  # '/dev/video0' for V4L2, csi://0 for CSI camera
display = videoOutput("display://0")  # 'display://0' for screen display

# Initialize dlib's face landmark predictor
# Make sure to have the 'shape_predictor_68_face_landmarks.dat' in the same directory or provide the path to it
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_predictor = dlib.shape_predictor(predictor_path)

# Initialize dlib's face detector (use detectNet for main face detection)
dlib_face_detector = dlib.get_frontal_face_detector()

# Loop until the display stops streaming
while display.IsStreaming():
    # Capture image frame from camera
    img = camera.Capture()

    # Check for capture timeout or any issue
    if img is None:
        continue  # Skip the frame if it's not captured

    # Convert the image to numpy array for dlib to work with
    np_img = cudaToNumpy(img)  # Convert CUDA image to numpy (needed for dlib)

    # Convert the numpy image from RGBA (default) to grayscale for dlib detection
    gray_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2GRAY)

    # Run face detection using Jetson inference
    detections = net.Detect(img, overlay="box")  # Overlay can be customized as "box", "label", "none"

    # Loop through the detected faces by Jetson
    for detection in detections:
        # Get bounding box coordinates (left, top, right, bottom)
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

        # Convert the bounding box to dlib rectangle format for landmark detection
        dlib_rect = dlib.rectangle(left, top, right, bottom)

        # Detect facial landmarks using dlib's shape predictor
        landmarks = face_predictor(gray_img, dlib_rect)

        # Extract the coordinates of the eyes (landmark points: 36-41 for the left eye, 42-47 for the right eye)
        left_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

        # Draw circles around the detected eye points for visualization
        for (x, y) in left_eye_points:
            cv2.circle(np_img, (x, y), 2, (255, 0, 0), -1)  # Left eye points (blue)

        for (x, y) in right_eye_points:
            cv2.circle(np_img, (x, y), 2, (0, 255, 0), -1)  # Right eye points (green)

        # Print face detection info
        class_id = detection.ClassID
        class_name = net.GetClassDesc(class_id)
        print(f"Detected {class_name} with confidence {detection.Confidence:.2f}")

    # Render the frame with face and eye detection to the display
    display.Render(img)

    # Display the current status including the FPS of the network
    display.SetStatus("Face and Eye Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
