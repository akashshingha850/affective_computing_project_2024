# face_eye_detection.py

import cv2
import mediapipe as mp
from jetson_inference import detectNet, imageNet
from jetson_utils import cudaAllocMapped, cudaCrop, cudaToNumpy

class FaceEyeDetector:
    def __init__(self, face_model, eye_model, labels, eye_margin_x=15, eye_margin_y=20):
        # Initialize face detection and eye classification networks
        self.face_detection = detectNet(face_model, threshold=0.5)
        self.eye_classification = imageNet(model=eye_model, 
                                           labels=labels, 
                                           input_blob="input_0", 
                                           output_blob="output_0")
        # Initialize MediaPipe face mesh for eye detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1)

        # Define margin for cropping eyes
        self.eye_margin_x = eye_margin_x
        self.eye_margin_y = eye_margin_y

    def detect_and_classify(self, img):
        # Convert the CUDA image to numpy for processing
        np_img = cudaToNumpy(img)
        rgb_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)

        # Detect faces using Jetson inference
        detections = self.face_detection.Detect(img, overlay="lines")

        left_eye_state, right_eye_state = "unknown", "unknown"
        cropped_face = None
        left_eye_img = None
        right_eye_img = None

        for detection in detections:
            if self.face_detection.GetClassDesc(detection.ClassID) == "face":
                # Get bounding box coordinates for the detected face
                left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)

                # Crop the face region for further processing with MediaPipe
                cropped_face = rgb_img[top:bottom, left:right]
                results = self.face_mesh.process(cropped_face)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        left_eye_coords, right_eye_coords = self.extract_eye_coordinates(face_landmarks, cropped_face, left, top)

                        if left_eye_coords:
                            left_eye_state, left_eye_img = self.classify_eye(left_eye_coords, img, np_img, is_left=True)

                        if right_eye_coords:
                            right_eye_state, right_eye_img = self.classify_eye(right_eye_coords, img, np_img, is_left=False)

        return left_eye_state, right_eye_state, cropped_face, left_eye_img, right_eye_img

    def extract_eye_coordinates(self, face_landmarks, face_roi, left_offset, top_offset):
        # Extract eye coordinates from face landmarks
        left_eye_coords = []
        right_eye_coords = []

        # Eye and eyebrow landmark indices
        eye_eyebrow_landmarks = [33, 133, 362, 263, 55, 46, 285, 276]
        for idx in eye_eyebrow_landmarks:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * face_roi.shape[1]) + left_offset
            y = int(landmark.y * face_roi.shape[0]) + top_offset

            if idx in [362, 263, 285, 276]:  # Left eye and eyebrow
                right_eye_coords.append((x, y))
            elif idx in [33, 133, 55, 46]:  # Right eye and eyebrow
                left_eye_coords.append((x, y))

        return left_eye_coords, right_eye_coords

    def classify_eye(self, eye_coords, img, np_img, is_left=True):
        # Define bounding boxes for the eyes
        x_min = max(min([coord[0] for coord in eye_coords]) - self.eye_margin_x, 0)
        x_max = min(max([coord[0] for coord in eye_coords]) + self.eye_margin_x, np_img.shape[1])
        y_min = max(min([coord[1] for coord in eye_coords]) - self.eye_margin_y, 0)
        y_max = min(max([coord[1] for coord in eye_coords]) + self.eye_margin_y, np_img.shape[0])

        # Ensure valid dimensions before cropping
        if x_max > x_min and y_max > y_min:
            eye_img = np_img[y_min:y_max, x_min:x_max]
            #eye_img = cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)
            #eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
            cropped_eye = cudaAllocMapped(width=eye_img.shape[1], height=eye_img.shape[0], format="rgb8")
            cudaCrop(img, cropped_eye, (x_min, y_min, x_max, y_max))

            # Classify the eye
            class_idx, confidence = self.eye_classification.Classify(cropped_eye)
            class_desc = self.eye_classification.GetClassDesc(class_idx)

            # Add classification result text to the eye image
            font_color = (255, 0, 0) if class_desc.lower() == "close" else (0, 255, 0)
            cv2.putText(eye_img, f"{class_desc}: {confidence * 100:.2f}%", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)

            return class_desc.lower(), eye_img

        return "unknown", None
