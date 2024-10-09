from jetson_inference import detectNet, imageNet
from jetson_utils import videoSource, videoOutput, cudaAllocMapped, cudaCrop, cudaToNumpy
import cv2

# Initialize face detection and classification networks
face_net = detectNet("facedetect", threshold=0.5)
classification_net = imageNet(model="/home/jetson/jetson-inference/data/networks/sleep/resnet18.onnx", 
                              labels="/home/jetson/jetson-inference/data/networks/sleep/labels.txt", 
                              input_blob="input_0", 
                              output_blob="output_0")

camera = videoSource("/dev/video0")  # Camera source
display = videoOutput("display://0")  # Main display output

cropped_face_window_open = False

while display.IsStreaming():
    img = camera.Capture()

    if img is None:
        continue

    # Detect faces in the image
    detections = face_net.Detect(img, overlay="lines")
    
    face_detected = False

    for detection in detections:
        if face_net.GetClassDesc(detection.ClassID) == "face":
            face_detected = True

            print(f"Detected face with confidence {detection.Confidence:.2f}")

            # Define the crop ROI and crop the face
            crop_roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
            cropped_img = cudaAllocMapped(width=crop_roi[2] - crop_roi[0], height=crop_roi[3] - crop_roi[1], format=img.format)
            cudaCrop(img, cropped_img, crop_roi)

            # Pass the cropped image to the classification network
            class_idx, confidence = classification_net.Classify(cropped_img)
            class_desc = classification_net.GetClassDesc(class_idx)
            print(f"Classified face as '{class_desc}' with {confidence * 100:.2f}% confidence")

            # Convert the cropped CUDA image to NumPy for OpenCV
            np_img = cudaToNumpy(cropped_img)

            # Display the cropped face with classification result and confidence
            cv2.putText(np_img, f"{class_desc}: {confidence * 100:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Cropped Face", np_img)
            cropped_face_window_open = True

    # Keep the main display running
    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(face_net.GetNetworkFPS()))


    # If no face detected, close the cropped face window if it's open
    if not face_detected and cropped_face_window_open:
        cv2.destroyWindow("Cropped Face")
        cropped_face_window_open = False

    # Break on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
