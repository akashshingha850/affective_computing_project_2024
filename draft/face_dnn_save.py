from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, cudaToNumpy, cudaAllocMapped, cudaCrop
import cv2
import os

# Create a directory to save face images
output_dir = "detected_faces"
os.makedirs(output_dir, exist_ok=True)

net = detectNet("facedetect", threshold=0.5)  # ssd-mobilenet-v2 for object detection  // facedetect for face detection
camera = videoSource("/dev/video0")      # '/dev/video0' for V4L2 csi://0 for csi camera
display = videoOutput("display://0") # 'my_video.mp4' for file

frame_count = 0

while display.IsStreaming():
    img = camera.Capture()

    if img is None:  # capture timeout
        continue

    detections = net.Detect(img)
    
    for detection in detections:
        class_id = detection.ClassID
        class_name = net.GetClassDesc(class_id)
        print(f"Detected {class_name} with confidence {detection.Confidence:.2f}")

        if class_name == "face":  # Save only face detections
            # Compute the ROI as (left, top, right, bottom)
            crop_roi = (int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom))
            
            # Allocate the output image with the cropped size
            imgOutput = cudaAllocMapped(width=crop_roi[2] - crop_roi[0],
                                        height=crop_roi[3] - crop_roi[1],
                                        format=img.format)
            
            # Crop the image to the ROI
            cudaCrop(img, imgOutput, crop_roi)
            
            # Convert the cropped CUDA image to a NumPy array
            np_img = cudaToNumpy(imgOutput)
            
            # Save the face image
            face_filename = os.path.join(output_dir, f"face_{frame_count}.jpg")
            #cv2.imwrite(face_filename, np_img)
            frame_count += 1

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))