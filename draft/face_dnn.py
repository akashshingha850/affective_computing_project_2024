from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

net = detectNet("facedetect", threshold=0.5)  # ssd-mobilenet-v2 for object detection  // facedetect for face detection
camera = videoSource("/dev/video0")      # '/dev/video0' for V4L2 csi://0 for csi camera
display = videoOutput("display://0") # 'my_video.mp4' for file

while display.IsStreaming():
    img = camera.Capture()

    if img is None:  # capture timeout
        continue

    detections = net.Detect(img)
    
    for detection in detections:
        class_id = detection.ClassID
        class_name = net.GetClassDesc(class_id)
        print(f"Detected {class_name} with confidence {detection.Confidence:.2f}")

    display.Render(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))