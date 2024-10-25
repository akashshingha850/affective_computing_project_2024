import os
import time
from jetson_inference import imageNet
from jetson_utils import loadImage

# Load the TensorRT model
model_folder = "models/mrl_vgg16"
net = imageNet(model=f"{model_folder}/vgg16.onnx", 
               labels="data/labels.txt", 
               input_blob="input_0", 
               output_blob="output_0")

# Paths to test folders
test_folders = ["data/test/close", "data/test/open"]

# Run inference on all images in the test folder and measure latency
latencies = []
iterations = 0

for folder in test_folders:
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            img = loadImage(image_path)

            # Measure the inference latency
            inference_start = time.time()
            class_idx, confidence = net.Classify(img)
            inference_end = time.time()

            # Calculate and store latency
            latencies.append(inference_end - inference_start)
            iterations += 1

# Calculate average latency and FPS
average_latency_trt = sum(latencies) / len(latencies)
fps_trt = 1.0 / average_latency_trt

print(f"TensorRT - Average Latency: {average_latency_trt:.6f} seconds")
print(f"TensorRT - FPS: {fps_trt:.2f}")
