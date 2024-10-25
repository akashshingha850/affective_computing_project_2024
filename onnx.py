import onnxruntime as ort
import numpy as np
import os
import time
from PIL import Image
import torchvision.transforms as transforms

# Load the ONNX model
model_path = "models/mrl_vgg16/vgg16.onnx"
ort_session = ort.InferenceSession(model_path)

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Paths to test folders
test_folders = ["data/test/close", "data/test/open"]

# Run inference on all images in the test folder and measure latency
latencies = []
iterations = 0

for folder in test_folders:
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder, filename)
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

            # Prepare input for ONNX Runtime
            ort_inputs = {ort_session.get_inputs()[0].name: input_tensor.numpy()}

            # Measure the inference latency
            inference_start = time.time()
            ort_outs = ort_session.run(None, ort_inputs)
            inference_end = time.time()

            # Calculate and store latency
            latencies.append(inference_end - inference_start)
            iterations += 1

# Calculate average latency and FPS
average_latency_onnx = sum(latencies) / len(latencies)
fps_onnx = 1.0 / average_latency_onnx

print(f"ONNX Runtime - Average Latency: {average_latency_onnx:.6f} seconds")
print(f"ONNX Runtime - FPS: {fps_onnx:.2f}")
