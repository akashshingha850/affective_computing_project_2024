#!/usr/bin/env python3

from jetson_inference import imageNet
from jetson_utils import loadImage
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import time  # Import the time module

# Paths to the folders containing images for "Sleepy" and "Awake" classes
close_image_folder = "data/test/close"
open_image_folder = "data/test/open"
model_folder = "models/mrl_vgg19"

# Extract model name from the model path
model_name = os.path.basename(model_folder)

# Ensure the model folder exists for saving results
os.makedirs(model_folder, exist_ok=True)

# Load the custom eye classification network (vgg16.onnx) with appropriate input/output blobs and label file
net = imageNet(model=f"{model_folder}/vgg19.onnx", 
               labels="data/labels.txt", 
               input_blob="input_0", 
               output_blob="output_0")

# Track predictions, ground truth, and latency
predictions = []
ground_truth = []
latencies = []  # List to store latency values

# Classify images in the "Sleepy" folder
for filename in os.listdir(close_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(close_image_folder, filename)
        img = loadImage(image_path)

        # Measure the classification latency
        start_time = time.time()
        class_idx, confidence = net.Classify(img)
        end_time = time.time()
        
        # Calculate and store latency
        latency = end_time - start_time
        latencies.append(latency)

        # Track ground truth as "Sleepy" (label 0) and prediction
        ground_truth.append(0)  # 0 for "Sleepy"
        predictions.append(class_idx)

# Classify images in the "Awake" folder
for filename in os.listdir(open_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(open_image_folder, filename)
        img = loadImage(image_path)

        # Measure the classification latency
        start_time = time.time()
        class_idx, confidence = net.Classify(img)
        end_time = time.time()
        
        # Calculate and store latency
        latency = end_time - start_time
        latencies.append(latency)

        # Track ground truth as "Awake" (label 1) and prediction
        ground_truth.append(1)  # 1 for "Awake"
        predictions.append(class_idx)

# Calculate and print average latency
average_latency = sum(latencies) / len(latencies)
print(f"\nAverage Latency per Image: {average_latency:.4f} seconds")

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(ground_truth, predictions)
print("\nConfusion Matrix:")
print(conf_matrix)

class_names = ["close", "open"]
classification_report_str = classification_report(ground_truth, predictions, target_names=class_names)
print("\nClassification Report:")
print(classification_report_str)

# Save classification report and confusion matrix
classification_report_path = os.path.join(model_folder, f"classification_report_{model_name}.txt")
with open(classification_report_path, "w") as file:
    file.write(classification_report_str)

# Convert and plot confusion matrix
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(4,4))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
confusion_matrix_image_path = os.path.join(model_folder, f"confusion_matrix_{model_name}.png")
plt.savefig(confusion_matrix_image_path)

print(f"\nClassification report saved as {classification_report_path}")
print(f"Confusion matrix saved as {confusion_matrix_image_path}")
