#!/usr/bin/env python3

from jetson_inference import imageNet
from jetson_utils import loadImage
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Paths to the folders containing images for "Close" and "Open" classes
close_image_folder = "/home/jetson/jetson-inference/python/training/classification/data/mrl/test/close"
open_image_folder = "/home/jetson/jetson-inference/python/training/classification/data/mrl/test/open"
model_folder = "/home/jetson/jetson-inference/python/training/classification/models/mrl_googlenet/"

# Ensure the model folder exists for saving results
os.makedirs(model_folder, exist_ok=True)

# Load the custom eye classification network (resnet18.onnx) with appropriate input/output blobs and label file
net = imageNet(model="/home/jetson/jetson-inference/python/training/classification/models/mrl_googlenet/googlenet.onnx", 
               labels="/home/jetson/jetson-inference/python/training/classification/data/mrl/labels.txt", 
               input_blob="input_0", 
               output_blob="output_0")

# Track predictions and ground truth
predictions = []
ground_truth = []

# Classify images in the "Close" folder
for filename in os.listdir(close_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        image_path = os.path.join(close_image_folder, filename)

        # Load the image (into shared CPU/GPU memory)
        img = loadImage(image_path)

        # Classify the image
        class_idx, confidence = net.Classify(img)

        # Track ground truth as "Close" (label 0) and prediction
        ground_truth.append(0)  # 0 for "Close"
        predictions.append(class_idx)

# Classify images in the "Open" folder
for filename in os.listdir(open_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        image_path = os.path.join(open_image_folder, filename)

        # Load the image (into shared CPU/GPU memory)
        img = loadImage(image_path)

        # Classify the image
        class_idx, confidence = net.Classify(img)

        # Track ground truth as "Open" (label 1) and prediction
        ground_truth.append(1)  # 1 for "Open"
        predictions.append(class_idx)

# Generate confusion matrix
conf_matrix = confusion_matrix(ground_truth, predictions)

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
class_names = ["Close", "Open"]
classification_report_str = classification_report(ground_truth, predictions, target_names=class_names)
print("\nClassification Report:")
print(classification_report_str)

# Save the classification report as a text file
classification_report_path = os.path.join(model_folder, "classification_report.txt")
with open(classification_report_path, "w") as file:
    file.write(classification_report_str)

# Convert confusion matrix to a pandas DataFrame for visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Plot and save confusion matrix as an image
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")

# Save the confusion matrix as an image
confusion_matrix_image_path = os.path.join(model_folder, "confusion_matrix_image.png")
plt.savefig(confusion_matrix_image_path)

# Print confirmation
print(f"\nClassification report saved as {classification_report_path}")
print(f"Confusion matrix saved as {confusion_matrix_image_path}")
