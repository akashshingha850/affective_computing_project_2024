#!/usr/bin/env python3

from jetson_inference import imageNet
from jetson_utils import loadImage
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Paths to the folders containing images for "Sleepy" and "Awake" classes
sleepy_image_folder = "data/test/sleepy"
awake_image_folder = "data/test/awake"
model_folder = "model/mrl_vgg16"

# Extract model name from the model path
model_name = os.path.basename(model_folder)

# Ensure the model folder exists for saving results
os.makedirs(model_folder, exist_ok=True)

# Load the custom eye classification network (vgg16.onnx) with appropriate input/output blobs and label file
net = imageNet(model=f"{model_folder}/vgg16.onnx", 
               labels="/data/labels.txt", 
               input_blob="input_0", 
               output_blob="output_0")

# Track predictions and ground truth
predictions = []
ground_truth = []

# Classify images in the "Sleepy" folder
for filename in os.listdir(sleepy_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        image_path = os.path.join(sleepy_image_folder, filename)

        # Load the image (into shared CPU/GPU memory)
        img = loadImage(image_path)

        # Classify the image
        class_idx, confidence = net.Classify(img)

        # Track ground truth as "Sleepy" (label 0) and prediction
        ground_truth.append(0)  # 0 for "Sleepy"
        predictions.append(class_idx)

# Classify images in the "Awake" folder
for filename in os.listdir(awake_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
        image_path = os.path.join(awake_image_folder, filename)

        # Load the image (into shared CPU/GPU memory)
        img = loadImage(image_path)

        # Classify the image
        class_idx, confidence = net.Classify(img)

        # Track ground truth as "Awake" (label 1) and prediction
        ground_truth.append(1)  # 1 for "Awake"
        predictions.append(class_idx)

# Generate confusion matrix
conf_matrix = confusion_matrix(ground_truth, predictions)

# Print confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Print classification report
class_names = ["Sleepy", "Awake"]
classification_report_str = classification_report(ground_truth, predictions, target_names=class_names)
print("\nClassification Report:")
print(classification_report_str)

# Save the classification report with model name in the filename
classification_report_path = os.path.join(model_folder, f"classification_report_{model_name}.txt")
with open(classification_report_path, "w") as file:
    file.write(classification_report_str)

# Convert confusion matrix to a pandas DataFrame for visualization
conf_matrix_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)

# Plot and save confusion matrix as an image with model name in the filename
plt.figure(figsize=(4,4))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")

# Save the confusion matrix as an image
confusion_matrix_image_path = os.path.join(model_folder, f"confusion_matrix_{model_name}.png")
plt.savefig(confusion_matrix_image_path)

# Print confirmation
print(f"\nClassification report saved as {classification_report_path}")
print(f"Confusion matrix saved as {confusion_matrix_image_path}")
