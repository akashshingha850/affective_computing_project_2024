import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time

# Paths to the folders containing images for "Close" and "Open" classes
close_image_folder = "data/test/close"
open_image_folder = "data/test/open"

# Load the VGG16 model architecture
model = models.vgg16()

# Modify the classifier to match the number of classes in your dataset (2 classes: "open" and "close")
model.classifier[6] = torch.nn.Linear(4096, 2)  # Change the final layer to output 2 classes

# Load the checkpoint
checkpoint = torch.load("models/mrl_vgg16/model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])  # Load model weights from the checkpoint

# Switch to evaluation mode
model.eval()

# Define transformation (similar to the one used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Track predictions, ground truth, and latency for PyTorch
predictions = []
ground_truth = []
latencies = []

# Classify images in the "Close" folder
for filename in os.listdir(close_image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(close_image_folder, filename)
        image = Image.open(image_path)
        
        # Ensure the image has 3 channels (convert grayscale to RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Measure the inference latency with PyTorch
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
        end_time = time.time()

        # Calculate and store latency
        latency = end_time - start_time
        latencies.append(latency)

        # Track predictions (you may need to adjust this based on your model output)
        _, class_idx = torch.max(output, 1)
        predictions.append(class_idx.item())
        ground_truth.append(0)  # Or 1 for "Open" images

# Calculate the average latency
average_latency_pytorch = sum(latencies) / len(latencies)
print(f"Average Latency for PyTorch model: {average_latency_pytorch:.4f} seconds")
