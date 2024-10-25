import os
import random
from PIL import Image

# Path to your dataset's image directories (update these as per your actual paths)
train_dir = 'train/'
val_dir = 'val/'
test_dir = 'test/'

# Function to get random images for a specific category (open or close)
def get_random_images_by_category(directory, category, count=3):
    category_path = os.path.join(directory, category)
    files = os.listdir(category_path)
    if len(files) < count:
        raise ValueError(f"Not enough images in category '{category}' of '{directory}'")
    random_images = random.sample(files, count)  # Select the specified number of random images
    return [os.path.join(category_path, img) for img in random_images]

# Load images, resize them to a fixed size, and create a collage
def create_collage(open_images, close_images, collage_path='collage.jpg', image_size=(128, 128)):
    images = []  # We will append the correct order of images here
    
    # Arrange the open-eye images in the first 3 columns and close-eye images in the last 3 columns for each row
    for i in range(3):  # 3 rows: train, val, test
        row = open_images[i*3:(i+1)*3] + close_images[i*3:(i+1)*3]  # First 3 are open, last 3 are close
        images.extend(row)
    
    # Set the size for the collage (6 images per row and 3 rows)
    collage_width = 6 * image_size[0]  # 6 images per row
    collage_height = 3 * image_size[1]  # 3 rows (train, val, test)
    
    # Create a blank canvas for the collage
    collage = Image.new('RGB', (collage_width, collage_height))
    
    # Paste the resized images into the collage in a 6x3 grid (first 3 columns for open, last 3 for close)
    for i, img in enumerate(images):
        x_offset = (i % 6) * image_size[0]
        y_offset = (i // 6) * image_size[1]
        img_resized = Image.open(img).resize(image_size)
        collage.paste(img_resized, (x_offset, y_offset))
    
    # Save the collage
    collage.save(collage_path)
    print(f"Collage saved at {collage_path}")

# Example usage:
# Get 3 open-eye and 3 close-eye images for each dataset (train, val, test)
train_open = get_random_images_by_category(train_dir, 'open', 3)
train_close = get_random_images_by_category(train_dir, 'close', 3)
val_open = get_random_images_by_category(val_dir, 'open', 3)
val_close = get_random_images_by_category(val_dir, 'close', 3)
test_open = get_random_images_by_category(test_dir, 'open', 3)
test_close = get_random_images_by_category(test_dir, 'close', 3)

# Combine images for collage (first 3 columns open, last 3 columns close)
open_images = train_open + val_open + test_open  # First 3 columns (open eyes)
close_images = train_close + val_close + test_close  # Last 3 columns (close eyes)

# Create the collage
create_collage(open_images, close_images)
