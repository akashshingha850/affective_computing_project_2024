import tensorflow as tf
import os

# Absolute path to the TensorBoard logs directory
logdir = 'C:/Users/abakash/OneDrive - Teknologian Tutkimuskeskus VTT/Documents/Github/affective_computing_project/models/mrl_vgg16/tensorboard'

# Directory to save exported images
export_dir = 'C:/Users/abakash/OneDrive - Teknologian Tutkimuskeskus VTT/Documents/Github/affective_computing_project/models/mrl_vgg16/exported_images'
os.makedirs(export_dir, exist_ok=True)

# Load the TensorBoard logs
for event in tf.compat.v1.train.summary_iterator(logdir):
    for value in event.summary.value:
        if value.tag == 'image_tag':  # Replace 'image_tag' with your actual image tag
            img_tensor = tf.image.decode_image(value.image.encoded_image_string)
            img_path = os.path.join(export_dir, f'image_{event.step}.png')
            tf.io.write_file(img_path, tf.image.encode_png(img_tensor))

print(f'Images exported to {export_dir}')