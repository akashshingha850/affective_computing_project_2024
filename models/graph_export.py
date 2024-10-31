import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Load the TensorBoard logs using TFRecordDataset
logfile = 'models/mrl_vgg16/tensorboard/20241019_022420/events.out.tfevents.1729293860.ubuntu.4700.0'
dataset = tf.data.TFRecordDataset(logfile)

export_dir = 'models/mrl_vgg16'  # Change this to your model folder
text_file_path = os.path.join(export_dir, 'log_summary.txt')
# Ensure the export directory exists
os.makedirs(export_dir, exist_ok=True)

# Lists to store values
steps = []
train_loss = []
val_loss = []
train_accuracy = []
val_accuracy = []

for raw_record in dataset:
    event = tf.compat.v1.Event.FromString(raw_record.numpy())
    for value in event.summary.value:
        if value.tag == 'Loss/train':
            steps.append(event.step)
            train_loss.append(value.simple_value)
        elif value.tag == 'Loss/val':
            val_loss.append(value.simple_value)
        elif value.tag == 'Accuracy/train':
            train_accuracy.append(value.simple_value)
        elif value.tag == 'Accuracy/val':
            val_accuracy.append(value.simple_value)

# Plotting Loss Curves
plt.figure(figsize=(12,3))
plt.subplot(1, 2, 1)
plt.plot(steps, train_loss, label='Train Loss')
plt.plot(steps, val_loss, label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plotting Accuracy Curves
plt.subplot(1, 2, 2)
plt.plot(steps, train_accuracy, label='Train Accuracy')
plt.plot(steps, val_accuracy, label='Validation Accuracy')
plt.title('Accuracy Curves')
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Save the figures
plt.tight_layout()
plt.savefig(os.path.join(export_dir,'loss_accuracy_curves.png'))
plt.show()


# Write the collected data to a text file
with open(text_file_path, 'w') as f:
    f.write("Step, Train Loss, Validation Loss, Train Accuracy, Validation Accuracy\n")
    for i in range(len(steps)):
        f.write(f"{steps[i]}, {train_loss[i] if i < len(train_loss) else 'N/A'}, "
                 f"{val_loss[i] if i < len(val_loss) else 'N/A'}, "
                 f"{train_accuracy[i] if i < len(train_accuracy) else 'N/A'}, "
                 f"{val_accuracy[i] if i < len(val_accuracy) else 'N/A'}\n")

print(f'Data exported to {text_file_path}')