import pickle
import matplotlib.pyplot as plt
import os
import torch

# Ensure save_dir exists
save_dir = "data_viz"
os.makedirs(save_dir, exist_ok=True)

# Load the saved data using pickle
with open('training_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Access the lists
training_output_loss = data["training_output_loss"]
training_vq_loss = data["training_vq_loss"]
validation_output_loss = data["validation_output_loss"]
validation_vq_loss = data["validation_vq_loss"]
ssim = data["ssim"]

epochs = 150

if epochs > len(training_output_loss) + 1:
    epochs = len(training_output_loss) + 1

# Now you can plot the graphs using these lists
epochs = range(1, epochs)

plt.plot(epochs, training_output_loss, label="Training Output Loss")
plt.plot(epochs, validation_output_loss, label="Validation Output Loss")
# plt.ylim(0, 0.01)  # Set the y-axis limits between 0 and 1
plt.xlabel('Epoch')
plt.ylabel('Training Output Loss')
plt.title('Training Output Loss over Epochs')
plt.legend()

# Save the figure to save_dir
save_path = os.path.join(save_dir, 'training_output_loss.png')
plt.savefig(save_path)
plt.close()  # Close the plot to prevent it from displaying


# Plot Training and Validation VQ Loss
plt.plot(epochs, training_vq_loss, label="Training VQ Loss")
plt.plot(epochs, validation_vq_loss, label="Validation VQ Loss")
# plt.ylim(0, max(max(training_vq_loss), max(validation_vq_loss)))  # Dynamically set the y limit
plt.xlabel('Epoch')
plt.ylabel('VQ Loss')
plt.title('VQ Loss over Epochs')
plt.legend()

# Save the VQ Loss figure
save_path_vq = os.path.join(save_dir, 'vq_loss.png')
plt.savefig(save_path_vq)
plt.close()  # Close the plot to prevent displaying


# Plot SSIM over epochs
plt.plot(epochs, ssim, label="SSIM")
# plt.ylim(0, 1)  # SSIM values typically range between 0 and 1
plt.xlabel('Epoch')
plt.ylabel('SSIM')
plt.title('SSIM over Epochs')
plt.legend()

# Save the SSIM figure
save_path_ssim = os.path.join(save_dir, 'ssim.png')
plt.savefig(save_path_ssim)
plt.close()  # Close the plot to prevent displaying