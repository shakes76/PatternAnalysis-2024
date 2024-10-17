import pickle
import matplotlib.pyplot as plt
import os
import torch

# Ensure save_dir exists
save_dir = "graphs"
os.makedirs(save_dir, exist_ok=True)

open_dir = 'data_viz/VQ-VAE.pkl'

# Load the saved data using pickle
with open(open_dir, 'rb') as f:
    data = pickle.load(f)

# Access the lists
training_reconstruction_loss = data["training_reconstruction_loss"]
training_vq_loss = data["training_vq_loss"]
validation_reconstruction_loss = data["validation_reconstruction_loss"]
validation_vq_loss = data["validation_vq_loss"]
ssim = data["ssim"]

epochs = 150

if epochs > len(training_reconstruction_loss) + 1:
    epochs = len(training_reconstruction_loss) + 1

# Now you can plot the graphs using these lists
epochs = range(1, epochs)
plt.figure(figsize=(12, 6))
plt.plot(epochs, training_reconstruction_loss, label="Training Reconstruction Loss")
plt.plot(epochs, validation_reconstruction_loss, label="Validation Reconstruction Loss")
plt.ylim(0, 0.1)  # Set the y-axis limits between 0 and 1
plt.xlabel('Epoch')
plt.ylabel('Training Reconstruction Loss')
plt.title('Training Reconstruction Loss over Epochs')
plt.legend()
plt.grid(True)

# Save the figure to save_dir
save_path = os.path.join(save_dir, 'training_reconstruction_loss.png')
plt.savefig(save_path)
plt.close()  # Close the plot to prevent it from displaying


# Plot Training and Validation VQ Loss
plt.figure(figsize=(12, 6))
plt.plot(epochs, training_vq_loss, label="Training VQ Loss")
plt.plot(epochs, validation_vq_loss, label="Validation VQ Loss")
plt.ylim(0, 50000) 
plt.xlabel('Epoch')
plt.ylabel('VQ Loss')
plt.title('VQ Loss over Epochs')
plt.legend()
plt.grid(True)

# Save the VQ Loss figure
save_path_vq = os.path.join(save_dir, 'vq_loss.png')
plt.savefig(save_path_vq)
plt.close()  # Close the plot to prevent displaying


# # Plot SSIM over epochs
# plt.figure(figsize=(12, 6))
# plt.plot(epochs, ssim, label="SSIM")
# # plt.ylim(0, 1)  # SSIM values typically range between 0 and 1
# plt.xlabel('Epoch')
# plt.ylabel('SSIM')
# plt.title('SSIM over Epochs')
# plt.legend()
# plt.grid(True)

# # Save the SSIM figure
# save_path_ssim = os.path.join(save_dir, 'ssim.png')
# plt.savefig(save_path_ssim)
# plt.close()  # Close the plot to prevent displaying