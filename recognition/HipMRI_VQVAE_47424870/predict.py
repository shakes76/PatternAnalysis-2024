import torch
import matplotlib.pyplot as plt
from modules import Encoder, Decoder
from dataset import get_data_loader

# Load the trained model
encoder = Encoder(in_channels=1, num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32)
decoder = Decoder(in_channels=128, out_channels=1)
encoder.load_state_dict(torch.load('encoder.pth'))
decoder.load_state_dict(torch.load('decoder.pth'))
encoder.eval()
decoder.eval()

# Load test data
image_dir = ''
test_loader = get_data_loader(image_dir, batch_size=1, shuffle=False)

# Predict and visualize results
with torch.no_grad():
    for i, image in enumerate(test_loader):
        encoded = encoder(image)
        decoded = decoder(encoded)

        # Visualize the original and reconstructed images
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image.squeeze(0).squeeze(0).numpy(), cmap='gray')
        
        plt.subplot(1, 2, 2)
        plt.title('Reconstructed Image')
        plt.imshow(decoded.squeeze(0).squeeze(0).numpy(), cmap='gray')
        
        plt.show()
        if i == 5:  # Visualize 5 test images
            break