import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from modules import VQVAE
from dataset import GrayscaleImageDataset
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

checkpoint = torch.load('results/vqvae_data_fri_oct_18_17_11_15_2024.pth')
param = checkpoint['hyperparameters']
model = VQVAE(param['n_hiddens'], param['n_residual_hiddens'],
              param['n_residual_layers'], param['n_embeddings'], param['embedding_dim'], param['beta']).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()
test_dir = 'HipMRI_study_keras_slices_data/keras_slices_test_png'
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),                        
])
dataset = GrayscaleImageDataset(image_dir=test_dir, transform=transform)
score =[]

for index,test_img in enumerate(dataset):
    img_test = test_img.to(device)  # c, h, w
    img_test = img_test.unsqueeze(0)  # 1, c, h, w

    _, img_reconstruct, _ = model(img_test)
    img_reconstruct = img_reconstruct.squeeze().detach().cpu().numpy()
    ind_score = (ssim(img_test.cpu().squeeze().numpy(), img_reconstruct, data_range=img_reconstruct.max() - img_reconstruct.min()))
    score.append(ind_score)


    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    folder = 'saved_images'
    os.makedirs(folder, exist_ok=True)
    filename = f'{index}_SSIM:{ind_score}.png'
    # Display image1
    axes[0].imshow(img_test.cpu().squeeze().numpy(), cmap='gray')
    axes[0].set_title("Original")

    # Display image2
    axes[1].imshow(img_reconstruct, cmap='gray')
    axes[1].set_title("reconstruct")

    plt.title(f'Score: {ind_score}')

    for ax in axes:
        ax.axis('off')
    save_path = os.path.join(folder, filename)
    plt.savefig(save_path)
    # Display the plot
    plt.show()
print(f'Average SSIM: {np.mean(score)}')