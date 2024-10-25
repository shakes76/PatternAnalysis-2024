import torch
import numpy as np
import argparse
import utils
import matplotlib.pyplot as plt
from modules import VQVAE
import os
from skimage.metrics import structural_similarity as ssim

parser = argparse.ArgumentParser()

epochs = 1
learning_rate = 1e-3
batch_size = 16
weight_decay = 1e-5

n_hiddens = 512
n_residual_hiddens = 512
n_residual_layers = 32
embedding_dim = 512
n_embeddings = 1024
beta = 0.1


parser.add_argument("--dataset_dir", type=str, default='HipMRI_study_keras_slices_data')  # Directory for .nii test data
parser.add_argument("--save", type=str, default="vqvae_data.pth")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = VQVAE(n_hiddens, n_residual_hiddens, n_residual_layers,
                  n_embeddings, embedding_dim, beta).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    return model

def predict_and_reconstruct(model, data_loader):
    model.eval()  
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            _, x_hat, _ = model(x) 
            x = x.cpu().numpy()     
            x_hat = x_hat.cpu().numpy() 
            yield x, x_hat


def main():

    test_path = os.path.join(args.dataset_dir, 'keras_slices_test')
    nii_files_test = [os.path.join(test_path, img) for img in os.listdir(test_path) if img.endswith(('.nii', '.nii.gz'))]
    x_test = utils.load_data_2D(nii_files_test, normImage=False, categorical=False)
    x_test_tensor = torch.from_numpy(x_test).float().unsqueeze(1) 

    test_loader = torch.utils.data.DataLoader(x_test_tensor, batch_size=batch_size)


    path = 'results/' + args.save
    model = load_model(path)


    for original, reconstructed in predict_and_reconstruct(model, test_loader):
        print(f"Original shape: {original.shape}, Reconstructed shape: {reconstructed.shape}")
        

        for i in range(min(5, len(original))): 
            print("Shape of reconstructed image:", reconstructed[i].shape)

            fig, axs = plt.subplots(1, 2)


            original_img = np.squeeze(original[i], axis=0)
            axs[0].imshow(original_img, cmap='gray')
            axs[0].set_title('Original')


            if reconstructed[i].shape[0] == 1: 
                reconstructed_img = np.squeeze(reconstructed[i], axis=0)
            else:  
                reconstructed_img = np.mean(reconstructed[i], axis=0)  

            ssim_score = ssim(original_img, reconstructed_img, data_range=reconstructed_img.max() - reconstructed_img.min())
            
            axs[1].imshow(reconstructed_img, cmap='gray')
            axs[1].set_title(f'SSIM Score: {ssim_score:.4f}')
            plt.savefig(f"reconstructed_{i}")
            plt.show()
        break  

if __name__ == "__main__":
    main()
