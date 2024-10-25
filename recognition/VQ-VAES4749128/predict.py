import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from modules import VQVAE,GatedPixelCNN
from dataset import GrayscaleImageDataset
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import nibabel as nib


def __read_nifti__(self, filepath):
        niftiImage = nib.load(filepath)
        inImage = niftiImage.get_fdata(caching='unchanged')
        dtype=np.float32
        inImage = inImage.astype(dtype)
        inImage = 255.0 * (inImage - inImage.min()) / inImage.ptp()
        return inImage

def predict(args):
    """
    Outputs images for VQ-VAE model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load trained  vq-vae model
    checkpoint = torch.load(args.checkpoint)
    param = checkpoint['hyperparameters']
    model = VQVAE(param['n_hiddens'], param['n_residual_hiddens'],
                param['n_residual_layers'], param['n_embeddings'], param['embedding_dim'], param['beta']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # predicts all input model and saves the result into a file
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),                        
    ])
    dataset = GrayscaleImageDataset(image_dir=args.test_dir, transform=transform)
    score =[]

    for index,test_img in enumerate(dataset):
        img_test = test_img.to(device)  # c, h, w
        img_test = img_test.unsqueeze(0)  # 1, c, h, w

        _, img_reconstruct, _ = model(img_test)
        img_reconstruct = img_reconstruct.squeeze().detach().cpu().numpy()
        ind_score = (ssim(img_test.cpu().squeeze().numpy(), img_reconstruct, data_range=img_reconstruct.max() - img_reconstruct.min()))
        score.append(ind_score)

        save_image(index, ind_score, img_test, img_reconstruct, args.save_dir)
    
    print(f'Average SSIM: {np.mean(score)}')


def save_image(index, ind_score, img_test, img_reconstruct, save_folder):
    """
    for saving VQ-VAE results alongside SSIMM score
    param index : the images' index,
    param ind_score: teh image's SSIM score 
    param img_test: the original input image
    param img_reconstruct: the output image
    param save_folder: the path to the folder that image is saved in
    """
    # plot and save reconstructed vs original image
    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 
    # folder = 'saved_images'
    os.makedirs(save_folder, exist_ok=True)
    filename = f'{index}_SSIM:{ind_score}.png'

    axes[0].imshow(img_test.cpu().squeeze().numpy(), cmap='gray')
    axes[0].set_title("Original")

    axes[1].imshow(img_reconstruct, cmap='gray')
    axes[1].set_title("reconstruct")

    plt.title(f'Score: {ind_score}')

    for ax in axes:
        ax.axis('off')
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path)
    # Display the plot
    plt.show()



def generate():
    """
    Generates image using PIXELCNN and returns output 
    """
    #Load previously trained vq-vae model
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load trained model
    checkpoint = torch.load(args.checkpoint)
    param = checkpoint['hyperparameters']
    model = VQVAE(param['n_hiddens'], param['n_residual_hiddens'],
                param['n_residual_layers'], param['n_embeddings'], param['embedding_dim'], param['beta']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    #Transform images

    transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 128)),
    transforms.ToTensor()
    ])
    #Loading pixel cnn model
    ckpt = torch.load(args.pixelcnn_checkpoint)
    pixelcnn = GatedPixelCNN(512, 32*32, 10).to(device)
    pixelcnn.load_state_dict(ckpt)
    pixelcnn.eval()
    #create latent code
    label = torch.zeros(1).long().to(device)
    sample = pixelcnn.generate(label, shape=(32, 32), batch_size=1)
    min_encoding_indices = sample.view(-1).unsqueeze(1)
    min_encodings = torch.zeros(
                min_encoding_indices.shape[0], model.vector_quantization.n_e).to(device)
    min_encodings.scatter_(1, min_encoding_indices, 1)
    # Decode quantised output
    z_q = torch.matmul(min_encodings, model.vector_quantization.embedding.weight).view([1, 32, 32, 64])
    z_q = z_q.permute(0, 3, 1, 2).contiguous()
    x_hat = model.decoder(z_q)
    #Visualise generated image
    img_reconstruct = x_hat.squeeze().detach().cpu().numpy()
    transformed_image = transform(img_reconstruct)
    transformed_image_np = transformed_image.numpy().squeeze()  # Remove channel dimension for grayscale
    plt.imshow(transformed_image_np, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  type=str, default='mri')
    parser.add_argument('--train_dir', type=str, default='/Selena/Comp3710_A3/HipMRI_study_keras_slices_data/keras_slices_train')
    parser.add_argument('--test_dir', type=str, default='/Selena/Comp3710_A3/HipMRI_study_keras_slices_data/keras_slices_test')
    parser.add_argument('--save_dir', type=str, default='/saved_images_reshaped')
    parser.add_argument("--checkpoint", type=str, default='/Selena/Vq_vae_comp3710/results/vqvae_data_fri_oct_25_16_29_40_2024.pth')
    parser.add_argument("-- pixelcnn_checkpoint", type=str, default='/Selena/Vq_vae_comp3710/results/vqvae_data_fri_oct_25_16_29_40_2024.pth')


    args = parser.parse_args()
    predict(args)

