"""
This file calls the all the necessary methods to train the Stable Diffusion model and then use it to generate images
"""

#s4701574
from dataset import *
from modules import *
from predict import *
from train import *




if __name__ == '__main__':

    #Specifying Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###################################################################################
    #Loading Data
    batch_size = 4 #Change if necessary
    data_train = "/home/groups/comp3710/ADNI/AD_NC/train" #Change if necessary
    data_test = "/home/groups/comp3710/ADNI/AD_NC/test" #Change if necessary
    dataloader = load_data(data_train, data_test, batch_size)

    ###################################################################################
    #Training VQVAE
    vqvae_trained = False #Change if you already have trained model
    vqvae_model = VQVAE(im_channels=3).to(device)
    if (vqvae_trained):
        vqvae_model.load_state_dict(torch.load("VQVAE_state_dict.pth"))
        vqvae_model.eval()
    else:
        train_vqvae(vqvae_model, dataloader, epochs=20, device=device, lr=0.00001)
        vqvae_model.eval()

    ###################################################################################
    #Training Diffusion Model
    diffusion_model_trained = False #Change if you already have trained model
    unetD = UNet().to(device)
    noise_scheduler = NoiseScheduler(timesteps=1000)
    diffusion_model = DiffusionModel(vqvae_model, unetD, noise_scheduler).to(device)
    if (diffusion_model_trained):
        diffusion_model.load_state_dict(torch.load("new_diffusion_model_state_dict.pth"))
        diffusion_model.eval()
    else:
        diffusion_optimizer = torch.optim.Adam(diffusion_model.unet.parameters(), lr= 0.0001)
        train_diffusion_model(diffusion_model, dataloader, diffusion_optimizer, epochs=23)
        diffusion_model.eval()

    ###################################################################################
    #Generate Images
    create_gif(diffusion_model)