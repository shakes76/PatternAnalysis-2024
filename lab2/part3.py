'''
A implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) designed for 256x256 greyscale images.
References:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://github.com/indiradutta/DC_GAN
https://arxiv.org/abs/1511.06434
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import utils as vutils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt

# Constants
IMAGE_SIZE = 256
LATENT_DIM = 256
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 12
BETA1 = 0.5
NGPU = 1 # num GPUs
NGF = 256 # Num generator filters
NDF = 256 # Num discriminator filters
NC = 1 # Num channels


# Custom Dataset class for grayscale images
class GrayscaleImageDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.image_paths = []

        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1] normalisation
        ])

        # Collect image paths
        image_folder = os.path.join(root_dir, f'keras_png_slices_{split}')
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith('.nii.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        image = self.transform(image)
        return image


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.ngpu = NGPU
        self.main = nn.Sequential(
            # Input layer - Input size: LATENT_DIM x 1 x 1
            nn.ConvTranspose2d(in_channels=LATENT_DIM, out_channels=NGF*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=NGF*8),
            nn.ReLU(inplace=True),
            
            # Upsampling layers - Input size: 2048 (NGF*8) x 4 x 4
            nn.ConvTranspose2d(in_channels=NGF*8, out_channels=NGF*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NGF*4),
            nn.ReLU(inplace=True),
            # Input size: 1024 x 8 x 8
            nn.ConvTranspose2d(in_channels=NGF*4, out_channels=NGF*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NGF*2),
            nn.ReLU(inplace=True),
            # Input size: 512 x 16 x 16
            nn.ConvTranspose2d(in_channels=NGF*2, out_channels=NGF, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NGF),
            nn.ReLU(inplace=True),
            # Input size: 256 x 32 x 32
            nn.ConvTranspose2d(in_channels=NGF, out_channels=NGF//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NGF//2),
            nn.ReLU(inplace=True),
            # Input size: 128 x 64 x 64
            nn.ConvTranspose2d(in_channels=NGF//2, out_channels=NGF//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=NGF//4),
            nn.ReLU(inplace=True),
            
            # Output layer - Input size: 64 x 128 x 128
            nn.ConvTranspose2d(in_channels=NGF//4, out_channels=NC, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output size: NC x 256 x 256
        )
    
    def forward(self, input):
        return self.main(input)
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = NGPU
        self.main = nn.Sequential(
            # Input layer: Input size: NC x 256 x 256
            nn.Conv2d(in_channels=NC, out_channels=NDF//4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # Downsampling: Input size: NDF//4 x 128 x 128
            nn.Conv2d(in_channels=NDF//4, out_channels=NDF//2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input size: NDF//2 x 64 x 64
            nn.Conv2d(in_channels=NDF//2, out_channels=NDF, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input size: NDF x 32 x 32
            nn.Conv2d(in_channels=NDF, out_channels=NDF*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input size: NDF*2 x 16 x 16
            nn.Conv2d(in_channels=NDF*2, out_channels=NDF*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input size: NDF*4 x 8 x 8
            nn.Conv2d(in_channels=NDF*4, out_channels=NDF*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(NDF*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # Input size: NDF*8 x 4 x 4
            nn.Conv2d(in_channels=NDF*8, out_channels=NC, kernel_size=4, stride=1, padding=0, bias=False)#,
            #nn.Sigmoid() # Removed Sigmoid - unsafe to autocast with BCELoss criterion - use BCEWithLogitsLoss
            # Output size: 1 x 1 x 1
        )
        
    def forward(self, input):
        return self.main(input).view(-1, 1)
    
    
# Function to save gan 
def save_gan(model, filename):
    torch.save(model.state_dict(), filename)
    
    
# Function to load GAN if needed for later inference
def load_model(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set to eval mode
    return model


def load_dataset(root_dir, split, batch_size):
    """
    Load a dataset for a specific split (train, test, or validate).
    """
    dataset = GrayscaleImageDataset(root_dir=root_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    return dataloader


def evaluate_discriminator_real(netD, dataloader, device):
    """
    Evaluate the Discriminator's performance on a given dataset.
    """
    netD.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data in dataloader:
            real_images = data.to(device)
            batch_size = real_images.size(0)
            
            # Discriminator output for real images
            output = netD(real_images).view(-1)
            predictions = (output > 0.5).float()
            
            total_correct += (predictions == 1).sum().item()
            total_samples += batch_size
    
    accuracy = total_correct / total_samples
    return accuracy


def evaluate_discriminator_fake(netD, netG, dataloader, device):
    """
    Evaluate the Discriminator's performance on generator created images.
    """
    netD.eval()
    netG.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in dataloader:  # We don't need the real data, just using it for batch size and number of iterations
            batch_size = _.size(0)
            
            # Generate fake images
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_images = netG(noise)
            
            # Discriminator output for fake images
            output = netD(fake_images).view(-1)
            predictions = (output < 0.5).float()
            
            total_correct += (predictions == 1).sum().item()
            total_samples += batch_size
    
    accuracy = total_correct / total_samples
    return accuracy
            


def save_image_grid(images, file_path, nrow=8):
    """
    Save a grid of images.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    vutils.save_image(images, file_path, normalize=True, nrow=nrow)


def evaluate(netD, netG, root_dir, device, latent_dim, batch_size=64):
    """
    Evaluate the Discriminator and generate images for test and validate sets.
    """
    # Test set evaluation
    test_loader = load_dataset(root_dir, 'test', batch_size)
    test_real_accuracy = evaluate_discriminator_real(netD, test_loader, device)
    test_fake_accuracy = evaluate_discriminator_fake(netD, netG, test_loader, device)
    print(f"Discriminator accuracy on test set: Real {test_real_accuracy:.4f}, Fake {test_fake_accuracy:.4f}")
    
    # Validation set evaluation
    val_loader = load_dataset(root_dir, 'validate', batch_size)
    val_real_accuracy = evaluate_discriminator_real(netD, val_loader, device)
    val_fake_accuracy = evaluate_discriminator_fake(netD, netG, val_loader, device)
    print(f"Discriminator accuracy on validation set: Real {val_real_accuracy:.4f}, Fake {val_fake_accuracy:.4f}")


def generate_and_save_images(netG, fixed_noise, file_path, num_images=64):
    """
    Generate images using the trained Generator and save them in a grid. 
    Specifically used to check final performance.
    
    Args:
    netG (nn.Module): The trained Generator model
    fixed_noise (Tensor): A fixed noise tensor to generate images
    file_path (str): The path to save the generated image grid
    num_images (int): Number of images to generate (default: 64)
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # Set the model to evaluation mode
    netG.eval()
    with torch.no_grad():
        # Generate fake images
        fake = netG(fixed_noise).detach().cpu()
    # Rescale images from [-1, 1] to [0, 1]
    fake = (fake + 1) / 2.0
    # Create a grid of images
    img_grid = vutils.make_grid(fake[:num_images], padding=2, normalize=False, nrow=8)
    # Save the grid
    vutils.save_image(img_grid, file_path, nrow=8)
    print(f"Generated images saved to {file_path}")
    
    
def save_progression_images(img_list, output_dir='progression_images'):
    """
    Save the progression images stored in img_list.
    
    Args:
    img_list (list): List of image grids showing Generator progression
    output_dir (str): Directory to save the images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img_grid in enumerate(img_list):
        file_path = os.path.join(output_dir, f'progression_image_{i * 500}.png')
        vutils.save_image(img_grid, file_path, normalize=False)
    
    print(f"Saved {len(img_list)} progression images to {output_dir}")
    
    
def main():    
    # Seed for reproducibility
    torch.manual_seed(1)
    
    device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")
    
    # Create generator
    netG = Generator().to(device)
    netG.apply(weights_init)
    # Create discriminator
    netD = Discriminator().to(device)
    netD.apply(weights_init)
    
    # Setup dataset and data loader
    dataset = GrayscaleImageDataset(root_dir='./keras_png_slices_data', split='train')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

    # Initialise BCE loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimiserD = optim.Adam(netD.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimiserG = optim.Adam(netG.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0   
    
    for epoch in range(NUM_EPOCHS): # Training Loop
        for i, data in enumerate(dataloader):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad() # Reset D gradients
            
            # train D with real
            real = data.to(device)
            b_size = real.size(0) # Get batch size
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device) # Create real labels
            
            with torch.cuda.amp.autocast():
                output = netD(real).view(-1) # Forward pass real batch for D
                errD_real = criterion(output, label) # Calc D err on real
            scaler.scale(errD_real).backward() # Calc gradients in back pass
            D_x = output.mean().item()
            
            # Train with fake
            noise = torch.randn(b_size, LATENT_DIM, 1, 1, device=device)
            with torch.cuda.amp.autocast():
                fake = netG(noise) # Gen fake image batch for G
                label.fill_(fake_label)# add fake labels
                output = netD(fake.detach()).view(-1) # classify with D
                errD_fake = criterion(output, label) # Calc D err
            scaler.scale(errD_fake).backward() # Calc grads for D
            D_G_z1 = output.mean().item() # Avg D out on fake
            # Total err for D
            errD = errD_real + errD_fake
            scaler.step(optimiserD) # Update D weights

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad() # reset G gradients
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            with torch.cuda.amp.autocast():
                output = netD(fake).view(-1) # classify fake batch with D
                errG = criterion(output, label) # Calc G's err
            scaler.scale(errG).backward() # Calc gradients for G
            D_G_z2 = output.mean().item() #  D's average output for fake data (2nd time)
            scaler.step(optimiserG) # update G weights
            
            scaler.update() # update scaler for next iter
            
            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{NUM_EPOCHS}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
            
            # Save Losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Check generator progress - return G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == NUM_EPOCHS-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
            iters += 1
    
    # Generate and save images on final generator
    generate_and_save_images(netG, fixed_noise, 'generated_images/final_generated_images.png')
    plot_losses(G_losses, D_losses)
    save_progression_images(img_list)
    evaluate(netD, netG, './keras_png_slices_data', device, LATENT_DIM)

    # Save final models
    save_gan(netG, 'models/generator_final.pth')
    save_gan(netD, 'models/discriminator_final.pth')
            
    return G_losses, D_losses, img_list


# Function to plot training losses
def plot_losses(G_losses, D_losses):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses_plot.png')
    

if __name__ == "__main__":
    G_losses, D_losses, img_list = main()