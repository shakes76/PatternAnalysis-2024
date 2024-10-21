

# Data Preprocessing

The data consists of brain images which either have Alzheimer's or don't. The training and test sets were concatenated. For transformations, the data was rescaled to be exactly 256x256, turned into pytorch tensors, and then normalized. The rescaling didn't change the quality of the image because the initial size was very close (240x256). 

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = dset.ImageFolder(root= data_Train, transform=transform)
test_dataset = dset.ImageFolder(root= data_Test, transform=transform)
combined_dataset = data.ConcatDataset([train_dataset, test_dataset])
```

# The Stable Diffusion Model

The main idea behind stable diffusion is to make all calculations in the latent space; this is mainly done to improve the efficiency of the model. The diffusion model itself works by predicting and subtracting noise from an image, adding some noise back, and then repeating the process until the result is recognizable and clear. The model in this project is unconditional, meaning no text or any similar input is required to generate images; therefore, the model randomly generates images which resemble the ones it was trained on. 

## Architecture 

The architecture of a conditional stable diffusion model is very similar to the unconditional one (the only difference being the extra embedding and attention), hence why the conditional (and more general) version of the model is shown. 

A simple autoencoder was used to turn the images from pixel space to latent space. The encoder and decoder have three convolutional layers and three transpose convolutional layers respectively, and the chosen latent dimension is 128 channels. 

The UNET was initially inspired from, and many modifications have been made for it to fit general structure of the stable diffusion model. The UNET contained three encoder blocks and three decoder blocks with a resnet block as the bottleneck. The timestep (a number which indicates the level of noise) was embedded in each block. 

# Training

The autoencoder and the UNET were trained separately. This was done because it was more efficient and it was easier to maximize the potential of each of them. MSE (mean squared error) used to calculate the loss of the autoencoder, and smooth l1 loss was used for the UNET. Since the UNET is the most essential and significant part of the stable diffusion model, only the UNET loss plot will be shown:





# Results

## Autoencoder

Below are images that have been encoded and then decoded by the autoencoder:

The high similarity between the original image and the final decoded image shows that the latent space accurately represents the original data. This is vital because it helps the UNET identify the added noise better.  

## UNET

To visualize the perfomance of the UNET, images had noise added to them, the UNET was then used to extract the noise, which was then subtracted from the original image. Below are the results:

When there is a lot of noise added, it may seem that the model isn't doing a good job because the resultant image is blurry, however that is expected because the task of removing that much noise and output a clear image all in one go is an extremely difficult job for any model to do. This is why, when generating images with a diffusion model, small and gradual denoising steps are taken repeatedly instead of it all being done at once. Moreover, we can see that when only a little noise is added, the model returns a perfectly clear image. 

In conclusion, the UNET is doing the job it's expected to do reasonably well, and that is to extract noise from images. 

## Image Generation

# Dependencies


```python
python
torch
torchvision
numpy
matplotlib 
PIL
math
```


# File Structure

- StableDiffusion-47015746
  - README.md
  - dataset.py
  - modules.py
  - train.py
  - predict.py
  - driver.py


