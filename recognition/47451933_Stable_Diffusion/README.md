# Stable Diffusion on ADNI Dataset
## Author
**Jamie Westerhout, S4745193**
## Description
This is an attempt at an implemention of latent diffusion model based on the stable diffusion architecture to generate new MRI brain scans. The model was trained on the ADNI dataset that contains a total 30520 brain MRI scans in which there are two types of scans AD (Alzheimer’s Disease) labeled as 0 in the results and NC (Normal Control) labeled as 1 in the results. The model is used to generate new MRIs where you can chose if you want a AD type image or an NC type image generated.

# Contents
- [File Structure](#File-Structure)
- [Background](#Background)
- [Implementation](#Implementing-Stable-Diffusion)
- [Usage](#Usage)
- [Results](#Results)
- [References](#References)

# File Structure
```
47451933_Stable_Diffusion
├── Results (train results + generation outputs + diagrams)
│   ├── BrainSeq1.png
│   ├── Brains1.png
│   ├── COMP3710SD_diagram.png
│   ├── COMP3710SD_diagram_infrence.png
│   ├── Diffusion_loss.png
│   ├── Diffusion_loss_2.png
│   ├── Diffusion_loss_21.png
│   ├── Diffusion_loss_22.png
│   ├── Diffusion_loss_23.png
│   ├── Diffusion_loss_24.png
│   ├── UMapPlot.png
│   ├── VAE_loss.png
│   ├── VAE_loss_2.png
│   ├── VAE_loss_21.png
│   └── VAE_loss_22.png
│   └── AD_Genrated_Brains.png
│   └── NC_Genrated_Brains.png
│   └── test_brains.png
├── .gitignore
├── README.md
├── dataset.py (for loading in the ADNI dataset)
├── modules.py (contains all the networks)
├── predict.py (use to generate new images)
├── project_setup.py (use to setup folder strucutre and check dataset is avaliable)
├── requirements.txt (what python librariesa re required)
├── train.py (train the model)
└── utils.py (contains extra functions used by most the files)
```
# Background
## Stable Diffusion
Stable diffusion is a generative model and more specifcally is a kind of diffusion model call latent diffusion (except for v3 which divereged from latent diffusion). it was developed by Stability AI with help from Runway and Compvis.
# Diffusion Models
Diffusion models are a type of model where images are generated from noise by iterativly removed the noise to constrcut new images. to train them they are train on the different levels of noise tagges with a time step, these images are then passed to a unet along with its timestep to try and predict the noise in the images at that timestep. to generate new images a completely noisy image is passed and it goes throughj every timestep predicting and removing the noise each time. this results in an image completely generated from noise since the u-net is predicting noise that when removed will look like the images its trained on.

Diffusion Models are an alternative to the GAN and standalone VAE approch to generating images. It produces a better varity of images than GANs and better quaility images then a VAE but at the cost of slower generation performance 
## Latent Diffusion Models
latent diffusion was devloped by reserachers at Ludwig Maximilian University and is improvement apon the the diffusion model by since using larger images was extremely slow with diffusion. Latent diffusion can drastically imporve the speed by doing the diffusion process in the latent space created by VAE this way its not doing the porcess on the full image can do it on a downsampled version, thus requiring signficantly less resources and time.

# Implementing Stable Diffusion
Stable diffusion V1 is the first stable diffusion model released and has an 8 factor downsampling VAE (Variational Autoencoder), a 860M U-Net and uses OpenAi's CLIP Bit0L/14 text encoder for conditiong on text prompts. 
Exactly replicating one of the stable diffusion model would require significantly more powerful hardward then what is avalible to train and be signficantly overcomplicted due to fact that this model is only required to handle mri brain scanes with 2 labels and for example doesnt need the complexity of the clip encoding since it doesnt need to be able to handle full text descriptions or image inputs etc. 
Since all the orignal versions of stable diffusion are types of latent diffusion model this is an implementation of stable diffusion's latent diffusion architecture for the ADNI dataset.

## Latent Diffusion Architecture used by Stable Diffusion
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f6/Stable_Diffusion_architecture.png" width="800">
this is the letent diffusion architecture used for the orginal stable diffusion versions.

### Key Components
- Encoder and Decoder to takes images from pixel space to latent space
- Forward diffusion process -> add increasing levels of noise over a set number of timesteps
- De-Noising U-Net to predict the noise of the current timestep
- Corss attention to condition on input
- Denoising step to remove the noise predicted by the unet

## Model Architecture Implemented
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram.png" width="800">
can see the implemented model architecture has all the same key componets as the stable diffusion models

### Key Components
- VAE Encoder and Decoder to takes images from pixel space to latent space
- Forward diffusion process -> add increasing levels of noise over a set number of timesteps (using noise schedular + sinsuodial time step embedings)
- De-Noising U-Net to predict the noise of the current timestep
- Corss attention to condition on wether the brain is AD or NC
- Denoising step to remove the noise predicted by the unet

## Setup

### Train, Test, Validation Split
the ADNI dataset comes with a train test split already but having an additional validation set is useful to see generalisation performance during traning to see if the model is overfitting. Since the images are broken up between a patient and then all the slices of there brain to split the data into an aditional set is not as asy as just taking a 10% extra split off since this would result in images data leakage. so to get an additional validation set the test set was split using the following algorthim (in sudo code):
```python
num_patients_for_val_set = num_images_in_train_set // 100
patients_seen = set()
for i in test_set:
    patient = i.split("_")[0]
    if len(patients_seen) <= num_patients_for_val_set:
        patients_seen.add(patient)
    if patient in patients_seen:
        move_from_test_to_val(i)
```
the orignal train set was used as the train set then what was left of the test set about 11k images was used for testing and for validation the new validation set created was used
### Data Transformations
the following transformation were applied to the images before laoding
```python
transforms = tvtransforms.Compose([
            tvtransforms.Grayscale(), #grascale images 
            tvtransforms.Resize(self.image_size), #the next two lines decrease the resolution to 200x200
            tvtransforms.CenterCrop(self.image_size),
            tvtransforms.ToTensor(), #turn the datat into a tensor if its not already
            tvtransforms.Normalize(0.5,0.5)]) #normilze the data
```
- the grey scale takes the images from 3 channels to 1 channel reducing the size of each image without loss of information beacuse there grey scale anyway
- resize to 200x200 since there orignally 256x240 so croped to 200x200 to make them sqaure to simplify things and slightly reduce the size of each image
- normilize the images otherwise this could cause poor performance
# Usage
before running either the training or predicting run the setup.py first to make sure all the correct folders are created for the models to save to when completed and it also check to make sure the dataset is present and in the right location and format

## Training
The the VAE for encoding and decoding is train sepreatly from the unet since it isnt reasonable for them both to be trained at the same since the unet relys on latent space produced from the VAE to train and if this is changing at the same time it won't learn properly. First the VAE is trained to encode images into latent space in this case since its a vae it produces a distripution over the latent space, it then reconstruct from a sample of the distribution.
### VAE Training
Trained to be able to encode and decode the image trying to get as similiar output to the image it put in.
### Encoding
each image is encoded using the vea wich then produces a mu and logvar for the latent space, to actually get an encoded image, a sample is taken from this latent space which also output as h
```python
# get mean and variance for sampling
# and to be used to calculate Kullback-Leibler divergence loss.
h = self.encoder(x)
mu = self.fc_mu(h)
logvar = self.fc_logvar(h)

# get sample
z = self.sample_latent(mu, logvar)
h = self.decode(z)

return h, mu, logvar
```
### Decoding
after each imge goes through the encoder the sample h then gets put throught he decoder, to reconstruct the image
```python
x_recon = self.decoder(z)
return x_recon
```
### Loss
the loss for a vae is combination of the conconstruction loss and Kullback-Leibler divergence loss to not onlt ensure that its reconstructing the images well by the KL loss works like regulisation ensuring thats its generlising well by measuring the distribution its producing
```python
#sudo code
reconstrcution_loss = MSE(reconstructed_images, images, reduction='sum')
Kullback-Leibler_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
loss = 0.5*recon_loss + Leibler_divergence_loss
```

### Unet Training

#### Encoding

#### Foward Diffusion (adding noise)

#### U-Net noise prediction

#### Reverse Diffusion

### Decoding

## Inference
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram_infrence.png" width="800">

# Results

# References
