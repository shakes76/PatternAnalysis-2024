# Stable Diffusion on ADNI Dataset
## Author
**Jamie Westerhout, S4745193**
## Description
This is an attempt at an implemention of latent diffusion model based on the stable diffusion architecture to generate new MRI brain scans. The model was trained on the ADNI dataset that contains a total 30520 brain MRI scans in which there are two types of scans AD (Alzheimer’s Disease) labeled as 0 in the results and NC (Normal Control) labeled as 1 in the results. The model is used to generate new MRIs where you can chose if you want a AD type image or an NC type image generated.

# Contents
- [File Structure](#File-Structure)
- [Background](#Background)
- [Implementation](#Implementing-Stable-Diffusion)
- [Usage]()

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
Stable diffusion is a kind of diffusion model call latent diffusion (except for v3 which divereged from latent diffusion).
# Diffusion Models

## Latent Diffusion Models
latent diffusion was devloped by reserachers at Ludwig Maximilian University

# Implementing Stable Diffusion
Stable diffusion V1 is the first stable diffusion model released and has an 8 factor downsampling VAE (Variational Autoencoder), a 860M U-Net and uses OpenAi's CLIP Bit0L/14 text encoder for conditiong on text prompts. 
Exactly replicating one of the stable diffusion model would require significantly more powerful hardward then what is avalible to train and be signficantly overcomplicted due to fact that this model is only required to handle mri brain scanes with 2 labels and for example doesnt need the complexity of the clip encoding since it doesnt need to be able to handle full text descriptions or image inputs etc. 
Since all the orignal versions of stable diffusion are types of latent diffusion model this is an implementation of stable diffusion's latent diffusion architecture for the ADNI dataset.

## Latent Diffusion Architecture used by Stable Diffusion
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f6/Stable_Diffusion_architecture.png" width="800">

## Model Architecture
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram.png" width="800">

## Setup

### Data Transformations

# Usage
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram_infrence.png" width="800">

## Training

## Inference

# Results

# References
