# Stable Diffusion Variant on ADNI Dataset
This is an attempt at an implemention of latent diffusion similar to whats used in stable diffusion to generate new MRI brain scans. The model was trained on the ADNI dataset that contains a total 30520 brain MRI scans in which there are two types of scans AD (Alzheimer’s Disease) labeled as 0 in the results and NC (Normal Control) labeled as 1 in the results. The model is used to generate new MRIs where you can chose if you want a AD type image or an NC type image generated.

# Contents

# Stable Diffusion / Latent Diffuion Overview
Stable diffusion originated from the latent diffusion project orginally devloped by reserachers at Ludwig Maximilian University in which stable diffusion v1-3 are just specific uses of the latent diffusion model and is created by CompVis with support from Stability AI and Runway. Stable diffusion V1 is the first stable diffusion model released and has an 8 factor downsampling VAE (Variational Autoencoder), a 860M U-Net and uses OpenAi's CLIP Bit0L/14 text encoder for conditiong on text prompts. Exactly replicating the stable diffusion model would require significantly more powerful hardward then what is avalible to train and be signficantly overcomplicted due to fact that this model is only required to handle mri brain scanes with 2 labels and for example doesnt need the complexity of the clip encoding since it doesnt need to be able to handle full text descriptions or image inputs etc. Since all the orignal versions of stable diffusion usees latent diffusion this is an implementation of letent diffusion that matchs the arcitecture of stable diffusion.
## Latent Diffusion Architecture used by Stable Diffusion
<img src="https://upload.wikimedia.org/wikipedia/commons/f/f6/Stable_Diffusion_architecture.png" width="800">

### Network Structure
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram.png" width="800">

## Setup

### Data Transformations

## Training

## Infrence
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram_infrence.png" width="800">

## References
