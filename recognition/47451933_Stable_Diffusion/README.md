# Stable Diffusion Variant on ADNI Dataset
This is an attempt at an implemention of latent diffusion similar to whats used in stable diffusion to generate new MRI brain scans. The model was trained on the ADNI dataset that contains a total 30520 brain MRI scans in which there are two types of scans AD (Alzheimer’s Disease) labeled as 0 in the results and NC (Normal Control) labeled as 1 in the results. The model is used to generate new MRIs where you can chose if you want a AD type image or an NC type image generated.

# Contents

# Stable Diffusion / Latent Diffuion Overview
Stable diffusion is a specfic implemention of latent diffusion created by CompVis with support from Stability AI and trained on a subset of the LAION databse. it uses OpenAi's CLIP Bit0L/14 text encoder for conditiong on text prompts. 
### Network Structure
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram.png" width="800">

## Setup

### Data Transformations

## Training

## Infrence
<img src="https://github.com/SixLeopard/PatternAnalysis-2024/blob/a681523d2aa48e0a22c2dd8d42716b387e8c94e9/recognition/47451933_Stable_Diffusion/results/COMP3710SD_diagram_infrence.png" width="800">

## References
