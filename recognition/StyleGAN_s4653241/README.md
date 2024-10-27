# StyleGAN2 on AD_NC Brain Dataset

## Overview

This StyleGAN2's application is to generate realistic like brain scans using the ADNI dataset for Alzheimer's disease. The primary goal is to provide a model that is able to generate "reasobably clear images". Using StyleGAN2 ability to

## Table of Contents
[Installation](#installation)
[Requirements](#requirements)
[Dataset](#dataset)
[File Structure](#file-structure)


## Installation
1. Download [ADNI dataset for Alzheimer’s disease](https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24).
2. Set paths of dataset in config.py
3. Check neccessary [requirements](#requirements) are met

## Requirements

| Package | Version |
| --- | --- |
|torch | 2.5.0.dev20240904 |
|torchvision | 0.20.0.dev20240904 |
|tqdm | 4.66.5 |
|numpy | 1.26.4 |
|pandas | 2.2.2 |
|matplotlib | 3.9.2 |
|seaborn | 0.13.2 |
|scikit-learn | 1.5.1 |
|scipy | 1.13.1 |
|pillow | 10.3.0 |
|os |  |


## Dataset

## File Structure

## Results

## Conclusion

## References

Task 8:
```Create a generative model of one of the ADNI brain data set (see Appendix for links) using either a variant of StyleGAN [10]/StyleGAN2 [11] or Stable Diffusion [12] that has a “reasonably clear image”. You should also include a TSNE or UMAP embeddings plot with ground truth in colors and provide a brief interpretation/discussion. See recent UQ work for hints of how to incorporate labels in style [13]. [Hard Difficulty]```
