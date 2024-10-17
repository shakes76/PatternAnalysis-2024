# GFNet README

## Project
- [Problem](#5 in COMP3710 Report)
- [Model](GFNet - a cutting-edge vision NN. GFNEt is based off the ViT Transformer models, but replaces the self-attention layer with a global filter layer utilising fft to learn spatial interactions)

## Description
This is an adapted implementation of [GFNet](https://ieeexplore.ieee.org/document/10091201) originally described in the paper by Y. Rao, et. al.
To minimise time spent training, transfer learning from the [gfnet-xs](https://github.com/raoyongming/GFNet/tree/master?tab=readme-ov-file) trained by Y. Rao, et. al.

The file modules.py contains the GFNet implemented as desribed in the paper [Global Filter Networks for Image Classification](https://arxiv.org/abs/2107.00645) by Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, and Jie Zhou ([GitHub](https://github.com/raoyongming/GFNet))

## Installation
Instructions for setting up the environment and dependencies.

# Dependencies
timm 
pytorch > 1.8

## License
MIT License