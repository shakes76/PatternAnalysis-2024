# GFnet classify Alzheimer's disease of the ADNI dataset

## What is FGnet  
GFNet is a conceptually simple yet computationally efficient architecture designed to advance the trend of minimizing inductive biases in vision models while maintaining log-linear computational complexity. The core idea behind GFNet is to learn the interactions among spatial locations in the frequency domain, diverging from traditional self-attention mechanisms in vision transformers and fully connected layers in MLPs.

In GFNet, interactions among tokens are modeled through a set of learnable global filters applied directly to the spectrum of input features. This unique approach enables the model to capture both long-term and short-term interactions effectively, as the global filters encompass all frequencies. Notably, GFNet learns these filters directly from raw data, eliminating the need for human priors.

GFNet builds upon the foundation of vision transformers, implementing key modifications to enhance efficiency. Specifically, it replaces the self-attention sub-layer with three pivotal operations:

- __2D Discrete Fourier Transform__: Converts input spatial features to the frequency domain.  
- __Element-wise Multiplication__: Applies global filters to the frequency-domain features.  
- __2D Inverse Fourier Transform__: Maps features back to the spatial domain.  

This use of Fourier transform facilitates information mixing across tokens, making the global filter layer significantly more efficient than self-attention and MLPs due to the 
𝑂(𝐿log𝐿) complexity of the Fast Fourier Transform (FFT) algorithm. Consequently, GFNet is less sensitive to token length 𝐿 and seamlessly integrates with larger feature maps and CNN-style hierarchical architectures.

The overall architecture of GFNet is illustrated in Figure 1. 

![image](https://github.com/user-attachments/assets/9f7d5961-35d6-427b-8405-f605675c47a7)  

ANDI dataset: 
The Alzheimer’s Disease Neuroimaging Initiative (ADNI) is a longitudinal, multi-center, observational study. The overall goal of ADNI is to validate biomarkers for Alzheimer’s disease (AD) clinical trials.

Use the 2D brain scan slice data from ANDI dataset to train and test this model. The preprocessed version of this data set can be found here https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24.  

two classes: Alzheimer’s disease (AD) and Cognitive Normal (CN). File structure is like follow with the images account:  

```
AD_NC/   
├── train/  
│   ├── AD/  // 10400
│   └── NC/  // 11120
└── test/  
    ├── AD/  // 4460
    └── NC/  // 4540
```
## Model Implementation
By using the GFnet base code from https://github.com/raoyongming/GFNet. Select the code to build the standard GFnet. Which include MLP, GlobalFilter, Block, PatchEmbed and GFNet classes. Set the following hyperparameter:  
```
img_size=224    # input image size
patch_size=16    # patch size
in_chans=1      # input channel 
num_classes=2    # number of classes
embed_dim=768    # embedding dimension
depth=12    # depth of transformer
drop_rate=0.    # dropout rate
drop_path_rate=0.    # stochastic depth rate
norm_layer=None    # normalization layer
```
Loss function use cross entropy loss. Use Adam optimerzer and set the learning rate to 0.001 and the weight decay to 0.001.




Reference:


