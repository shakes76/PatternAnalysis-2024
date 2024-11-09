# GFnet classify Alzheimer's disease of the ADNI dataset

## What is FGnet  
GFNet is a conceptually simple yet computationally efficient architecture designed to advance the trend of minimizing inductive biases in vision models while maintaining log-linear computational complexity. The core idea behind GFNet is to learn the interactions among spatial locations in the frequency domain, diverging from traditional self-attention mechanisms in vision transformers and fully connected layers in MLPs. [1]

In GFNet, interactions among tokens are modeled through a set of learnable global filters applied directly to the spectrum of input features. This unique approach enables the model to capture both long-term and short-term interactions effectively, as the global filters encompass all frequencies. Notably, GFNet learns these filters directly from raw data, eliminating the need for human priors. [1]

GFNet builds upon the foundation of vision transformers, implementing key modifications to enhance efficiency. Specifically, it replaces the self-attention sub-layer with three pivotal operations: [1]

- __2D Discrete Fourier Transform__: Converts input spatial features to the frequency domain.  
- __Element-wise Multiplication__: Applies global filters to the frequency-domain features.  
- __2D Inverse Fourier Transform__: Maps features back to the spatial domain.  

This use of Fourier transform facilitates information mixing across tokens, making the global filter layer significantly more efficient than self-attention and MLPs due to the 
𝑂(𝐿log𝐿) complexity of the Fast Fourier Transform (FFT) algorithm. Consequently, GFNet is less sensitive to token length 𝐿 and seamlessly integrates with larger feature maps and CNN-style hierarchical architectures. [1]

The overall architecture of GFNet is illustrated in this Figure. [1]  

![image](https://github.com/user-attachments/assets/9f7d5961-35d6-427b-8405-f605675c47a7)  

ANDI dataset: 
The Alzheimer’s Disease Neuroimaging Initiative (ADNI) is a longitudinal, multi-center, observational study. The overall goal of ADNI is to validate biomarkers for Alzheimer’s disease (AD) clinical trials.

Use the 2D brain scan slice data from ANDI dataset to train and test this model. The preprocessed version of this data set can be found here https://filesender.aarnet.edu.au/?s=download&token=a2baeb2d-4b19-45cc-b0fb-ab8df33a1a24.  
All samples are brain scan slice image like follow:  

![388726_93](https://github.com/user-attachments/assets/c9a32634-5c5e-499e-822b-3df24bf2d7eb)

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
embed_dim=512    # embedding dimension
depth=12    # depth of transformer
drop_rate=0.    # dropout rate
drop_path_rate=0.1    # stochastic depth rate
norm_layer=None    # normalization layer
```
Loss function use cross entropy loss. Use Adam optimerzer and set the learning rate to 0.001 and the weight decay to 0.001.

## Data Processing
Firstly, download the AD_NC dataset. Split the original train set into the new train set and validation set with a 90% to 10% ratio.
For the new train set, do the following data augmentation.  
```
transforms.Grayscale(num_output_channels=1),
transforms.RandomRotation(20),
CropBrainRegion(),
transforms.Resize((224, 224)),
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
transforms.ToTensor(),
transforms.Normalize(mean=[0.1174], std=[0.2163])
```
for the val and test set, do the following data augmentation.
```
transforms.Grayscale(num_output_channels=1),
CropBrainRegion(),
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.1174], std=[0.2163])
```
mean and standard deviation is calculated by the overall dataset.  
CropBrainRegion is a function that use cv2 to cut the black area of a brain image. By this way we can let the model focus on the brain area.  

## Traning Process
Run 65 epochs and visualizing the training process:  
![training_process](https://github.com/user-attachments/assets/fbab4760-3f8d-4d1c-90bb-528445718e7c)

Prediction on test set:  
```
Confusion Matrix:
[[2747 1713]
 [ 455 4085]]
test Loss: 1.4276, test Acc: 0.7591
Precision: 0.7046
Recall: 0.8998
F1 Score: 0.7903
```
On one NVIDIA a100 graph card, the total traning time is 2:06:36.  

## How to Test
You can use 'final_validate(model, test_loader, criterion, device)' and 'draw_training_log()' function in the end of train.py. They are already there. And you can also run predict.py solo. If you use this way, don't forget to edit the path of your model and dataset inside the predict.py

## Dependency Library
Make sure your environment have following dependency:
```
cv2（OpenCV）
torch（PyTorch）
numpy
Pillow（PIL）
torchvision
timm
scikit-learn（sklearn）
tqdm
```
You can use the following code to download these dependencies.
If you have cuda:  
```
pip install opencv-python torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 numpy Pillow timm scikit-learn tqdm
```
If you don't have cuda:  
```
pip install opencv-python torch torchvision torchaudio numpy Pillow timm scikit-learn tqdm
```

## Run this Project
You can easily run this project and change any parameters you want by using train.py. The 'split_val_set' function in train.py only need to run once. It will cover the previous new dataset if you run multiple times. Nothing terrible will happend but better don't waste 3 min.

## Conclusion
This code use GFnet with some data augmentation get 75.91% accuracy on the test set of ANDI 2D brain scan slice data after training 2:06:36 on one NVIDIA a100 graph card.

## Reference
[1] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global filter networks for image classification. (NeurIPS)
