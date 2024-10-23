# GFNet for ADNI Brain Data

## Description
A Global Filter Network (GFNet) is a type of deep learning architecture used in computer vision. A **Fourier Transform** is applied to the input data resulting in operation in the **frequency domain**. This enables the use of global filtering operations, which make the network more efficient at handelling larger scale patterns. Instead of using localised filters (as with CNNs), a **Global Filter** is implemented which spans the entire input space. This allows the processing of infomation globaly without requiring deeper convolutional layers or attention mechanisms. These traits prove to be suitible for use in the task of classifying and diagnosing Alzheimer's from brain scans in the ADNI dataset.

![Visualisation of GFNet](readme_support/GFNet_Overview.png)
Visualisation of GFNet ([ResearchGate, 2023](https://www.researchgate.net/figure/An-overview-of-the-split-frequency-filter-network-Our-network-structure-based-on-GFNet_fig1_372975394))

## Dependencies
The following dependencies are required: m
* torch >= 1.8.0
* torchvision
* matplotlib

*Note:* To use the required `rffn2` and `irfft2` function from PyTorch, verion 1.8.0 or greater is required.

*Note:* The results generated from this model may differ between runs as the filters are set randomly. With the same hyperparamters, similar results could be achieved.

## Example Model - Training

### Inputs
To train the model, run `train.py`
These are the parameters specified for the model:
```
GFNet(embed_dim=768, img_size=224, patch_size=16, in_chans=3, mlp_ratio=4, depth=12, drop_rate=0., num_classes=1000)
```
*Note:* These parameters are the defaults for the GFNet.

### Outputs
[Example Model Output](readme_support/train_output.txt)

### Plot
![Model Accuracies](readme_support/model_accuracies.png)

## Example Model - Prediction

### Inputs
Once you have a trained model, to predict on the mode, run `predict.py`
This is how the model is being loaded:
```
model = GFNet()
model.load_state_dict(torch.load("GFNet-Model.pth"))
```

### Outputs
```
Test - Loss: 0.7239, Accuracy: 67.44%
```

## Pre-Processing
The data was already split into a test and train set so no splitting of the data was required.

The training data was augmented using random rotations and random resized cropping as not all of the images were alligned.

The images were resized to 224x224 to match the standard for GFNet.

## References
[1] “An overview of the split-frequency filter network.” ResearchGate, 2023. https://www.researchgate.net/figure/An-overview-of-the-split-frequency-filter-network-Our-network-structure-based-on-GFNet_fig1_372975394

[2] raoyongming, “Global Filter Networks for Image Classification,” GitHub, 2021. https://github.com/raoyongming/GFNet