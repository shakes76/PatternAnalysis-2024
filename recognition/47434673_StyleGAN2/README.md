# COMP3710 StyleGAN2 Brain Image Generator
A StyleGAN2 model trained on the OASIS dataset.

## Objective
The objective is to create a generative model of the ADNI brain data set using a StyleGAN2
that has a reasonably clear image. Furthermore, a UMAP embeddings plot with ground truth in colours 
is included.

## StyleGAN2 - Algorithm Description
### GAN
The goal of the generative model is to learn the underlying distribution of the training dataset. 
The generative adversarial network (GAN) performs a min-max game between two convolutional networks; the 
generator and the discriminator. The generator takes random noise input and outputs a fake image with a 
fake distribution, while the discriminator performs a binary classification task of distinguishing between 
fake and real images. The goal of the generator is to generate a fake image that highly resembles the real 
image.

### StyleGAN
StyleGAN is a generative adversarial network (GAN) architecture that uses a style-based generator for image synthesis.
In contrast to traditional GANs, StyleGAN does not start directly from a random noise vector, but uses a mapping network to
map the noise to an intermediate latent space first. Random vectors from this space and by an affine transformation become style vectors, which represent the dientangled features from the training distribution. Using adaptive instance normalisation (AdaIN),
the style vectors are directly embedded into the intermediate layers of the generator. Stochastic variation control adds 
much finer details into the image through the addition of noise input into the generator similarly using AdaIN blocks.
The StyleGAN allows for style mixing, since the styles are separated across different layers which are progressively trained 
using different data sources. This allowing for better control over high-level and low-level features independently, making it possible to blend, interpolate, or adjust specific aspects of the generated image. 

### StyleGAN2
StyleGAN2 builds upon the original StyleGAN architecture by addressing several limitations, such as visible blob-like artifacts on the images and the progressive growing problem. StyleGAN2 removes the blob-like artifacts by replacing the AdaIN layer with weight modulation. Instead of manipulating the feature maps using AdaIN, the convolution kernel weights are scaled
with a style vector in weight modulation and then the kernel is normalised in weight demodulation. This removes the progressive growing problem which introduces the blob-like artifacts. The resulting architecture produces higher quality images than the StyleGAN.

## The Problem
stuff

## Data
The dataset that this model is trained on is the publicly available OASIS brain dataset. This dataset contains 2D MRI image slices of brains. The directory structure of this dataset is the following:

```
└───keras_png_slices_data
    ├───keras_png_slices_seg_test
    ├───keras_png_slices_seg_train
    ├───keras_png_slices_seg_validate
    ├───keras_png_slices_test
    ├───keras_png_slices_train
    └───keras_png_slices_validate
```

This model trains on images taken from all the folders, i.e. the dataloader does not discriminate between images in the train, test, or validate directories. This is so that this model can train on more diverse images.

## Requirements

This program has been tested to run on Windows. 64-bit Python3.11 or later is recommended. Anaconda3 or later is recommeded. The required libraries are: 

pytorch
torchvision
pytorch-cuda
cudatoolkit
numpy
matplotlib
tqdm

Libraries may use newer versions.

This should ideally be run with a NVIDIA A100 GPU with 128GB of DRAM. Testing on other GPUs has not been performed.

## Code Structure

The following files are included in this repository:

```
dataset.py
modules.py
predict.py
train.py
utils.py
config.py
```

To use pre-trained models instead of training new models, please set the `load_models` hyperparameter in `utils.py` to `True`.
That is: `load_models = True`.
Also set the `model_path` hyperparameter to the file path of these pretrained models. The pretrained model files 
should be `.pth` files.

Optionally, a seed for training the model can be set by changing the `seed` hyperparameter in `utils.py`. The `seed` should be an integer. Otherwise, a random `seed` will be chosen.

## Results

## Regular Training

### Extended Training
The graphs below show the loss of the generator and discriminator during their training cycles for 50 epochs (35500 iterations). The training time for this model was approximately 8hrs and 24mins on NVIDIA A100 GPU. 
![Disc_loss](assets/Disc_loss.png)

![Gen_loss](assets/Gen_loss.png)

![Comb_loss](assets/Combined_loss.png)




## Conclusion

## References


