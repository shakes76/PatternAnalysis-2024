# StyleGAN Training from Scratch from ADNI Data Set

This repository discusses the **training data augmentation**, **module development**, and the overall **training process** for the original StyleGAN model. Although more advanced models such as **StyleGAN2** and **StyleGAN3** have since been introduced, the original **StyleGAN** was chosen for this project due to its pioneering role in integrating the **style-based architecture** into generative adversarial networks (GANs).

For more details on the architecture, please refer to the original paper: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948).
## Data Set 

For this training, the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** dataset was used. The dataset consists of MRI brain scans of patients with **Alzheimer's Disease (AD)** and **Normal Controls (NC)**. All images are in grayscale and have a resolution of **256 x 256 pixels**.

- **AD Image**: 

![AD Image](/recognition/Readme_images/218391_78.jpeg)

- **NC Image**: 

![NC Image](/recognition/Readme_images/808819_88.jpeg)
    

The dataset contains approximately **30,000 images** in total, with **20,000** images allocated for training and **10,000** for testing. For the training of my StyleGAN, I exclusively used the training images, and they were sufficient to generate clear MRI brain scans.
## File Structure

This repository consists of the following five major files:

- **`dataset.py`**: Responsible for all data augmentation and batch loading.
- **`model.py`**: Defines the model architecture implemented using PyTorch.
- **`params.py`**: Contains important parameters for the model.
- **`train.py`**: Defines the training loop and training function.
- **`predict.py`**: Implements a class for loading models and generating images.


## Model Architecture 
<img src="recognition/Readme_images/image.png" alt="Model Architecture" width="500"/>


The StyleGAN model architecture is similar to the GAN with a few changes. The GAN model relied an adversarial network which trained a Generator and a Discriminator to progressively improve the image generation outputs. However, a major issue with the GAN model is that it was prone to overfitting, model collapes and offered little diversity in the output images.

Relative to the GAN model some significant changes were made. 

**Introduction of the Mapping Network:**

This network maps the Latent space **$z$** with another latent space $w$. This allows more control over the generated images by influencing different layers of the generator with styles.

**Adaptive Instance Normalisation:** 

${AdaIN}(x_i, y) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$

Adaptive Instance Normalization (AdaIN) adjusts the **mean** ( $\mu(x_i)$ ) and **variance**( $\sigma(x_i)$ ) of the input features (like an image) based on the style you want to apply.

**Mixing Regularisation:**

This technique passes multiple latent vectors to different style layers, preventing overfitting and helping the model generalize styles better.

Some components of the model such as the discriminator were taken directly from the ProGAN paper. In addition, a gradient penalty module was created which was based of the **Wasserstein GAN with Gradient Penalty** paper. 

The goal of the gradient penalty is to ensure that small changes in the input (real or generated images) lead to small, gradual changes in the discriminator's output. To achieve this, the authors of WGAN-GP introduced a method that interpolates between real and fake images, then encourages the discriminator to respond smoothly to these changes by penalizing large gradients. This helps improve the stability and performance of the model during training.


```Python
def gradient_penalty(disc, real, fake, alpha, train_step, device="cpu"):

	BATCH_SIZE, C, H, W = real.shape
	beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
	interpolated_images = real * beta + fake.detach() * (1 - beta)
	interpolated_images.requires_grad_(True)
	
	mixed_scores = disc(interpolated_images, alpha, train_step)
	gradient = torch.autograd.grad(
	inputs=interpolated_images,
	outputs=mixed_scores,
	grad_outputs=torch.ones_like(mixed_scores),
	create_graph=True,
	retain_graph=True,
	)[0]
	
	gradient = gradient.view(gradient.shape[0], -1)
	gradient_norm = gradient.norm(2, dim=1)
	gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
	
	return gradient_penalty
```

The model implementation was heavily inspired by a digital Ocean blog by [Abd Elilah TAUIL](https://blog.paperspace.com/author/abd/).

## Data Augmentation

```Python
augmentation_transforms = transforms.Compose([
	transforms.Grayscale(num_output_channels=1),
	transforms.Resize((image_size, image_size)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor()
])
```

To reduce overfitting, the following augmentations were applied. Random horizontal flips were performed, but vertical flips were excluded.

Additionally, the images were converted to grayscale with a single channel to improve training efficiency.

The image size was progressively adjusted throughout training, starting from 4x4 and scaling up to 256x256 per batch. This process will be discussed in detail in the following section.

## Training 

Due to the length and intensity of training, the model was trained on an external H100 gpu. The 2 data classes AD and NC were trained separately, in order to allow us to intentionally generate each class. 

### Test run 1 

The initial run had wide range of issues. For this model I attempted to directly generate 256 x 256 images, which manifested poorly as the model did not converge resulting in unrecognisable images. 

<img src="recognition/Readme_images/image copy.png" alt="Initial Test Results" width="500"/>

This issue occurred due to an error in the data augmentation and generator setup. During this test run, the input image was processed with 3 channels, and augmentations such as saturation adjustments, blurring, and discoloration were applied. This led the model to mistakenly interpret the image as colored. Additionally, using 3 channels caused the generator and discriminator to converge more slowly, contributing to the poor image quality observed above.

**Adjustments:**

1. The generator and data augmenter we set to only output greyscale images with single channels
2.  The image sizes were progressively increased `IMAGE_SIZES = [4, 8, 16, 32, 64, 128, 256]`


### Test run 2

The images produces after the initial test 2 were significantly better. However, there was still one major problem related to GPU resource allocation. 

It was notices that during the training CPU usage would remain relatively low and would occasionally drop off. 

<img src="recognition/Readme_images/CPU Usage.png" alt="Initial Test Results" width="500"/>

This indicated that there was room to further utilise the CPU or the the model was being bottle necked in a particular location. A similar problem was faced with the GPU utilisation, where particularly in the earlier image sizes **(4 to 64)**, the GPU was being under utilised. 

<img src="recognition/Readme_images/GPU Usage.png" alt="Initial Test Results" width="500"/>

**Adjustments:**
1. To further utilise the CPU the number of workers used in the data loader was increased from **6 -> 10**
2. In addition the batch sizes were doubled each image size 

`BATCH_SIZES = {4: 256, 8: 128, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4}`

`BATCH_SIZES = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8}`

## Training Results 

The following plots show the progressive output from the generator at each image size, slowly increasing from 4, 8 , 16, 32, 64, 128 and finally 256. In the earlier stages, the images appear highly pixelated due to the low resolution. Some blurring was unintentionally introduced by the image scaling software used during processing.

![NC Image Progress](/recognition/Readme_images/NC image progress.png)

![AC Image Progress](/recognition/Readme_images/AD image progress.png)