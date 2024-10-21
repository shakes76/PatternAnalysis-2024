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

$
\text{AdaIN}(x_i, y) = y_{s,i} \cdot \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}
$

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
