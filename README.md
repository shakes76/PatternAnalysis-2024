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
