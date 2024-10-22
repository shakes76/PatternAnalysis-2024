# COM3710 - s47605790 - StyleGAN on ADNI dataset (Task 8) -  Documentation
This report will be completed with documentation about the StyleGAN model for the ADNI brain dataset, covering various aspects of implementation, data training and testing, analysis of the results, and further evaluation of the project.

## **Contents** 
- [The Problem](#The-Problem)
- [Requirements](#Requirements)
- [Dataset](#Dataset)
- [Code Structure](#Code-Structure)
- [Model implementation](#Model-Implementation)
- [Result](#Result)
- [Analysis/Evaluation](#Analysis/Evaluation)
- [References](#References)

## **The Problem**
This project involves implementing **StyleGAN** on the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** brain dataset to generate realistic brain MRI images. The task is to train a GAN (Generative Adversarial Network), specifically the **StyleGAN**. The primary challenge is to generate realistic MRI scans by progressively learning style-based features. The model combines **noise injection, adaptive instance normalization (AdaIN), and modulated convolutions (Affine Transformation)** to generate synthetic brain images.

![](https://github.com/nguy3ntt/PatternAnalysis-2024/blob/topic-recognition/recognition/47605790styleGAN_ADNI/readme_materials/stylegan_structure.jpg)

## **Requirements**
To ensure your project runs smoothly, here’s a list of dependencies along with their versions:
* Python: 3.8 or later
* PyTorch: 1.10.0 or later
* Torchvision: 0.11.0 or later
* Numpy: 1.21.2 or later
* Matplotlib: 3.4.3 or later
* PIL (Python Imaging Library) via Pillow: 8.4.0 or later
* UMAP-learn: 0.5.2 or later
* scikit-learn: 1.0.2 or later

## **Dataset**
The dataset used in this project is the ADNI dataset, which contains MRI brain scans categorized into different classes such as Alzheimer's Disease (AD) and Normal Control (NC). These scans are used for training the GAN to generate images that resemble real brain MRIs. The dataset is split into training and testing sets located at Rangpur:

* **Training set:** /home/groups/comp3710/ADNI/AD_NC/train
* **Testing set:** /home/groups/comp3710/ADNI/AD_NC/test

NOTE: These are the actual dataset locations that I used to train and test my implementation (The directories in my code are used for quick debug).

The dataset is processed in 2D, and images are resized to 128x128 pixels before being fed into the model.

## **Code Structure**
The code is structured into the following components:
* *dataset.py*: Handles the loading of the ADNI dataset and applies necessary transformations like resizing and normalization.
* *modules.py*: Contains the main components of my StyleGAN model.
* *train.py*: Contains the training loop, where the model is trained over multiple epochs, and images are generated after each epoch.
* *predict.py*: Carry out the implementation of TSNE and UMAP embeddings plot with ground truth in colors; performs testing with the pre-trained (saved after training) generator and the mapping network.
* *readme_materials*: assets for the README files (images, plots)
* *README.md*: Documentation of this project (This file)

## **Model Implementation**
Here’s a general walkthrough of my model, which contains the key components of my StyleGAN implementation:

1. **Mapping Network:**
     * The Mapping Network converts the random input vector (called z), which comes from a random distribution, into a new space called w. This transformation allows the generator to have more control over the style of the image it will generate. It's like giving the generator better tools to create different image styles.

2. **AdaIN (Adaptive Instance Normalization):**
    * AdaIN is a layer that helps the generator change the look of the images it generates. It uses the w vector (from the mapping network) to modify the internal features of the image. This allows the model to alter the style and content of the images dynamically as it generates them.

3. **Noise Injection:**
    * Noise Injection adds random noise to the image generation process at different stages. This randomness helps introduce variation in the images, which prevents the generator from always producing the same type of image. It's like adding tiny random imperfections that make the generated images more diverse and realistic.

4. **Modulated Convolution:**
    * In the Modulated Convolution layer, the generator applies w to scale (modulate) its filters. This means the convolution process, which is a fundamental part of image generation, is influenced by the style vector. This allows the generator to create images with different textures and features depending on the style it receives.

5. **Generator:**
    * The Generator is the central component responsible for transforming the latent vector z into a high-dimensional image. Beginning with a low-resolution feature map, the Generator refines this map through multiple layers, progressively increasing detail and resolution. Style information from the Mapping Network and noise are incorporated at various stages, allowing the generator to produce highly detailed and stylistically controlled images that are indistinguishable from real ones.

6. **Discriminator:**
    * The Discriminator serves as the evaluative counterpart to the Generator, determining whether an image is real (from the dataset) or fake (generated by the Generator). It processes the image through a series of down-sampling layers, extracting crucial features for making this determination. The Discriminator's ability to correctly classify images is key to improving the Generator's outputs through adversarial training.

7. **Adversarial Loss:**
    * This is the loss function that guides both the generator and discriminator during training. The discriminator tries to correctly classify images as real or fake, while the generator tries to trick the discriminator into thinking its fake images are real. The loss measures how good each is at their task and adjusts their weights accordingly.

## **Results**
For my StyleGAN implementation, the following preprocessing steps were applied:
* All input images were resized to 128x128 pixels. This resizing is performed during the dataset loading phase in the ADNIDataset class using the `transforms.Resize((128, 128))` function.
* The pixel values of the images were normalized to the range [-1, 1] using the transformation `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`. This transformation ensures that each color channel (even though the images are grayscale, they're treated as RGB with 3 channels) has pixel values within the appropriate range for the model.
Training, Validation, and Testing Splits:
* The code uses the `train_loader` which loads all the images from the train directory (AD and NC classes) for training. 
* A separate `test_loader` is used to load images from the test directory, which is in predict.py file. The test_loader provides data that the model has not seen during training.

The train.py run was used to train the generator (120 epochs), mapping network, and discriminator on the ADNI dataset in modules.py. At the end of training, the generator and mapping network models were saved and used to generate testing images (predict.py):

* **Training Loss Plot:** 

![](https://github.com/nguy3ntt/PatternAnalysis-2024/blob/topic-recognition/recognition/47605790styleGAN_ADNI/readme_materials/loss_plot.jpg)

* **Generated Images (8 images selected from Testing):**

![](https://github.com/nguy3ntt/PatternAnalysis-2024/blob/topic-recognition/recognition/47605790styleGAN_ADNI/readme_materials/test_combined.jpg)

* **UMAP embedding plot:**

![](https://github.com/nguy3ntt/PatternAnalysis-2024/blob/topic-recognition/recognition/47605790styleGAN_ADNI/readme_materials/umap.jpg)

The next section (Analysis/Evaluation) will discuss these results in detail.

## **Analysis/Evaluation**
**Loss Plot During Training:**
* Generator Loss *(blue line)*: The generator loss fluctuates significantly over the course of training, with large spikes that don't seem to stabilize much as the training progresses. This suggests that the generator is struggling to produce images that are convincing enough to fool the discriminator consistently.
* Discriminator Loss *(orange line)*: The discriminator loss stays relatively low and stable, meaning it performs well at distinguishing real images from generated ones throughout training.
* Possible Issues:
    * The generator's high loss indicates that it has difficulty minimizing its error (noisy inputs, lacking strenght in style conditioning, or an inadequate learning rate).

**Generated Test Images:**
* The test images exhibit a high level of noise, with pixelated artifacts, especially around the edges of the brain structures.
* The noise in the images indicates that the generator isn't learning to produce clear, realistic images.
* Possible Issues:
    * Insufficient model capacity, too much random noise injection, or improper loss handling.

**UMAP Embedding Plot:**
* The UMAP plot shows a clear separation between the generated and real images, with little overlap between the two classes.
* The generator is not able to learn the data distribution of the real images accurately (Can be visualize in the Result section)

Despite the model's success in generating images, the generated images contained significant noise and artifacts. The UMAP plot showed a clear separation between generated and real images, suggesting that while the generated images capture some structure, they are not yet indistinguishable from the real ones. The loss plots also indicate that there is room for improvement in stabilizing the generator and improving its learning.

*In conclusion,* while the StyleGAN implementation for this project demonstrated the capability to generate synthetic brain images, additional refinements are required to improve image quality and overall performance.

## References

T. Karras, S. Laine, andT.Aila, “AStyle-BasedGeneratorArchitecture forGenerativeAdversarial Networks,” arXiv:1812.04948 [cs, stat], Mar. 2019, arXiv: 1812.04948. [Online]. Available: http: //arxiv.org/abs/1812.04948