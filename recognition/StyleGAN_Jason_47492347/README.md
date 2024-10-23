# StyleGAN Brain MRI Generative Model
This is an image generation model trained on the ADNI brain dataset with the goal of producing "reasonably clear" deepfake images of brain MRI's.


## Overview
The field of medical imaging is currently undergoing significant changes due to recent advancements in machine learning, which is being utilised for tasks such as disease detection and image enhancement. Despite this progress, there are many challenges that remain, one of which is that medical datasets are often lacking in size and diversity. Many tasks in machine learning require subtantial amounts of data to achieve meaningful solutions. This project aims to address this problem by training a generative model capable of producing images that imitate real world data, specifically brain MRI scans.


## StyleGAN Lore

### What is a GAN?
Generative Adversarial Network (GAN) is a machine learning framework developed by scientist Ian Goodfellow and his team in 2014. At a high level, it can be conceptualised as an evolutionary competition between two models, the Generator and the Discriminator. The Generator's task is to gradually improve at producing fake imitations of the training data, while the Discriminator, which is trained alongside, is tasked with evaluating whether the images it receives are from the realdata set or produced by the Generator. Over many iterations of this process, the Generator eventually becomes capable of creating realistic deepfake images (in this case) that are difficult for the Discriminator to distinguish from the original dataset.

<figure align="center">
    <img src=assets/Generative_adversarial_network.svg.png>
    <figcaption>Visualisation of GAN</figcaption>
</figure>

### StyleGAN and its improvements
While traditional GANs were able to display impressive potential, a key limitation was that there was a lack of control over the outputs, especially in terms of semantics and details. The StyleGAN architecture, first introduced in a 2018 paper by Nvidia researchers, was a revision of the GAN with noticeable improvements- almost a complete restructure of the Generator's design resulted in a model that allows for much better fine-tuned control over image outputs. This was achieved by the incorporation of the concept of "style", which allows for the Generator to separate different components of the image and control them independently of one another. Instead of starting with a single latent vector $Z$, StyleGAN contains a mapping network that first transforms $Z$ into an intermediate latent space $W$, which is then used to feed adjustment inputs at different layers of the model. Through this process, we can address the problem of "entanglement" in traditional GANs, where adjusting for a certain feature of an image would affect many other unrelated features. Overall, the StyleGAN archetecture allows for more intuitive and structured control over the Generator outputs, ultimately leading to better results.

<figure align="center">
    <img src=assets/GAN-StyleGAN-Comparison.png>
    <figcaption>GAN/StyleGAN Comparison</figcaption>
</figure>

## ADNI Dataset and Data Preprocessing
The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset is a "comprehensive and widely used collection of longitudinal clinical, imaging, genetic, and other biomarker data". The specific dataset used for this project was a collection of real 2D brain MRI scans presented as side-view jpeg images. The data is divided into two categories- Normal Control (NC) for healthy brains, and Alzheimer's Disease (AD). While each subset were also divided into train/test splits initially, I combined them together into just the training dataset- this specific generative task did not require a train/test/validate split. This resulted in roughly 15,000 images per train dataset.

Simple augmentation was performed on the data- a resize to 128x128 pixels for quicker training times, a greyscale conversion to use 1 channel, and normalisation using mean=0.5, stdev=0.5 for easier computations. I decided against using traditional methods such as random rotations or flips to maintain the consistence in orientation of the images in the original dataset.


## StyleGAN Architecture Details

<figure align="center">
    <img src=assets/StyleGAN_architecture.png>
    <figcaption>StyleGAN Architecture</figcaption>
</figure>

### Progressive Growing
The original StyleGAN paper is actually an improvement based on a preceding improvement called ProGAN, which is an archetectural change to the traditional GAN where instead of training the model at full resolution, we start at a very small image size (8x8 in this case) and progressively grow the image throughout the training process. This method has shown to result in a much more stable training process and overall better results.

### Mapping Network
The mapping network is a simple multi-layer perceptron (MLP). It is responsible for the aforementioned mapping of latent vector $Z$ into the intermediate latent space $W$. Initially, pixel normalization is performed on the input (following the structure given by the ProGAN paper), and it then passes through eight weight scaled linear layers separated by ReLU activation functions.

<figure align="center">
    <img src=assets/feature_map.png>
    <figcaption>Feature Mapping</figcaption>
</figure>

### Generator
The Generator starts takes a small random noise vector (8x8 pixels in this case) as input, and gradually converts them into higher resolution image outputs. At every image resolution, the structure contains a series of convolutional layers for feature extraction (much like a traditional GAN), Adaptive Instance Normalization (AdaIN), and random noise injections for extra variability. The AdaIN step is where the style injection happens using the $W$ space. Unlike other normalization techniques commonly used in ML such as GroupNorm/BatchNorm, Instance Normalization is done on individual "instances"- per channel for each individual image, in this case. The "Adaptive" part of this process is referring to how the mean and variance are "aligned" with the style feature maps given by the mapping network- this is how "style transfer" happens. Finally, the upsampling process from a lower to higher resolution uses a parameter we call $\alpha$, which is linearly interpolated from 0 to 1, we smoothly "fade in" new layers into the image until it has grown to the desired size. These main processes are looped until we have reached the final resolution size and all transformations have been completed.

<figure align="center">
    <img src=assets/AdaIN_formula.png>
    <figcaption>AdaIN Operation</figcaption>
</figure>

### Discriminator
The structure of the Discriminator is left mostly unchanged from traditional GAN. One difference is that it also trains alongside the Discriminator using progressively growing images. It contains convolutional layers paired with LeakyReLU activation functions for feature extraction, gradually downsampling until it can map to a prediction vector output.


## Results
For each run of training, a set of example images were generated using the current state of the model for each resolution step, resulting in 5 sets of generations per model (including the final results). I also plotted the values of both generator and critic (discriminator) loss functions per resolution step.


### 2 Epoch Trial Run (AD)
I initially trained a model on the AD dataset for 2 epochs to test if it was running correctly. The final generations at step 5 already showed promising results; I could tell that the progressively growing architecture was proabably what was doing most of the heavy lifting here.

<figure align="center">
    <figcaption>2 Epoch (AD) Trial Run Results</figcaption>
    <img src=assets/2_Epoch_AD_Generations_4x4.png style="width:360px;height:360px">
</figure>

### 20 Epoch Trial Run (AD)
Without having a good idea of when would be a good time to stop training, I decided to run my model model for 10 times the epochs for a total of 20.

<figure align="center">
    <figcaption>20 Epoch (AD) Trial Run Results</figcaption>
    <img src=assets/20_Epoch_AD_Generations_4x4.png style="width:360px;height:360px">
</figure>

<figure align="center">
    <figcaption>20 Epoch Trial Run Loss Plots: Steps 1 to 5</figcaption>
    <img src=assets/20AD_lossplot_batch.png>
</figure>

### 12 Epoch Final Models
From the 20 epoch run, it was clear to see that the loss values became stabilised and were no longer getting closer to 0 from around halfway to three fifths of the way into training. Therefore, I decided that I would set the training loop to run for 12 epochs per resolution for my final models. Here are the results and per batch/epoch loss plots for my models:

<figure align="center">
    <figcaption>12 Epoch AD Generations</figcaption>
    <img src=assets/12_Epoch_AD_Generations_4x4.png style="width:360px;height:360px">
</figure>

<figure align="center">
    <figcaption>12 Epoch NC Generations</figcaption>
    <img src=assets/12_Epoch_NC_Generations_4x4.png style="width:360px;height:360px">
</figure>

<figure align="center">
    <figcaption>AD Loss Per Batch: Steps 1 to 5</figcaption>
    <img src=assets/12AD_lossplot_batch.png>
</figure>

<figure align="center">
    <figcaption>AD Mean Loss Per Epoch: Steps 1 to 5</figcaption>
    <img src=assets/12AD_lossplot_epoch.png>
</figure>

<figure align="center">
    <figcaption>NC Loss Per Batch: Steps 1 to 5</figcaption>
    <img src=assets/12NC_lossplot_batch.png>
</figure>

<figure align="center">
    <figcaption>NC Mean Loss Per Epoch: Steps 1 to 5</figcaption>
    <img src=assets/12NC_lossplot_epoch.png>
</figure>

### UMAP Embeddings
In an attempt to further gauge the performance of the models, specifically its ability to generate sufficiently different images depending on whether it was trained on the AD dataset or the NC dataset, I tried a UMAP embedding using 1000 sample generations from each model.

<figure align="center">
    <figcaption>UMAP Embeddings Plot: (Red: AD, Purple:NC)</figcaption>
    <img src=assets/UMAP.png>
</figure>

Unfortunately, it appeared so that perhaps the difference between AD and NC brain MRI's were too similar to each other by nature, and therefore the UMAP algorithm (or my basic attempt to perform it) was unable to produce visually meaningful results with datapoints clustered around others of their class. To make sure that this was not due to the GAN generations themselves being the issue, I also ran the algorithm using ground truth:

<figure align="center">
    <figcaption>ADNI Dataset UMAP Embeddings Plot: (Red: AD, Purple:NC)</figcaption>
    <img src=assets/UMAP_ADNI.png>
</figure>

## Dependencies
If you want to try running this yourself, this is a list of dependencies which I know for sure is sufficient for this project, however earlier versions of some of them may also work (I have not been able to test for other versions). Newer versions are likely to work. My training was done using an Nvidia 40 series video card and ran on Windows 11- you will probably need to tweak some of the code to get it running on AMD cards or on different OS.

Dependency | Version
---------- | :-------:
Python | 3.10+
PyTorch | 2.4.0
PyTorch CUDA | 12.4
Nvidia CUDA | 12.6
Numpy | 1.26.4
Torchvision | 0.19.0
Matplotlib | 3.8.4
tqdm | 4.66.5
umap-learn | 0.5.6


## Generating Images with a Pre-Trained Model
Run the following script to generate images using a model that has already been trained and saved (you will have to train one first, and store it in the saved_models directory). Specify the model, output name, number of images to generate, and the seed.

```
python predict.py --model <model_name> --output <output_name> --n <number_of_images> [--seed <random_seed>]
```

### Example Usage:

```
python generate_images.py --model gen_my_model --output example_outputs --n 10 --seed 69
```

The generated images will be saved to your SRC directory under the following structure:
```
generated_images/
|-- example_outputs/
    |-- img_0.png
    |-- img_1.png
    |-- img_2.png
    |-- ...
    |-- img_9.png
```


## Extension Areas
I was overall satisfied with the results I was able to achieve with the StyleGAN architecture. To account for hardware deficiencies (only one GPU) and time constraints, I chose to train my models at lower resolutions- this could potentially be an area of improvement. In addition, attemping a different dimension reduction visualisation algorithm such as t-SNE may produce some more meaningful results. Further improvements could also be made by using more advanced image generation architectures such as StyleGAN2/StyleGAN3/DDPM.


## References
https://arxiv.org/pdf/1710.10196

http://arxiv.org/pdf/1812.04948

https://arxiv.org/abs/1912.04958

https://adni.loni.usc.edu/data-samples/adni-data/

https://blog.paperspace.com/implementation-of-progan-from-scratch/

https://blog.paperspace.com/implementation-stylegan-from-scratch/#models-implementation

https://umap-learn.readthedocs.io/en/latest/basic_usage.html
