# StyleGAN for ADNI Brain Dataset

This project implements a StyleGAN model to generate synthetic brain images using the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset. The model is designed to learn the distribution of brain images and generate new samples that resemble the training data.

## Table of Contents

- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)




## Installation
- pip
- tensorflow
- matplotlib
- scikit-learn

## Dataset

The model uses the ADNI dataset, which can be downloaded from the [ADNI website](http://adni.loni.usc.edu/). The dataset should be organized in the following directory structure:

├── train/ <br/>
│&emsp;&emsp;├── AD/ <br/>
│&emsp;&emsp;└── NC/ <br/>
└── test/ <br/>
&emsp;&emsp;├── AD/ <br/>
&emsp;&emsp;└── NC/ <br/>


Make sure to place the images in the respective folders for training and testing.

## Usage

1. **Training the Model**: To train the StyleGAN model, run the `train.py` script. This will load the dataset, train the model, and generate sample images.

```
python train.py
```

2. **Generating Images**: After training, you can generate new images using the trained model. Run the `predict.py` script.

```
python generate_images.py
```

3. **t-SNE Visualization**: After training, the script will generate a t-SNE visualization of the real and generated images, saved as `t-SNE_visualization.png`.


## Model Architecture

The StyleGAN model consists of the following components:

- **Generator**: Generates synthetic images from random latent vectors.
- **Discriminator**: Evaluates the authenticity of the generated images.
- **AdaIN Layer**: Adaptive Instance Normalization layer used in the generator to control the style of the generated images.

The model is built using TensorFlow and Keras, and the architecture is defined in the `modules.py` file.

## Results

After training, the model generates synthetic brain images that resemble the training data. The quality of the generated images can be evaluated visually and through the discriminator's output.

## License

--------------------------------

## Acknowledgments

- [ADNI](http://adni.loni.usc.edu/) for providing the dataset.
- [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/) for the deep learning framework.
- [Scikit-learn](https://scikit-learn.org/stable/) for the t-SNE implementation.
