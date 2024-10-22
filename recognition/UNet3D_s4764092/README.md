# 3D U-Net for Prostate MRI Segmentation
###### YA HONG s4764092
This project uses **UNet-3D** to train the Prostate 3D dataset for medical volumetric image segmentation, evaluated by the **Dice similarity coefficient**. The dataset has significant **class imbalance** (e.g., background: 1,068,883,043 pixels, prostate: 1,771,500 pixels). The prostate is difficult to segment accurately due to its small volume, blurred boundaries, and the complexity of anatomical structures. To address these challenges, I applied data augmentation strategies, adjusted the model’s initial feature size (`init_features`), and experimented with various loss functions (such as **Dice loss**, weighted cross-entropy, and their combinations) to improve segmentation performance for minority classes.

## About the Model
UNet3D is a three-dimensional extension of the 2D U-Net, specifically designed to handle 3D data. This model operates directly on entire volumetric data through 3D convolutions, capturing richer spatial contextual information that is crucial for understanding complex 3D structures. Unlike the 2D U-Net, which processes images slice by slice, UNet3D maintains spatial continuity of images, preventing the loss of cross-slice information during processing—this is extremely important in medical imaging analysis. Additionally, by integrating multi-scale features and skip connections between the encoder and decoder, UNet3D significantly enhances segmentation precision, especially for small-volume and complex-shaped structures. These features make UNet3D highly effective in 3D medical imaging segmentation tasks, particularly suited for handling complex images such as prostate MRI.

### Architecture
The model consists of the following components:

- **Encoder**: Composed of 5 convolutional blocks, each consisting of two 3D convolution layers. After each convolution, Batch Normalization is applied to standardize the feature maps, followed by a ReLU activation function for non-linearity. Downsampling is performed using 3D MaxPooling layers, gradually extracting multi-scale features from the input data.

- **Bottleneck**: Positioned between the encoder and decoder, this layer further extracts deep features. Its structure is similar to the convolutional blocks, consisting of two 3D convolution layers, BatchNorm, and ReLU, designed to capture high-level features.

- **Decoder**: The decoder restores the spatial resolution of the feature maps through 3D transpose convolutions (ConvTranspose3d). After each upsampling step, skip connections are used to concatenate corresponding encoder and decoder features, ensuring high-resolution details are preserved.

- **Skip Connections**: Each decoder level is connected to the corresponding encoder level, combining feature maps to ensure critical details are not lost during upsampling, enhancing segmentation accuracy.

- **Final Convolution Layer**: A 1x1x1 convolution layer is used to generate the final segmentation output, followed by a Softmax activation function for multi-class segmentation prediction.

The process is illustrated in the following figure:

![UNet 3D Architecture](model_architecture.png)
