# UNet_PriankaIndla

## U-Net for MRI Image Segmentation

### 1.  Description of Model

The U-Net model is a convolutional neural network designed for biomedical image segmentation. It addresses the challenge of segmenting complex medical images, such as MRI scans, by using an encoder-decoder structure that combines high-resolution and low-resolution feature maps through skip connections. The U-Net architecture is particularly well-suited for tasks where precise locating is essential.

**Problem Solved**: In this project, the U-Net model is applied to prostate regions in MRI scans as part of the HipMRI Study on Prostate Cancer. Accurate segmentation of prostate tumours within MRI scans aids clinicians in identifying regions of interest, which is crucial for diagnosing and planning treatments for prostate cancer. This project aims to achieve high segmentation accuracy, targeting a Dice similarity coefficient of at least 0.75 on the test set. This metric indicates how well the model's segmented regions match the ground truth, reflecting the model’s reliability in identifying tumour boundaries accurately. 

### Dataset, Pre-processing and Data Splits

**Dataset**: The dataset consists of pre-processed 2D slices in Nifti file format, which must be read and processed in a specific way to enable accurate analysis. Each 2D slice represents a cross-section of the MRI scan, and these slices are handled individually to produce segmentation masks labelling critical prostate regions.

**Data Splits**:
The data is split into training, validation, and testing sets. The training set enables the model to learn, the validation set is used for hyperparameter tuning and avoiding overfitting, and the test set evaluates the model’s ability to identify the tumour segment.

**Pre-processing**:

1. **Resizing**: Images are resized to a fixed shape of 256x128 to standardise input dimensions.
2. **Normalisation**: Pixel values are normalised to a range of [0,1] b. This helps improve model convergence and stabilises training.

### 2. Algorithm Workflow

The U-Net architecture consists of a symmetric encoder-decoder structure:

- **Encoder (Downsampling)**: Extracts high-level features from the input image using convolution and pooling layers. The encoder gradually reduces the spatial dimensions, capturing important information.
- **Bottleneck**: Optimises loss function, therefore, increases the efficiency of the model without losing much important information.
- **Decoder (Upsampling)**: Recovers spatial details and upscales the feature maps to the original input size using transpose convolutions. Skip connections between corresponding encoder and decoder layers help in retaining smaller details.

### **1. modules.py**

This file defines the U-Net architecture.

**Core Components**:

1. **U-Net Architecture Overview**:
    - **Encoder (Downsampling)**: Extracts high-level features from the input image using convolution and pooling layers. The encoder gradually reduces the spatial dimensions, capturing important information.
    - **Bottleneck**: Optimises loss function, therefore, increases the efficiency of the model without losing much important information.
    - **Decoder (Upsampling)**: Recovers spatial details and upscales the feature maps to the original input size using transpose convolutions. Skip connections between corresponding encoder and decoder layers help in retaining smaller details.
2. **Helper Function: double_conv**:
    - The double_conv method is used repeatedly in the U-Net architecture. It consists of two convolutional layers, each followed by batch normalisation and ReLU activation.

### 2. dataset.py

This file is responsible for **data loading and preprocessing**. It defines a custom dataset class called ProstateMRIDataset, which is used by PyTorch Dataloader to efficiently manage loading batches of images during training and prediction.

**Core Components**:

- **Data Paths**: The dataset class is initialised with paths to the image and segmentation files.
- **Preprocessing**: Images and labels are loaded and resized to the U-Net model’s input dimensions.
- **Tensor Conversion**: Each image and label pair is converted to tensors so they’re compatible with PyTorch.

### 3. train.py

This file is where the U-Net model is trained. It handles loading data, initialising the model, training, validating, and saving the best-performing model.

**Core Workflow**:

- **Data Splitting**: Images and segmentation mask datasets are defined.
- **Training Loop**:
    - **Forward Pass**: Sends images through the U-Net model.
    - **Loss Calculation**: Measures the difference between predictions and ground truth using a loss function, specifically BCELogitsLoss for binary segmentation.
    - **Backpropagation**: Optimises the model weights to minimise the loss.
- **Validation**: After each epoch, the model’s performance on the validation set is evaluated using Dice score, which is a metric used to measure segmentation quality.
- **Saving the Best Model**: The model with the best Dice score on the validation set is saved as best_model.pth.

### 4. predict.py

This file is designed for model inference on new data, applying the trained U-Net model to unseen images.

**Core Workflow**:

- **Load Model**: Loads the saved model weights from best_model.pth.
- **Data Loading**: Uses the dataset and dataloader but here it only loads the image and segmentation test dataset.
- **Prediction**:
    - **Forward Pass**: Generates a segmentation prediction for each test image.
    - **Postprocessing**: Applies thresholding to generate binary predictions and optionally resizes the output to match the input dimensions.
- **Visualisation**: Saves original image and ground truth segmentation, providing a visual assessment of the model's accuracy. This is useful for evaluating the model’s segmentation quality.

### 3. Dependencies and Reproducibility

To ensure reproducibility, the following dependencies must be installed to create the correct environment (NOTE: environment name is ***torch_env)***

```

- Python == 3.8
- torch == 2.0.1
- torchvision == 0.15.2
- nibabel == 5.1.0
- numpy == 1.24.3
- scikit-image == 0.21.0
- tqdm == 4.66.1
- matplotlib == 3.7.2

```

To install the dependencies, you can run:

```bash

pip install torch==2.0.1 torchvision==0.15.2 nibabel==5.1.0 numpy==1.24.3 scikit-image==0.21.0 tqdm==4.66.1 matplotlib==3.7.2

```

### 4. Results

Plots and images were saved using the terminal path example:

```r
scp s4749392@rangpur.compute.eait.uq.edu.au:/home/Student/s4749392/dice_score_plot.png ~/Desktop/

```
The Dice Score plot provides insight into the model’s segmentation performance over the training epochs. By the 13th epoch, the Dice Score surpasses the target threshold of 0.7. This value indicates that the overlap between the predicted segmentation and the ground truth is around 70% or better, showing the model’s success. However, the dice score is decreasing meaning further training must be conducted to improve accuracy of model.

Figure 1: Dice Score over 13 Epochs 
![dice_score_plot](https://github.com/user-attachments/assets/c81055ef-881c-4cbe-824c-5500d32fed24)


Training and validating loss decreases over 13 epochs. Therefore, the accuracy of the model increases as more epochs run. The steady decrease in training loss implies that the model is gradually improving its accuracy. The steep and unstable validation loss may indicate overfitting, therefore, further analysis of model parameters must be conducted to optimise model.

Figure 2: Train and Validation Loss over 13 Epochs
![loss_plot](https://github.com/user-attachments/assets/ff686a05-9389-44c7-965e-445cfa0b10c4)


The results outputted after the predict.py file was run is shown in Figures 3 to Figure 7. Each figure shows an original image randomly chosen from the test dataset, its corresponding ground truth and the predicted segmentation this model outputted. It can be seen that the segments overlap ground truth segment but overfitting can be seen, therefore, further trianiing must occur to fine tune model.

Figure 3: Image 1 chosen for prediction
![prediction_0](https://github.com/user-attachments/assets/049205b6-8b7e-4170-9a15-c52ada49cc04)

Figure 4: Image 2 chosen for prediction
![prediction_1](https://github.com/user-attachments/assets/09c0979a-6cf7-4ab6-8cb4-5aac87b541b7)

Figure 5: Image 3 chosen for prediction
![prediction_2](https://github.com/user-attachments/assets/e342310e-63ad-46a7-a00a-cff31189fea6)

Figure 6: Image 4 chosen for prediction
![prediction_3](https://github.com/user-attachments/assets/de590168-adcc-4f1d-975b-bd9feab5dca0)

Figure 7: Image 5 chosen for prediction
![prediction_4](https://github.com/user-attachments/assets/16093749-2903-480b-9195-cf72fedd3c39)





