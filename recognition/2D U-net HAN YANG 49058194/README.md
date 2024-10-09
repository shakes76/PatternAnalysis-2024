# COMP3710-Report

## Introduction
This project segments MRI images of the prostate using a 2D U-Net model with all labels having a minimum Dice similarity coefficient
of 0.75 on the test set on the prostate label.

## Model Architecture and Introduction
This model is a 2D UNet with encoder decoder structure and skip connections[^1].<br />
<br />
       ![image text](https://github.com/yanghan8458/COMP3710-Report/blob/main/%E8%81%94%E6%83%B3%E6%88%AA%E5%9B%BE_20241009103621.png "DBSCAN Performance Comparison")
### The core components of U-Net：
- **Encoder**：<br />
Also known as compression path or downsampling path, it is mainly used to build deep network structures and extract deep semantic information. It includes multiple program blocks, each typically consisting of a 3x3 convolution (using Relu activation function) and a 2x2 pooling layer with a stride of 2 (downsampling). After each program block is processed, the size of the feature map will gradually decrease.
- **Decoder**：<br />
Also known as the extension path or upsampling path, it is symmetrical to the encoder part. It also includes multiple program blocks, each of which increases the size of the feature map through a 2x2 upsampling operation with a step size of 2, and then concatenates the feature maps with the corresponding levels of the encoder, i.e. concatenation. Finally, further processing is performed through two 3x3 convolutions (Relu).
- **Skip Connection**：<br />
Located between the encoder and decoder, it typically consists of two 3x3 convolutional layers. Its function is to further extract and fuse features, providing more useful information for the decoder.
<br />
The 2D U-Net model used here operates on individual 2D slices of MRI images, making it computationally efficient while still leveraging the spatial information inherent in the data. By using a binary cross-entropy loss function combined with the Dice similarity coefficient as an evaluation metric, this model is optimized to accurately segment the prostate with minimal overlap error[^2].

## How It Works
The U-Net model employs a convolutional neural network architecture specifically designed for image segmentation tasks. It consists of an encoder-decoder structure with skip connections that enable the model to preserve spatial information while capturing context from the input images.<br />
<br />
The process begins with loading Nifti files containing MRI slices using the Nibabel library, converting them into NumPy arrays for preprocessing. Each slice is then resized and normalized to ensure uniform input dimensions of `[1, 1, 128, 128]` for grayscale images. <br />
<br />
During training, the model minimizes the binary cross-entropy loss, using the Dice similarity coefficient as a metric for evaluation. Once trained, the model can predict segmentation masks for new MRI slices, providing a clear delineation of the prostate gland against the surrounding tissues, which is crucial for accurate diagnosis and treatment planning.


## File Structure and Descriptions
- `download.py`
  - **Purpose**: Downloads MRI data from a specified URL and processes `.nii` and `.nii.g`z files into `.npy` format for training.
  - **Key Functions**:
    - `download_and_extract(url, extract_to)`: Downloads and extracts the dataset.
    - `load_and_process_nii_files(root_dir, save_dir)`: Loads and resizes NIfTI images to 2D slices, then saves them as NumPy arrays.
- `modules.py`
  - **Purpose**: Implements the U-Net model architecture used for MRI image segmentation.
  - **Key Components**:
    - `DoubleConv`: Two sequential convolutional layers.
    - `Down`: Max pooling followed by double convolution.
    - `Up`: Upsampling and concatenation.
    - `OutConv`: Final layer to output the segmentation map.
- `dataset.py`
  - **Purpose**: Defines a custom PyTorch `Dataset` class, `ProstateMRIDataset`, to load the preprocessed MRI slices (`.npy` files).
  - **Key Functions**:
    - `__len__()`: Returns the total number of images.
    - `__getitem__(idx)`: Loads an individual image as a PyTorch tensor.
- `train.py`
  - **Purpose**: Handles model training using the U-Net architecture.
  - **Key Functions**:
    - `train_model(root_dir, num_epochs, lr)`: Trains the U-Net model on the dataset, saves the model, and logs the loss.
- `predict.py`
  - **Purpose**: Evaluates the trained model and computes the Dice Similarity Coefficient for segmentation accuracy.
  - **Key Functions**:
    - `dice_score(pred, target)`: Calculates the Dice score.
    - `predict_and_evaluate(root_dir, model_path)`: Loads the model and evaluates it on the test dataset.
- `test_driver.py`
  - **Purpose**: Test driver script that calls and runs algorithm.
  - **Key Steps**:
    - Download and preprocess data.
    - Train the U-Net model.
    - Evaluate and predict the segmentation quality using the trained model.
- `README.md`
  - **Purpose**: Project documentation.
 
## Dependencies
The project requires the following dependencies[^5]:
- `torch==1.10.0`: PyTorch for building and training the model.
- `numpy==1.21.0`: NumPy for handling numerical operations and array manipulations.
- `nibabel==3.2.1`: Nibabel for loading Nifti files.
- `matplotlib==3.4.3`: Matplotlib for visualizing results and training metrics.

## Usage：Steps to Run the Project

- ### Step 1. Download and Process Data
&nbsp; &nbsp; Run `download.py` to download the `.nii` files from the given URL, process them, and convert them to `.npy` format.

- ### Step 2. Train the Model
&nbsp; &nbsp; Once the `.npy` files are prepared, train the U-Net model.

- ### Step 3. Make Predictions and Evaluate
&nbsp; &nbsp; After training, using `predict.py` to evaluate the model's performance on the test set. It calculates the Dice score to assess segmentation quality.

- ### Step 4. Full Pipeline Execution
&nbsp; &nbsp; The entire pipeline (from data download to model evaluation) can be executed via `test_driver.py`.

## Details
- **Loss function**: `Binary cross entropy with Logits`.
- **Optimizer**: `Adam`.
- **Evaluation metric**: `Dice similarity coefficient`[^6].
  ### Training
  #### 1. Inputs:
  The inputs to the `train.py` script include:
  - **Training data**: <br />Preprocessed images (such as slices from MRI scans) and corresponding ground truth segmentation masks.
    - Images: `[batch_size, 1, 128, 128]`, where `batch_size=128` <br />`1` represents the grayscale channel, and `128x128` are the spatial dimensions of the image.
      - Example: `torch.Size([128, 1, 128, 128])`
    - Masks (ground truth): `[batch_size, 1, 128, 128]`, similarly structured as the images, but representing the ground truth segmentation masks. <br />
 Thus, the masks and images will both have the same dimensions: `[128, 1, 128, 128]` for a batch of 128 grayscale images with corresponding binary masks.
  - **Hyperparameters**: <br />Training settings like the learning rate, number of epochs, batch size, etc. These could be specified within the script or passed via a configuration file or command-line arguments. 
    - `learning_rate = 0.001`
    - `num_epochs = 30`
    - `batch_size = 128`
  - **Model architecture**: <br />The `UNet` model itself, which is built and instantiated inside the script.
    - This involves defining the model and initializing it with the desired number of input/output channels, as I have done with `n_channels=1` (grayscale input) and `n_classes=1` (binary segmentation).
  - **Optimizer and loss function**: <br />Components needed to train the model.
    - Optimizer: `Adam`.
    - Loss function: `Binary Cross Entropy`, `Dice Loss`.
  #### 2. Outputs:
  The outputs typically include:
  - **Trained model weights**:<br /> The model's learned parameters after training. These are often saved in a file, such as `unet_model.pth`, at the end of training for future use.
    - This is done using `torch.save(model.state_dict(), 'unet_model.pth')` so that we can later reload the model for inference or further training.
  - **Training logs**: <br />These usually include the loss value for each epoch and sometimes additional metrics (e.g., Dice score).
    - For instance, in each epoch in my `train.py`, its output is as follows <br />
      Epoch [1/30], Loss: 0.4332 <br />
      Epoch [2/30], Loss: 0.3708 <br />
      Epoch [3/30], Loss: 0.3142 <br />
      Epoch [4/30], Loss: 0.2643 <br />
      Epoch [5/30], Loss: 0.2347 <br />
      Epoch [6/30], Loss: 0.2208 <br />
      Epoch [7/30], Loss: 0.2127 <br />
      Epoch [8/30], Loss: 0.2099 <br />
      Epoch [9/30], Loss: 0.2075 <br />
      Epoch [10/30], Loss: 0.2060 <br />
      Epoch [11/30], Loss: 0.2054 <br />
      Epoch [12/30], Loss: 0.2042 <br />
      Epoch [13/30], Loss: 0.2030 <br />
      Epoch [14/30], Loss: 0.2022 <br />
      Epoch [15/30], Loss: 0.2011 <br />
      Epoch [16/30], Loss: 0.2002 <br />
      Epoch [17/30], Loss: 0.1994 <br />
      Epoch [18/30], Loss: 0.1983 <br />
      Epoch [19/30], Loss: 0.1977 <br />
      Epoch [20/30], Loss: 0.1967 <br />
      Epoch [21/30], Loss: 0.1964 <br />
      Epoch [22/30], Loss: 0.1960 <br />
      Epoch [23/30], Loss: 0.1955 <br />
      Epoch [24/30], Loss: 0.1953 <br />
      Epoch [25/30], Loss: 0.1951 <br />
      Epoch [26/30], Loss: 0.1952 <br />
      Epoch [27/30], Loss: 0.1951<br />
      Epoch [28/30], Loss: 0.1951<br />
      Epoch [29/30], Loss: 0.1950<br />
      Epoch [30/30], Loss: 0.1950.<br />
  - **Graphs/Plots**: <br />Visual representation of the training process, often showing:
    - Loss curve: A plot of training/validation loss over the epochs.
     ![image text](https://github.com/yanghan8458/COMP3710-Report/blob/main/Figure_1.png "DBSCAN Performance Comparison")
  ### Predicting
  #### 1. Inputs:
  - **Model**:<br /> The `UNet` model is loaded from the `unet_model.pth` file.
    - This model expects grayscale MRI slices of shape `[1, 1, 128, 128]` as input.
    - The `unet_model.pth` file contains the trained weights of the model, which are loaded using `torch.load`.
  - **Image**: <br />A single MRI slice that is fed into the model for prediction. Each input image has a shape of `[1, 1, 128, 128]`, where:
    - `1`: Corresponds to the batch size (in this case, just a single image).
    - `1`: Corresponds to the number of channels (grayscale, single channel).
    - `128x128`: Represents the height and width of the image (spatial dimensions).
  #### 2. Output:
  - **Prediction**: <br />The output of the model is a binary segmentation mask of the same shape as the input image.
    - Original Image：<br />This image is the original MRI slice image input to the model, usually a grayscale image, showing the cross-section of the human body with relatively clear structure.
    - Prediction Image：<br />This is the probability map output by the model, usually with values between `[0,1]`, representing the probability that each pixel belongs to the prostate (or other target). After using the `torch. sigmoid ()` function, the prediction obtained is a grayscale image, where[^7]:
      - The `target area` (such as the prostate) will be displayed as a brighter area.
      - The `background area` is a darker (almost black) region.
      - The bright spots in the prediction should cover the target area in the original image well, and the boundaries may be slightly blurred, but the overall shape should be close to the contour of the target organ.
    - Prediction results[^8]:
      - `Target area`: The prostate or other targets are highlighted and clearly segmented from the background, with slightly blurred edges.
      - `Background area`: The background pixels remain close to black because they do not belong to the target area.
      - `Prediction error`: When the loss is 0.2, there are some false positive or false negative areas (i.e. the model incorrectly predicts certain backgrounds as target areas or misses a part of the target), but the overall effect is still close to the true target.<br />
      ![image text](https://github.com/yanghan8458/COMP3710-Report/blob/main/Prediction%20figure.png "DBSCAN Performance Comparison")
In the prediction results, the prostate area seems to be well identified and segmented by the model, especially the white area in the middle should be the target area. The background parts of other areas were not misclassified, as expected, indicating that the model has good ability to distinguish between the background and the target area. Although there are slight differences and noise, especially on some bright spots in the lower right corner (white spots), the overall segmentation effect is reasonable.
  - **Evaluation**: <br />The Dice score is computed to evaluate the quality of the segmentation. The predicted segmentation mask is compared to the ground truth mask (if available) to compute this metric[^9].
    - The `Dice score` (or `Dice similarity coefficient`) measures how well the predicted segmentation mask matches the ground truth mask. It ranges between 0 and 1, where:
      - `0` means no overlap between the prediction and ground truth.
      - `1` means perfect overlap between the prediction and ground truth.
    - To compute the Dice score, you need both the predicted mask and the ground truth mask for each image. The Dice score is calculated using the following formula:
       ![image text](https://github.com/yanghan8458/COMP3710-Report/blob/main/formula.png "DBSCAN Performance Comparison")
      This can be broken down into:
      - `|Prediction∩Ground Truth|`: The number of pixels where both the predicted and ground truth masks are 1 (the intersection of the two masks).
      - `|Prediction|`: The number of pixels in the predicted mask that are labeled as 1 (the sum of pixels in the prediction).
      - `|Ground Truth|`: The number of pixels in the ground truth mask that are labeled as 1 (the sum of pixels in the ground truth).
    - In prostate segmentation, a Dice score of `0.75` or `above` is considered reasonable or good, meaning the model has learned to segment the prostate with fairly high accuracy.
    - In my `predict.py` code. <br />Output: `Model achieved the desired Dice score of 0.75 or above: 0.81`.  
 
## Result
The model successfully achieve a Dice score of at least 0.75 on the test set.<br />
With `test_driver.py` output: `Model achieved the desired Dice score of 0.75 or above: 0.81`.

---

## References
[^1]: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *Medical Image Computing and Computer-Assisted Intervention* (MICCAI), 234-241.
[^2]: Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J. (2017). Generalised Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations. 2017 International Workshop on Deep Learning in Medical Image Analysis (DLMIA). 
[^5]: Paszke, Adam, et al. "PyTorch: An imperative style, high-performance deep learning library." *Advances in neural information processing systems* 32 (2019).
[^6]: Taha, A. Asem, and Allan Hanbury. "Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool." *BMC medical imaging* 15.1 (2015): 29.
[^7]: Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. Medical Image Computing and Computer-Assisted Intervention (MICCAI). Springer, LNCS, Vol. 9351, pp. 234–241. 
[^8]: Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203–211. 
[^9]: Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss function for highly unbalanced segmentations." *Deep learning in medical image analysis and multimodal learning for clinical decision support*. Springer, Cham, 2017.
