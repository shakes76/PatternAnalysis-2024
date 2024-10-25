# Pattern Analysis - 2D UNet Segmentation for Prostate Cancer Detection
## Description

This project implements a 2D UNet architecture for the segmentation of prostate cancer using MRI scans. The goal of the model is to accurately segment cancerous regions in MRI slices based on preprocessed medical images. The UNet is designed as a fully convolutional neural network with an encoder-decoder architecture that learns to perform pixel-wise segmentation.

The dataset used in this project consists of processed MRI slices and their corresponding segmentation masks. The segmentation task is binary, meaning the model predicts a mask where each pixel indicates either the presence or absence of cancerous tissue. This solution is implemented using PyTorch and is trained to minimize binary cross-entropy loss.

## Algorithm

- The implemented model uses a **UNet architecture**, which is well-suited for segmentation tasks due to its encoder-decoder structure. Here’s a breakdown of the working principles:

  1. **Encoder Path**: The encoder extracts features from the input image through a series of convolutional layers, with each layer followed by batch normalization and a ReLU activation. The encoder uses max pooling to downsample the feature maps, which increases the receptive field and captures more context at each successive layer.
  2. **Bottleneck**: After the encoding layers, a bottleneck layer captures the most abstracted features from the image, representing the essential information necessary to reconstruct the output segmentation mask.
  3. **Decoder Path**: The decoder upsamples the bottleneck output back to the original input resolution, using transpose convolutions for upsampling. Each decoder layer is paired with corresponding layers from the encoder via skip connections. These skip connections retain spatial detail lost during downsampling, which is crucial for pixel-accurate segmentation.
  4. **Output Layer**: The final output layer applies a sigmoid activation to produce a binary mask with values between 0 and 1, where each pixel represents the probability of being part of the cancerous region.
  5. **Loss Function**: The model is trained using binary cross-entropy loss, which penalizes incorrect predictions for binary classification tasks like this.
  6. **Preprocessing**: The MRI images and segmentation masks are first converted to a consistent format and size. The mask is binarized to ensure a clear distinction between cancerous and non-cancerous regions. This preprocessing optimizes data loading speed and ensures uniformity for effective model training.

  The **UNet architecture** enables high-resolution segmentation with minimal information loss, making it ideal for medical segmentation tasks where fine details are essential.

A visualization of how the model works is shown below:

![download](C:\Users\11vac\Desktop\download.jpg)



## Usage

Each script serves a specific function in the pipeline. Below is a summary of how each script is used, along with an overview of key code comments for clarity.

### File Overview and Usage

1. **file_trans.py**: Preprocesses raw `.nii.gz` MRI images and masks by resizing them and saving as `.pt` files for faster loading.

   ```python
   # Load the .nii.gz file and expand dimensions to add a channel
   image = nib.load(os.path.join(input_dir, filename)).get_fdata()
   image = np.expand_dims(image, axis=0)  # Add channel dimension
   ```

   **Usage**: Run this script before training to convert raw MRI data to `.pt` format. Specify input and output directories.

   ```bash
   python file_trans.py
   ```

2. **dataset.py**: Defines the `ProstateCancerDataset` class, which loads the preprocessed `.pt` files and applies any additional transformations.

   ```python
   # Convert mask to binary format where values > 0 are set to 1
   mask = (mask > 0).float()  # Binary mask for cancerous regions
   ```

   **Comments**:

   - The binary conversion in the `__getitem__` method ensures compatibility with binary cross-entropy loss.
   - The `__len__` method provides the dataset size, while `__getitem__` retrieves and prepares each image-mask pair.

3. **modules.py**: Implements the UNet architecture with detailed layers and skip connections.

   ```python
   def forward(self, x):
       enc1 = self.encoder1(x)  # Initial encoding layer
       ...
       dec1 = self.decoder1(dec1)
       return torch.sigmoid(self.output(dec1))  # Sigmoid for binary output
   ```

   **Comments**:

   - Skip connections are formed by concatenating features from encoder and decoder at each level, preserving spatial details.
   - Sigmoid activation at the output layer is used to generate probabilities for binary segmentation.

4. **train.py**: Defines and runs the training loop.

   ```python
   # Training loop over epochs
   for epoch in range(epochs):
       model.train()
       ...
       running_loss += loss.item()
   ```

   **Usage**: Run this script to train the model with the specified hyperparameters. It loads the dataset, initializes the model, and trains for the configured number of epochs.

   ```bash
   python train.py
   ```

   **Comments**:

   - The training loop calculates and accumulates loss, prints progress after every 100 steps, and evaluates loss per epoch.
   - DataLoader parameters like `num_workers` and `pin_memory` are set for faster loading.

5. **predict.py**: Loads the trained model and generates predictions on the test dataset, displaying input and predicted masks.

   ```python
   def predict():
       with torch.no_grad():  # No gradient tracking for inference
           for images, masks in test_loader:
               outputs = model(images.to(device))
   ```

   **Usage**: Run this script after training to visualize the segmentation results on test data.

   ```bash
   python predict.py
   ```

   **Comments**:

   - Uses `torch.no_grad()` to disable gradient computation during inference, saving memory and computation.
   - Loads the trained model weights and iterates through the test set to generate predictions.
   - Visualizes both the input image and the predicted mask side by side.

## Preprocessing

MRI scans and their corresponding segmentation masks are stored as `.nii.gz` files. Before feeding these images into the model, they are preprocessed:

- The images are resized to a consistent shape of `256x256` pixels.
- Masks are converted to binary format, where pixel values greater than 0 are set to 1.
- Preprocessed images and masks are saved as `.pt` files for faster loading during training.

The preprocessing code can be found in `file_trans.py`, and an example of usage is:

`input_dir = r'1HipMRI_study_keras_slices_data/keras_slices_train'`

`output_dir = r'1HipMRI_study_keras_slices_data/processed_train'`

`preprocess_and_save(input_dir, output_dir)`



## Model Training

The model is trained using the `BCELoss` (Binary Cross-Entropy Loss) for 50 epochs, with a batch size of 8 and a learning rate of 0.001. The Adam optimizer is used for gradient-based optimization.

Training setup:

- **Dataset**: The dataset is split into training and testing sets, where 85% of the data is used for training and 15% is used for validation.
- **Training**: The model is trained using the preprocessed dataset, and the loss is printed after every 100 iterations to track progress.
- **Validation**: After each epoch, the model is evaluated on the validation set.

Example Code for Training:

`python train.py`



## Predicting Segmentation Masks

Once the model is trained, predictions can be made on the test dataset using the `predict.py` script. This will load the pre-trained model and make predictions for each MRI slice in the test dataset, visualizing the original image and the predicted mask.

### Example Code for Prediction:

`python predict.py`



## Dependencies

The following dependencies are required to run this project:

- Python >= 3.8
- PyTorch >= 1.8.0
- nibabel (for handling MRI `.nii.gz` files)
- matplotlib (for visualization)
- torchvision (for image transformations)

- NumPy 1.22
- Scikit-learn 1.1.1

You can install all dependencies via `pip`:

`pip install torch torchvision nibabel numpy scikit-learn matplotlib`



## Reproducibility

The model is reproducible given the same dataset and preprocessing steps. The random seed can be set in PyTorch to ensure that the model training process is deterministic.

`torch.manual_seed(47315240)`



## Example Inputs, Outputs, and Results

Example MRI slice and its segmentation output:

- **Input:** 256x256 MRI image slice (grayscale).
- **Output:** 256x256 binary mask (1 - cancerous region, 0 - background).

### Training Loss:

After 50 epochs, the training loss converges to a low value, indicating that the model has successfully learned to perform segmentation.



## Preprocessing Steps and Justification

1. **Resizing**: All images and masks are resized to a uniform size of `256x256`. This ensures that the model can process the images efficiently and uniformly.
2. **Binary Masking**: Since the problem is binary segmentation, the masks are thresholded to convert non-zero values into 1 (indicating cancerous regions) and keep 0 for non-cancerous regions.
3. **Data Splitting**: The dataset is split into training (85%) and validation (15%) sets to ensure the model is properly evaluated during training.

## License

This project is open-source and licensed under the MIT License.





