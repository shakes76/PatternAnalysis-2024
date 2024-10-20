# Medical Image Segmentation with UNet

This repository implements a **medical image segmentation** pipeline using the **UNet architecture** in PyTorch. The model is designed to solve segmentation problems on 2D medical images, specifically for MRI slices, using the Dice coefficient as the main evaluation metric. Medical image segmentation plays a critical role in diagnosing, planning, and treatment in medical imaging tasks by helping isolate regions of interest, such as organs or abnormalities, from medical images.

## Algorithm Description

**UNet** is a deep convolutional network architecture specifically designed for image segmentation tasks. It uses an encoder-decoder structure: the encoder reduces the spatial dimensions while increasing feature depth, and the decoder restores the spatial dimensions using up-sampling. This architecture is particularly effective for tasks like medical image segmentation where pixel-level accuracy is crucial. The task involves using MRI slices as input and producing binary masks that highlight areas of interest.

The segmentation model is trained to minimize the difference between predicted masks and ground truth labels using the **Binary Cross-Entropy (BCE)** loss and evaluated with the **Dice Similarity Coefficient**. The problem it solves is identifying and segmenting particular structures from MRI images, which is crucial in clinical applications such as planning surgeries or analyzing medical scans.

## How It Works

The training process includes feeding the MRI images into the UNet model, which generates a segmentation mask for each image. The model is optimized to minimize the **Binary Cross-Entropy Loss** function, and the Dice coefficient is calculated to assess the overlap between the predicted segmentation and the ground truth labels. We also visualize the segmentation outputs for qualitative assessment. Testing on unseen data allows the evaluation of generalization performance.

### Architecture

- **Encoder**: A series of convolutional layers followed by max-pooling, which reduces spatial dimensions while capturing features.
- **Decoder**: A series of upsampling layers with skip connections from the encoder, restoring the spatial dimensions and making pixel-wise predictions.
- **Skip Connections**: Help retain spatial information that would otherwise be lost during downsampling, ensuring fine-grained segmentation details.

## Dependencies

The code requires the following dependencies:
- **Python**: 3.8+
- **PyTorch**: 1.10.0+
- **NumPy**: 1.21.0+
- **Matplotlib**: 3.4.3+
- **Nibabel**: 3.2.1 (for handling NIFTI files)

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Reproducibility
To ensure the reproducibility of the results:
- We use a fixed random seed for dataset shuffling and weight initialization.
- Training and testing datasets are split using an 80-20 ratio.
- Consistent preprocessing methods are applied, including normalization of input images.
- Set the random seed for reproducibility in PyTorch:
```bash
torch.manual_seed(42)
```

## Preprocessing and Dataset Splits
The MRI images are preprocessed with the following steps:

1. Normalization: Each MRI slice is normalized to have a mean of 0 and a standard deviation of 1.
2. Augmentation (optional): Random flips and rotations are applied to improve model generalization.


Data is split as follows:
- Training: 80% of the data is used for training. This allows the model to learn the majority of the patterns in the data.
- Testing: 20% of the data is reserved for testing to evaluate the model’s generalization on unseen data. This split ensures that we can fairly evaluate the model's performance.

## Example Input and Output
# Input
- A 2D MRI slice (e.g., .nii.gz format).
# Output
- A binary mask for segmentation.

# Example Code for Loading and Testing the Model
```bash
# Load the trained model
model = UNet().to(device)
model.load_state_dict(torch.load('./model.pth'))
model.eval()

# Load test data
test_dataset = MedicalImageDataset(image_dir='path_to_test_data', normImage=True, load_type='2D')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Evaluate and visualize the predictions
test_model(model, test_loader)
visualize_predictions(model, test_loader, num_images=4)
```

# Example Dice Coefficient Output
```bash
Copy code
> Testing
Average Dice Coefficient: 0.8532
```
# Example Visualizations

		
## Training, Validation, and Testing Split Justification
The 80-20 split between training and testing is standard practice in machine learning, ensuring enough data for training while retaining a portion for unbiased testing. This ensures the model generalizes well to new, unseen data. Preprocessing such as normalization is used to standardize the input data, making the training process more stable.

We apply random transformations (e.g., rotations, flips) as part of the data augmentation strategy to make the model more robust to variations in the images. These augmentations help the model generalize better to variations in the real-world data, which is crucial for medical applications.

## Figures and Visualizations
An example of the model's predictions alongside the original MRI image and the ground truth mask is shown below.


## Citation and References
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint arXiv:1505.04597.
