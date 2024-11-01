# Lesion Detection on ISIC Dataset with YOLOv7

## Description:

This project aims to detect lesions in dermoscopic images from the ISIC 2017/2018 dataset using the YOLOv7 object detection model. The primary goal is to implement a solution that achieves a minimum Intersection Over Union (IoU) of 0.8 on the test set, ensuring reliable detection and localization of lesions within each image. Additionally, the model is expected to achieve a suitable accuracy for lesion classification, enhancing the utility of this approach for real-world applications in skin cancer detection.

## Problem Statement:

The primary objective of this project is to detect lesions in dermoscopic images from the ISIC 2017/2018 dataset using the YOLOv7 model, aiming for an Intersection Over Union (IoU) of at least 0.8. Accurate lesion detection is essential for early skin cancer diagnosis, especially for aggressive forms like melanoma, where early treatment significantly improves outcomes. Automated detection systems can streamline the diagnostic process, reduce errors, and support dermatologists in more efficient, accurate diagnoses, potentially enhancing patient survival rates.

## Algorithm Explanation:

YOLOv7 is a state-of-the-art, single-stage object detection model that processes entire images in a single forward pass, enabling real-time detection with high accuracy. It achieves this by optimizing network architecture and training strategies, resulting in faster inference speeds and improved precision compared to previous models.

## YOLOv7 Architecture:

![YOLOv7 Architecture](yolov7_architecture.png)

## Dependencies:

> The following libraries and versions are required to run the lesion detection project:

- `torch` (PyTorch): for deep learning model implementation and training
- `torchvision`: for transformations applied to the images
- `numpy`: for numerical operations and array manipulation
- `opencv-python` (cv2): for image processing tasks
- `Pillow`: for handling image file loading
- `matplotlib`: for plotting and visualizing results

## How It Works

### Data Preprocessing
For consistent input dimensions, each image is resized to 640x640 pixels. Data augmentation includes random horizontal and vertical flips, color jitter, and normalization based on ImageNet statistics to improve model generalization. A custom transformation pipeline is applied using PyTorch's `torchvision.transforms`.

### Model Implementation
The `LesionDetectionModel` class implements the YOLOv7 model for lesion detection. This class loads pre-trained YOLOv7 weights via PyTorch Hub, allowing for efficient and accurate lesion detection on dermoscopic images. The model is loaded onto the specified device (either CPU or GPU) and optimized to use the available hardware resources.

1. **Model Initialization**:
   The model is initialized with pre-trained YOLOv7 weights, loading it onto the designated device. If the model’s backbone layers are detected, they are frozen to retain learned features and accelerate training by focusing only on the last layers for lesion-specific learning.

2. **Forward Pass**:
   The `forward` method performs a direct pass through the model, processing each image batch and returning bounding box predictions for lesions. This is done with `torch.no_grad()` to prevent gradient computation, making the inference process faster.

   ```python
   # Example of a forward pass
   pred = model.forward(images)

### Training Process
The training pipeline is built using PyTorch and includes data loading, model optimization, and performance tracking over multiple epochs. Key configurations include 10 epochs, a batch size of 16, and a learning rate of 0.001.

1. **Model and Data Loading**:
   - The YOLOv7-based `LesionDetectionModel` is loaded with pre-trained weights and set to the available device (GPU or CPU).
   - Data loaders for the ISIC training and validation datasets are set up with batch sizes for efficient processing.

2. **Training and Validation**:
   - During each epoch, the model performs a forward pass on the training data, calculates loss using binary cross-entropy (BCEWithLogitsLoss), and optimizes using Adam. A learning rate scheduler adjusts the rate to improve convergence.
   - Validation loss is computed without gradients to assess performance on unseen data, helping prevent overfitting.

   ```python
   # Example training and validation process
   for epoch in range(NUM_EPOCHS):
       train_loss = train_one_epoch()
       val_loss = validate()

### Prediction Process
The prediction pipeline in `predict.py` loads a trained YOLOv7 model to detect lesions in new images. It includes preprocessing, model inference, and result visualization.

1. **Model Loading**:
   - The `LesionDetectionModel` class loads the model with the trained weights on the specified device (GPU or CPU). The model is set to evaluation mode to prevent gradient computations, optimizing it for inference.

2. **Image Preprocessing**:
   - Each test image is resized to 640x640, converted to RGB, and normalized. These transformations ensure consistency with the model’s expected input.

3. **Inference**:
   - The `predict_image` function performs a forward pass, generating bounding box predictions for lesions. Non-maximum suppression is applied to filter out overlapping boxes, retaining only the most confident predictions based on IoU and confidence thresholds.

   ```python
   detections = predict_image(image_path)

## Code Comments and Usage Documentation

### Usage
To run the training and prediction scripts, follow these instructions:

1. **Training**: Use `training.py` to train the lesion detection model. Ensure you have specified the correct paths for data directories and adjust hyperparameters as needed.
   ```bash
   python training.py --data_dir path/to/data --epochs 10 --batch_size 16

2. **Predictiom**: Use predict.py to run inference with the trained model. Make sure to specify the path to the saved model weights.
   ```python
   python predict.py --model_path path/to/model_checkpoint.pth

## References

- "Skin Cancer Detection Using Convolutional Neural Networks: A Systematic Review," *National Center for Biotechnology Information (NCBI)*, https://pmc.ncbi.nlm.nih.gov/articles/PMC9324455/

