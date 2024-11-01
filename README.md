# Lesion Detection on ISIC Dataset with YOLOv7

## Description:

> This project aims to detect lesions in dermoscopic images from the ISIC 2017/2018 dataset using the YOLOv7 object detection model. The primary goal is to implement a solution that achieves a minimum Intersection Over Union (IoU) of 0.8 on the test set, ensuring reliable detection and localization of lesions within each image. Additionally, the model is expected to achieve a suitable accuracy for lesion classification, enhancing the utility of this approach for real-world applications in skin cancer detection.

## Problem Statement:

> The primary objective of this project is to detect lesions in dermoscopic images from the ISIC 2017/2018 dataset using the YOLOv7 model, aiming for an Intersection Over Union (IoU) of at least 0.8. Accurate lesion detection is essential for early skin cancer diagnosis, especially for aggressive forms like melanoma, where early treatment significantly improves outcomes. Automated detection systems can streamline the diagnostic process, reduce errors, and support dermatologists in more efficient, accurate diagnoses, potentially enhancing patient survival rates.

## Algorithm Explanation:

> YOLOv7 is a state-of-the-art, single-stage object detection model that processes entire images in a single forward pass, enabling real-time detection with high accuracy. It achieves this by optimizing network architecture and training strategies, resulting in faster inference speeds and improved precision compared to previous models.

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

## Example Installation
> To install these dependencies, you can use the following command:
  ```bash
  pip install torch torchvision numpy opencv-python Pillow matplotlib

## References

- "Skin Cancer Detection Using Convolutional Neural Networks: A Systematic Review," *National Center for Biotechnology Information (NCBI)*, https://pmc.ncbi.nlm.nih.gov/articles/PMC9324455/

