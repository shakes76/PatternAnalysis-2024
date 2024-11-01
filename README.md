# Melanoma Detection using YOLO11

## Overview

Melanoma is one of the most aggressive forms of skin cancer, and early detection significantly increases survival rates. This project leverages the YOLO11 (You Only Look Once) deep learning algorithm by Ultralytics to automatically detect melanoma in dermoscopic images. YOLO11 is a cutting-edge object detection model that can detect multiple objects within an image in real time. This project adapts YOLO11 for binary classification of skin lesions as either *melanoma* or *benign*, making it a powerful tool for aiding in early skin cancer diagnosis. The project detects lesions within the ISIC 2017/8 data set with all detections having a minimum Intersection Over Union of 0.8 on the test set and a suitable accuracy for classification.





*Figure: Sample output of YOLO11 detecting melanoma in a dermoscopic image*

## How it Works

YOLO11 is a single-stage object detection model that processes the entire image in a single forward pass, predicting bounding boxes and classification scores simultaneously. It divides the input image into a grid, with each grid cell responsible for detecting an object within its bounds. Using anchor boxes, the model generates bounding box coordinates and confidence scores, optimized for melanoma detection by training on a labeled dataset of dermoscopic images. The final model can localize and classify skin lesions as either melanoma or benign in real time.

## Dependencies

To run this project, the following dependencies are required:

- **Python**: 3.10
- **Ultralytics**: 8.3.2 (includes YOLO11)
- **PyTorch**: 2.4.1+cu121
- **OpenCV**: 4.5.3
- **Matplotlib**: 3.4.2

Ensure you install the dependencies via:
```bash
pip install ultralytics opencv-python-headless matplotlib
```

To reproduce the results, a GPU with CUDA support is recommended. The model was trained on an NVIDIA Tesla T4 GPU for optimal performance.

## Dataset Preparation and Pre-Processing

### Dataset

The model was trained on the ISIC (International Skin Imaging Collaboration) dataset, a comprehensive collection of dermoscopic images labeled for melanoma and benign conditions. The dataset was divided as follows:

- **Training Set**: 70% of the data
- **Validation Set**: 20% of the data
- **Testing Set**: 10% of the data

This split ensures the model has a sufficient amount of data for learning while keeping a balanced validation and testing set for evaluating performance.

### Pre-Processing

1. **Resizing**: Images were resized to 640x640 pixels to ensure consistency and efficient processing.
2. **Normalization**: Pixel values were normalized to [0, 1] for faster convergence during training.
3. **Bounding Box Conversion**: Annotations in the ISIC dataset were converted to YOLO format, with bounding boxes specified by the center (x, y), width, and height, normalized by image dimensions.
4. **Data Augmentation**: Techniques such as random rotation, scaling, and flipping were applied to the training data to improve the model’s robustness to variations.

For more details on the dataset and augmentation methods, refer to the [ISIC Archive](https://www.isic-archive.com/).

## Training the Model

To train the YOLO11 model, we use transfer learning from a pre-trained checkpoint, fine-tuning it on the melanoma dataset for 50 epochs. The training configuration is specified in the `melanoma.yaml` file, where the dataset paths and class names are defined.

In the training set, these images are associated with various labels. 
<img width="452" alt="image" src="https://github.com/user-attachments/assets/63603c7f-6a5d-419f-8472-81e105ee35ca">


### Example Training Command

```python
from ultralytics import YOLO

# Load a pre-trained YOLO11 model
model = YOLO('yolo11n.pt')

# Train the model
model.train(data='melanoma.yaml', epochs=50, imgsz=640)
```

The model’s performance is evaluated using mean Average Precision (mAP), precision, and recall metrics on the validation set. 

## Example Inputs and Outputs

### Input
Input images should be high-resolution dermoscopic images, such as those from the ISIC dataset, formatted as `.jpg` or `.png` files.


### Output
The model outputs bounding boxes and classification labels. Below is an example output for a sample input image.


### Sample Code for Inference


## Results Visualization

After training, the model can detect melanoma with high accuracy. Below is a visualization of the performance metrics on the validation set:

<p align="center">
  <img src="path/to/metrics_plot.jpg" width="70%" alt="Training metrics">
</p>

*Figure: Training and validation loss over epochs*

## Exporting the Model

To export the model for deployment, YOLO11 provides options for various formats. For instance, to export the model to ONNX:

```python
model.export(format='onnx')
```

## Conclusion

This project demonstrates the power of YOLO11 for real-time melanoma detection in dermoscopic images. With proper training and pre-processing, YOLO11 achieves high accuracy, making it a valuable tool for early skin cancer diagnosis.

## References

- ISIC Archive: [ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection](https://www.isic-archive.com/)
- Ultralytics YOLO Documentation: [YOLO Docs](https://docs.ultralytics.com/)

---

This README provides comprehensive guidance on setup, training, and usage of YOLO11 for melanoma detection. Adjust paths and parameters as necessary for optimal performance on your dataset.
