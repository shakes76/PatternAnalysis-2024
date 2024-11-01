# YOLO11 Melanoma Detection

This repository provides instructions and code for training and deploying a YOLO11 model for melanoma detection, using the ISIC dataset or any custom melanoma dataset.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Validation](#validation)
- [Inference](#inference)
- [Exporting the Model](#exporting-the-model)

## Introduction

Melanoma detection using deep learning techniques can aid early diagnosis and reduce mortality. YOLO11, the latest version of the YOLO model by Ultralytics, is a fast and accurate model suitable for melanoma detection in medical images.

This project fine-tunes YOLO11 on a melanoma dataset to classify and localize skin lesions as "melanoma" or "benign".

## Requirements

Install the necessary libraries:
```bash
pip install ultralytics
```

## Dataset Preparation

1. **Download the Dataset**: Download the ISIC dataset from the [ISIC Archive](https://www.isic-archive.com/).
2. **Organize the Data**: Arrange your dataset in the following structure:
   ```
   /datasets/melanoma
   ├── images
   │   ├── train
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── val
   │       ├── image1.jpg
   │       ├── image2.jpg
   │       └── ...
   └── labels
       ├── train
       │   ├── image1.txt
       │   ├── image2.txt
       │   └── ...
       └── val
           ├── image1.txt
           ├── image2.txt
           └── ...
   ```
3. **Label Format**: Each `.txt` label file should contain one line per bounding box, in YOLO format:
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   - `class_id`: `0` for melanoma, `1` for benign.
   - `<x_center>`, `<y_center>`, `<width>`, and `<height>` should be normalized by image width and height.

## Configuration

Create a YAML file named `melanoma.yaml` to specify the dataset for YOLO training:

```yaml
# melanoma.yaml
path: /content/datasets/melanoma  # Dataset root directory
train: images/train               # Train images folder
val: images/val                   # Validation images folder

names:
  0: melanoma
  1: benign
```

## Training

To train YOLO11 on the melanoma dataset, use the following script in Python:

```python
from ultralytics import YOLO

# Load the pre-trained YOLO11 model
model = YOLO('yolo11n.pt')  # Load a lightweight version; options include yolo11s.pt, etc.

# Train the model
model.train(data='melanoma.yaml', epochs=50, imgsz=640)  # Modify epochs and image size as needed
```

This will fine-tune the YOLO11 model on your melanoma dataset for 50 epochs.

## Validation

After training, evaluate the model’s performance using the validation set:

```python
# Validate the model
results = model.val()
```

The validation metrics, including mAP (mean Average Precision), precision, and recall, will be displayed to help gauge model performance.

## Inference

To run inference on new images, use the following code:

```python
# Run inference on an image
results = model('/path/to/sample/image.jpg')
results.show()  # Display results with bounding boxes and class labels
```

The model will output bounding boxes around detected lesions with classifications as "melanoma" or "benign."

## Exporting the Model

You can export the model for deployment in different formats like ONNX, TensorFlow, and TensorRT.

```python
# Export to ONNX format
model.export(format='onnx')
```

Supported formats include `torchscript`, `onnx`, `openvino`, `tflite`, and more. Refer to the [Ultralytics documentation](https://docs.ultralytics.com/modes/export) for further details.

## Notes

- Adjust `epochs`, `batch_size`, and `imgsz` based on dataset size and hardware capabilities.
- Fine-tuning larger models like `yolo11s.pt` may yield better results but will require more computational resources.

## Acknowledgments

- This project utilizes the YOLO11 model from [Ultralytics](https://github.com/ultralytics/ultralytics).
- ISIC dataset provided by the [ISIC Archive](https://www.isic-archive.com/).
