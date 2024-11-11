# Melanoma Detection using YOLO11

## Overview

Melanoma is one of the most aggressive forms of skin cancer, and early detection can significantly increase survival rates. This project leverages the YOLO11 (You Only Look Once) deep learning algorithm by Ultralytics to automatically detect melanoma in dermoscopic images, distinguishing it from other skin conditions like benign lesions and nevus. YOLO11 is a state-of-the-art object detection model. 

<img width="126" alt="image" src="https://github.com/user-attachments/assets/9860b174-fe24-42f1-bf8c-41f44a9a1440">

*Figure: Sample output of YOLO11 detecting a lesion in a dermoscopic image*

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

- **Training Set**: 80% of the data
- **Validation Set**: 10% of the data
- **Testing Set**: 10% of the data

This split ensures the model has a sufficient amount of data for learning while keeping a balanced validation and testing set for evaluating performance.

### Pre-Processing

Pre-Processing
The preprocessing pipeline prepares the melanoma dataset for efficient and consistent model training. First, a metadata CSV file is generated for each dataset split (train, validation, and test). This metadata file serves as an index, listing each image path along with its corresponding class label (nevus, seborrheic keratosis, or melanoma). Labels are mapped to integers, with benign classes (nevus and seborrheic keratosis) labeled as 0 and malignant (melanoma) as 1. This structure allows for efficient data loading and simplifies referencing images during training. See below.

<img width="452" alt="image" src="https://github.com/user-attachments/assets/63603c7f-6a5d-419f-8472-81e105ee35ca">

Each image is then processed by:
Decoding from JPEG format and resizing to a standardized size of 299x299 pixels, ensuring consistency in model input dimensions.
Normalization, where pixel values are scaled to the [0,1] range for optimized training.
Caching the dataset to reduce I/O bottlenecks, and shuffling the training data with a buffer size of 1000 to ensure varied batches.
Batching and Prefetching: Images are batched into sets of 64, and prefetch is used to load data in the background, preventing delays and ensuring data availability during model training.

For more details on the dataset and augmentation methods, refer to the [ISIC Archive](https://www.isic-archive.com/).

## File Structure
The file structure should be organised as follows, with the labels folders being generated from the dataset code.  

<img width="220" alt="COMP3710_YOLO" src="https://github.com/user-attachments/assets/a0847c14-9b3c-4a48-b1b4-64926a327a16">

Label files look as follows, giving the location of the lesion bounding box.

<img width="506" alt="0 565326 0 458000 0 234375 0 269000" src="https://github.com/user-attachments/assets/c6dc9445-c395-4a45-b3fd-7b332bbaca26">


## Training the Model

To train the YOLO11 model, we use transfer learning from a pre-trained checkpoint, fine-tuning it on the melanoma dataset for 50 epochs. The training configuration is specified in the `melanoma.yaml` file, where the dataset paths and class names are defined.

In the training set, these images are associated with various labels. 


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
The dataset used for melanoma detection consists of dermoscopic images from the ISIC archive. The image dataset includes three main types of lesions: nevus, seborrheic keratosis, and melanoma. Each lesion type is stored in separate folders, and each image has an associated label to identify the type of lesion. The dataset follows the structure required for machine learning tasks, ensuring that each image file name is unique and follows a standardized naming convention (e.g., ISIC_0000000.jpg).

<img width="332" alt="Screen Shot 2024-11-01 at 15 57 33" src="https://github.com/user-attachments/assets/f7a7d701-5c88-4927-b76f-fd55857d0f65">

In the provided dataset folder structure, each lesion type is represented by high-resolution .jpg images. Additionally, there are auxiliary files with names ending in _superpixels.png or _perpixels.png, which appear to contain data that may be used for other types of analysis, such as texture segmentation or pixel intensity mapping. However, for the purpose of training a melanoma detection model, only the main dermoscopic images in .jpg format are used.


<img width="452" alt="image" src="https://github.com/user-attachments/assets/37b86b10-2843-4971-b537-d7c7ca75c936">



### Output
The model outputs bounding boxes and classification labels. 

<img width="457" alt="Screen Shot 2024-11-01 at 16 57 28" src="https://github.com/user-attachments/assets/900889e5-c126-4688-ad3d-8a7b276c1831">


<img width="448" alt="Screen Shot 2024-11-01 at 16 57 50" src="https://github.com/user-attachments/assets/007b5afe-ddbb-4ca5-9b13-6f681333b823">


## Results Visualization

After training, the model can detect lesions with high accuracy. 

<img width="452" alt="image" src="https://github.com/user-attachments/assets/c70abb36-df4a-4a52-a73f-26c3c1b41004">

*Figure: Training and validation loss over epochs. This was from an earlier test, eventually, 31 epochs were chosen*


TRAIN BATCH:
￼<img width="803" alt="Screen Shot 2024-11-12 at 08 56 53" src="https://github.com/user-attachments/assets/bd7f530b-0f77-4b25-ae61-24f47d04cdfc">


VAL BATCH
￼<img width="1199" alt="Screen Shot 2024-11-12 at 08 57 38" src="https://github.com/user-attachments/assets/d408e126-993f-41e9-9cc2-7d010c89784b">


<img width="1591" alt="Screen Shot 2024-11-12 at 08 56 27" src="https://github.com/user-attachments/assets/403a5e6a-7ad6-434a-88ec-9c63b0c49f05">

Normalised Confusion Matrix

<img width="881" alt="Confusion Matrix Normalized" src="https://github.com/user-attachments/assets/8a5ec5d0-cef7-4746-87c9-9beb8a4ffffa">

<img width="1173" alt="F1-Confidence Curve" src="https://github.com/user-attachments/assets/2fb3fc9f-443a-452e-b7de-f45d0357fb51">
<img width="1169" alt="Screen Shot 2024-11-12 at 08 56 05" src="https://github.com/user-attachments/assets/4888d724-aa53-4f65-8b12-3a1ab37cec51">
<img width="1194" alt="Precision-Recall Curve" src="https://github.com/user-attachments/assets/a9d9c0f6-acfc-433d-b8a5-b6306ca20dd3">


## Testing


## Conclusion

This project demonstrates the power of YOLO11 for real-time melanoma detection in dermoscopic images. With proper training and pre-processing, YOLO11 achieves high accuracy, making it a valuable tool for early skin cancer diagnosis.

## Future Improvements



## References

- ISIC Archive: [ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection](https://www.isic-archive.com/)
- Ultralytics YOLO Documentation: [YOLO Docs](https://docs.ultralytics.com/)



