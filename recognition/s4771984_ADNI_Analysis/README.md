
# Pattern Recognition and Analysis
## Classification of Alzheimer's Disease (Normal or AD) using Transformers GFNet6

## Problem Description
- Alzheimer's Disease is a neurodegenerative disorder that affects millions worldwide. For this, diagnosis and intervention should be as early as possible to ensure effective care. The project primarily related to the classification of AD vs. NC brain MRI images through the use of a **GFNet6 architecture**-based machine learning model. This model has adopted a Transformer-based architecture for vision tasks; the architecture allows for high accuracy in the classification of medical imaging.
- The dataset used in the development of this project is the **ADNI** dataset, which stands for Alzheimer's Disease Neuroimaging Initiative. This is a very standard dataset under medical research. Our goal is to achieve at least 80% or higher classification accuracy on the test dataset.

## Dataset Description
The ADNI (Alzheimer's disease Neuroimaging Initiative) dataset is widely used dataset in Medical research and Machine Learning, especially for the studies related to the Alzheimer's disease. It was created to support the discovery for the biomakers for the early detection and progression of Alzheimer's disease. The dataset includes various types of data collected from patients, ranging from imaging data to genetic information and clinical assessments. 

## Key Features of the ADNI Dataset : 
### 1) Imaging Data
- Magnetic Resonance Imaging (MRI) : In general, high-resolution structural imaging of the brain is conducted to analyze brain anatomy.
- Positron Emission Tomography(PET) : A type of functional imaging to evaluate activity of the brain; this usually targets amyloid and tau protein accumulation in the brain-perhaps the most well-known hallmarks of Alzheimer's disease. 

### 2) Clinical and Demographic Data : 
- Participants' age, sex, education, and medical history are given, which may be useful in trying to correlate the clinical outcomes with Alzheimer's disease.

### 3) Diagnostic Lagels : 
- For the dataset given to us had two categories which are mentioned below.
- Cognitive Normal (CN) : People who are healthy and in whom there is no evidence of cognitive impairment.
- Alzheimer's Disease (AD) : This group consists of patients with a diagnosis of Alzheimer's disease.

### 4) Longitudinal Data :
- The dataset also includes longitudinal follow-ups where patients data is collected over time. This will help in the essential for studying the disease progression and effectiveness of interventions.

## Algorithm Description
Traditional CNNs are very effective at extracting the spatial features of images via convolution and pooling operations. For this project, our approach follows the architecture of a GFNet-a traditional CNN combined with a Global Filter Layer that captures the local and global features of MRI images. The Global Filter Layer in GFNet adds the capability for capturing long-range dependencies across feature maps, which is particularly important for recognizing subtle patterns that can span larger areas of the brain. Fully connected layers after feature extraction carry out the classification into Cognitive Normal(CN) and Alzheimer's Disease(AD).

## Working of Algorithm & Architecture
Each convolutional layer in the GFNet model is followed by a max-pooling layer for downsampling and a dropout layer to regularize the feature maps, consequently avoiding overfitting. The gray-scale MRI image of size 128×128×1 pixels serves as input to the model. In this image, convolutions extract local spatial features such as texture and edges. It includes several convolutional and pooling operations, after which it applies the Global Filter Layer to further refine feature maps capturing global patterns of an entire image. These refined features go through a fully connected layer for classification. Non-linearity within this model is introduced with the use of the Exponential Linear Unit activation function, enhancing convergence during training. The model also applies dropout in order to reduce overfitting for randomly dropped units during training. The last output layer will predict which class the image belongs to-NC or AD.

## GFNet Architecture
The following image shows the architecture of the GFNet model used for Alzheimer's Disease classification.
![GFNet Architecture](https://drive.google.com/uc?export=view&id=1_MtDGSkJuh3BRswejXlqUUzrtikPBpcN)

# Requirements
- Python 3.8+
- Tensorflow/PyTorch
- Numpy
- Matplotlib
- Scikit-learn
- Google Colab Pro+ For Faster Training
