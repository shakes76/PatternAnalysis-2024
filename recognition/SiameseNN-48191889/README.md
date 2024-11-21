# Siamese Neural Networks on the Classification of Melanoma
By Ainul Malik Zaidan Ismail - 48191889

This algorithm trains a Siamese Neural Network to classify between cases of melanoma. The algorithm takes two 256 pixel x 256 pixel sized RGB images of skin lesions as input and predicts if the two images are of the same class or not. This can be utilized in the prediction of benign or malignant cases of melanoma through the comparison between pre-labelled pictures. The solution to this classification problem can be beneficial in the field of medicine, aiding doctors in providing accurate diagnoses.


### Network Architecture

The Siamese Neural Network is constructed based on the architecture given by G. Koch, R. Zemel, R. Salakhutdinov et al. in their paper "Siamese neural networks for one-shot image recognition" depicted in the diagram below.

![Screenshot 2024-10-24 at 5 58 33 PM](https://github.com/user-attachments/assets/0ca9be1c-fd3f-4c5a-bc05-8b362549ca53)

The diagram describes one of the two twin channels of the Siamese Neural Network, through which the two images will be encoded into smaller feature maps. Since the twin channels will contain the same weighted features, the encoded result of each channel will be able to provide class information of the image, and be processed through a sigmoid function to determine if the two images belong in the same class or not.

## Training Dataset
The SNN is trained using the [Kaggle ISIC 2020 Challenge Dataset](https://challenge2020.isic-archive.com/), resized to 256 x 256 from the [dataset uploaded by Nischay Dhankhar](https://www.kaggle.com/datasets/nischaydnk/isic-2020-jpg-256x256-resized/data) on Kaggle.

The dataset contains 33,126 images of skin lesions alongside a CSV file with metadata about the dataset. The pictures are split into 32,542 cases of benign melanoma, and 584 cases of malignant melanoma. 


### Pre-Processing
Due to the imbalanced nature of the dataset, where the majority class contains 32,542 items while the minority class only contains 584, sampling methods will be necessary to be able to accurately train the SNN model on unseen data. 

The sampling method will be a mixture of both undersamping and oversampling: perform undersampling through random sampling on the majority class by a factor of 0.5, and perform oversampling through duplication on the minority class by a factor of 4. The reason why oversampling on the the minority class to the size of the majority class is because of the redundancy of too much duplicate data.

This will result in a dataset that is much more balanced.
```
Size of benign (majority/0) class:  32542
Size of malignant (minority/1) class:  584

Resized benign class:  16271
Resized malignant class:  2336
```

### Train-Test Split
The dataset will be trained on 80% of the dataset, while the other 20% is utilized for training. This split is preferred as it maintains the majority of data for training and leaves enough for testing based on the small minority class.

## Usage
### train.py
The `train.py` module trains a Siamese Neural Network defined in `modules.py` on the training dataset. Running the module stores the trained model as `siamese_model_final.pth` alongside the loss plot of the training process. The loss plot is as follows.

![snn_final](https://github.com/user-attachments/assets/3a934bb8-4b48-4112-a4c4-1df285c4a833)


### predict.py
The `predict.py` file contains a testing script based on the training/testing split of the dataset. It uses the model trained in `train.py` and evaluates its performance on the testing dataset. Running the code will display evaluation metrics of the model. 

```
Test Accuracy: 98.23
Precision: 98.86
Recall: 97.59
F1-score: 98.22

Confusion Matrix:
[1834   21]
[  45 1822]
```

Predicting new images can be done using the following code:
```
model = SiameseNetwork().to(device)

model_path = "siamese_model_final.pth"
model.load_state_dict(torch.load(model_path))

# assign img_path to 256 x 256 images
img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")
model.predict(img1, img2)
```
This will return either 0 or 1, where 0 indicates that they are not part of the same class and 1 indicates that they are.

## Dependencies
- PyTorch: For calculating the numerous linear equations in training the neural network. 

- Scikit Learn Metrics: To provide evaluation metrics on the trained dataset.

- Matplotlib: To plot the loss function of the training process.

## References
Generative AI (ChatGPT) has been used in the learning process for code inspiration and referencing throughout the project. 

G. Koch, R. Zemel, R. Salakhutdinov et al., “Siamese neural networks for one-shot image recognition,” in
ICML deep learning workshop, vol. 2. Lille, 2015, p. 0.

Training a Classifier — PyTorch Tutorials 1.5.0 documentation. (n.d.). Pytorch.org. https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

‌Neural Networks — PyTorch Tutorials 1.6.0 documentation. (n.d.). Pytorch.org. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

‌Datasets & DataLoaders — PyTorch Tutorials 1.11.0+cu102 documentation. (n.d.). Pytorch.org. https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

‌Tam, A. (2023, January 30). How to Evaluate the Performance of PyTorch Models - MachineLearningMastery.com. MachineLearningMastery.com. https://machinelearningmastery.com/how-to-evaluate-the-performance-of-pytorch-models/




