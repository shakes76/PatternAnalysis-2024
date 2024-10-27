# 1 Training parameters
- Image input size: 3, 224, 224
- Training bacth size: 32
- Learning rate: 0.001
- Epochs = 25

Model is trained using 1200 images, validated using 900 images and tested using 300 images.

# 2 Model Evalution

Final accuracy: 93.88% of unseen test data is predicted correctly.

Image below shows the accuracy using confusion matrix.

![Alt text](images/confusion_matrix.png)

# 3 Training process
The train.py module trains the data. Data is first loaded and preprocessed. Then the model is trained using the training data. The model is then evaluated using the test data. The model wieghts is saved in the model directory. 
