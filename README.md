# 1 Training parameters
- Image input size: 3, 224, 224
- Training bacth size: 32
- Learning rate: 0.001
- Epochs = 25

Model is trained using 80% of the data and tested using 20% of the data. 

Total training images: 21520
Total validation images: 7200
Total test images: 1800

Following code is used to load the data and split the data into training, validation and test data.

```
train_dataset = datasets.ImageFolder(
    root=r"ADNI_AD_NC_2D\AD_NC\train", transform=transform
)
test_dataset = datasets.ImageFolder(
    root=r"ADNI_AD_NC_2D\AD_NC\test", transform=transform
)


val_size = int(0.8 * len(test_dataset))
test_size = len(test_dataset) - val_size
val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

```

# 2 Libraries used

Following are the main libraries used.
timm
```
conda install conda-forge::timm
```

torch
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

sklearn
```
conda install anaconda::scikit-learn
```

seaborn
```
conda install anaconda::seaborn
```

numpy
```
conda install anaconda::numpy
```

# 3 Model Evalution

Final accuracy: 93.88% of unseen test data is predicted correctly.

Image below shows the accuracy using confusion matrix.

![Alt text](images/confusion_matrix.png)

# 4 Training process
The train.py module trains the data. Data is first loaded and preprocessed. Then the model is trained using the training data. The model is then evaluated using the test data. The model wieghts is saved in the model directory. 
