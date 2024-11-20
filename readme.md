# UNet2D Implementation On HipMRI

UNET is a convolutional neural network that specialises in image segmentation. As can be seen in Figure 1, it has two halves. The first encodes the image, reducing its spacial dimensions while encoding the features, and the second decodes, using the observed features to segment and classify the regions of the image as it upscales it. Figure 1 also shows how the decoder uses links to the encoder layers to perfectly regain the structure of the image while classifying its regions by upsampling the feature map. The main building block of UNet is the 3x3 convolution followed by a ReLU activation. These use a kernel followed by the nonlinear transformaiton to identify and encode increasingly high level features of the image while encoding, then do the reverse when decoding. 

Here Unet lets us segment the different regions of MRI scans of hips. It uses the Adam Optimiser and catagorical cross entropy loss. It takes in one hot encodings of the classes and outputs one hot softmax classification probability vectors. This project is good for helping me test my understanding of AI Principles and Practices while creating something that doctors may find useful.

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*zYrwp34DslR_9wLHMVAITg.png)
Figure 1: Unet architecture


## Dependencies
```
Package                 Version
----------------------- ---------
keras                   3.4.1  
nibabel                 5.3.1  
numpy                   2.1.2  
setuptools              72.1.0  
skimage                 0.0  
tensorflow              2.17.0   
tqdm                    4.66.4
```

Other requriements:
- Linux (if on windows find and replace all the '/'s in the path strings as '\\').
- 64GB Ram or higher (takes 40GB when training)
	- If you get an exit code 137 or 9:SIGKILL, you don't have enough RAM available. Try closing some programs (or downloading some more RAM).
- A good CPU (I Trained it on my PC to avoid waiting for Rangpur, so it runs on the CPU.)

## Example
#### Download
Fork the repository and/or download the repository with:
```sh
git clone https://github.com/Edmondson-codes/PatternAnalysis-2024.git
```

#### Install Dependencies
cd into the PatternAnalysis-2024 folder and run:
```sh
pip install -r /requirements.txt
```

Then download the HipMRI dataset and put it in a folder called `data` inside the `2DUnet_47454822` folder.
#### Train & Test
from the PatternAnalysis-2024 folder run:
```sh
python /2DUnet_47454822/train.py
```

If you haven't trained a model, It should output this:
```sh
100%|██████████| 11400/11400 [00:09<00:00, 1143.81it/s]
100%|██████████| 11400/11400 [00:04<00:00, 2480.91it/s]
100%|██████████| 540/540 [00:00<00:00, 1136.28it/s]
100%|██████████| 540/540 [00:00<00:00, 2473.20it/s]
Training Model
Epoch 1/2
357/357 - 2850s - 8s/step - accuracy: 0.8688 - loss: 0.4499
Epoch 2/2
357/357 - 2861s - 8s/step - accuracy: 0.9740 - loss: 0.0655
Finished training, saving
Saved and DONE
Testing Model
17/17 ━━━━━━━━━━━━━━━━━━━━ 29s 2s/step - accuracy: 0.9761 - loss: 0.9691
Dice Coefficient: 0.969436764717102
Accuracy: 0.9767849445343018
```

If you have already trained a model, it should output this:
```sh
100%|██████████| 11400/11400 [00:12<00:00, 916.95it/s]
100%|██████████| 11400/11400 [00:05<00:00, 2216.67it/s]
100%|██████████| 540/540 [00:00<00:00, 921.80it/s]
100%|██████████| 540/540 [00:00<00:00, 2225.11it/s]
Testing Model
17/17 ━━━━━━━━━━━━━━━━━━━━ 29s 2s/step - accuracy: 0.9749 - loss: 0.9662
Dice Coefficient: 0.9658582806587219
Accuracy: 0.9749886989593506
```

If a model exists in the models file with the name `model_name`, it will test it. If not, it will train then test it. 

The model it creates will be saved to a folder called models in 2DUnet_47454822.

#### View Results


## Data Processing
#### Pre-processing
The scans of Patient 19 in week 2 were of a different size to all the others so I removed them from both the training sets, test sets, and validation sets.  

I originally used random corps, flips and normalisation during training, but I found the model trained faster without them while not suffering a significant performance drop. Thus I went with the simpler option to increase development velocity despite the small accuracy trade off (as I was running the entire model on the CPU, it is hard to understate how important quick runs are. 1 epoc takes 42 mins. Still less than how long I've had to wait for rangpur though...).

#### Data Split
I preserved the data's original train/validate/test split as there is about 5% as much test data as train data (540 to 11400) and I needed all the training data I could get after removing crops and flips, so I kept the splits the same. I also didn't do anything fancy like K-fold cross validation as it was just unfesable given the computational requirements. This increased development velocity and allowed me to focus on improving the architecture of the model and debugging it (so much debugging), and considering that I barely managed to finish this assignment due to all the others I was juggling, this was definitely the right choice as it allowed a functional model to be submitted. 
