# 2D Unet to segment MRI images of prostate cancer 
This repository contains a TensorFlow Keras implimentation of a Binary classification Unet to segment prostate cancer MRI scans. 

#### Files: 
dataset.py    
- A file used to read in the .nii files into training, validation and test sets
modules.py    
- A file containing the Unet architecture, including the Encoder, decoder and bottleneck 

train.py
- A file containing the functions to train the unet and save the model

predict.py
- A file to show how well the Unet predicts test images


### Model 
###### Filters
The filters applied during the encoder step are [64,128,256,512] and [512,256,128,64] for the decoder.
###### Encoder
The encoder takes the input tensor and for each filter in the list completes two convolutions with a 3x3 kernel size and a relu activation function.
It then saves the resulting tensor into the skip connection so it can be used in the decoder.
A 2x2 max pooling is then used and the resulting tensor output.
This happens four times, once for each filter.  

###### Bottleneck
The bottleneck step applies two convolutions using a 3x3 kernel and the relu activation function.
it then outputs the tensor to be used in the decoder

###### Decoder
The decoder then takes this tensor, completes an up convolution using a 2x2 kernel and a relu function
The relevent skip connection is then concatenated with the tensor from the upconv.
This concated tensor is then fed through two convolutions using a 3x3 kernel and relu activiation
One last convolution using the sigmoid function is used to complete the binary classification.

### Training
The Unet is trained with an adam optimiser with an initial learning rate of 0.0001. It uses a combined loss function that consists of binary cross entropy and dice loss as this resulted in the best performance. 
The training also uses early stoppage, this stops the training when the validation loss performance does not increase after three consecutive epochs.
A learning rate scheduler is also used to monitor the validation loss, if the validation does not improve after two epochs the learning rate is halved.
In the results only six epochs were used to reduce the training time, If it were to be done again 20 epochs would be used to make sure that the model converges but doesnt overtrain.
It also uses a validation set to evaluate performance after each epoch, helping to monitor overfitting by checking how well the model performs on new data.

### Testing 
After the model is trained it then is given new data to segment. This data is  


### Performance



#### Required dependencies 
- TensorFlow (for Keras layers, models, and callbacks)
- NumPy (for numerical operations)
- Matplotlib (for plotting)
- NiBabel (for neuroimaging data handling)
- tqdm (for progress bars)
- scikit-image (for image transformations)
- pathlib (for filesystem path manipulations)

#### Future improvements
The dataset given is for multiclass segmentation. The implimentation of my unet and training is only for binary classification, I tried implimenting the multiclass model by using a softmax activation function rather than sigmoid and modifying my train.py to handle the multiple classes however I could not get it working. In the future I will look into modifying the implimentation so that it can do multiclass segmentation.

#### How to run
Before running the model the relevent files paths need to be added into dataset.py 
Once this is done all that is needed to be run is the predict.py file with no arguments.
This will train, validate, test and print the results of the model.

If a powerful graphics card is in your system, you may be able to increase the batch size in train.py this will result in faster training.

### References 
Reference for Dice coefficient metric implementation in the train.py function
Stack Overflow. "Dice coefficient not increasing for U-Net image segmentation." 
https://stackoverflow.com/questions/67018431/dice-coefficent-not-increasing-for-u-net-image-segmentation


