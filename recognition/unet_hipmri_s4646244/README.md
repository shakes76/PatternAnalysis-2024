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

#### Folders
Unet_images
- A folder containing the testing results of running different epochs

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
In the results twelve epochs were used to reduce the training time however only 6 were run as the model stopped early.
It also uses a validation set to evaluate performance after each epoch, helping to monitor overfitting by checking how well the model performs on new data.

### Testing 
After the model is trained it then is given new data to segment. This data is  

### Performance
The model was run using a batch size of four and twelve epochs, only six were run as it stopped early. 
The results for the model being trained on three and six epochs can be seen in the Unet_images folder.

The model is very good at binary segmentaion of the prostate cancer images, If i had more time I would convert it to do multi class segmentation.
It segments most of the regions well except very small areas.
It can be seen that the mean dice test score is just above 0.65 however there are many datapoints that fall below this region.
When looking at the dice coefficents over each epoch it sharply increases and then slowly tapers off, this is the same for the loss function except it sharply decreases. 

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


