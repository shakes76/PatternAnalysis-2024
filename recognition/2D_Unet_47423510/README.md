# 2D UNet For Predicting Prostate Cancer from HipMRI Scans  
**NOTE** -- Unable to conduct tests as training data was made unavaliable (https://edstem.org/au/courses/18266/discussion/2340286) & email to shakes.
## Prostate Cancer HipMRI Application
The purpose of this CNN UNet model is to assist in the diagnosis of prostate cancer through analysis of Hip MRI scans. The model's purpose is to be trained on images, such that it can segment and identify points of interest within the image. By training the Unet model in such a way that it recognises propoerties of cancer and malignent cells, it can assist profesionals in pinpointing the precise location of these issues.
## Algorithm Description

### UNet Model
The UNet model is a convoluted neural network that works by taking image segmentation, encoding it via several convolution layers. Then decoding it via the transpose convolution, while taking raw imports from the opposite encoding stage. That is to say, that a 2D Unet Model, for each layer of convolution stores the resultant convoluted image, and then concatenates these convoluted images with the same depth and width images obtained during the decoding steps. In doing this trianing can be conducted faster as more of the initial image is retained during training steps. 

### Encoding steps
Each level of encoding consists of 2 convolution steps, used for capturing the context of the image, and then a pooling step which reduces the dimensions of the image in preperation of the next step. This dimension reduction results in a increase in channels porportional to the level of dimension reduction. This spacial reduction is done until a desired bottleneck dimension, with a large number of convolution channels, is reached. Each convolution step is done using the Tensorflow.keras library's inbuilt Convolution function (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D). The kernel size used was 3x3, with the convolution being padded such that the output was the same dimensions as the input. The activation function chosen was a rectifier function, such that only the positive components of the arguments are used.

### Decoding steps
Once the bottleneck is reached, 2 more convolution steps are done, but no further dimension reduction is conducted on the image, in preperation of the restoration of the image. During the decode steps of the model, The reduced image is transformed using Tensorflow.keras library's transpose convolution (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose), such that a dimension reduction step is undone. Resulting in a halving of channels, but a doubling of image size. The image is then concatenated with a snapshot of the image at the encoded layer of the same dimension. this combined image is then convoluted twice using the same parameters as encode, before undergoing the next decoding step. Once the image is of the original image's dimension it is convoluted once with a 1x1 kernel through 1 channel, using softmax activation so a segmented image is obtained.

The model is then trained, such that each channel's weighting and bias can be accurately estimated by the model. Resulting in a reduction of the loss value. The loss function chosen was the catagorical crossentropy (https://www.tensorflow.org/api_docs/python/tf/keras/losses/categorical_crossentropy). 

## ~~ Example usage~~ 

### Preprocessing
The program handles all several aspects of dataset loading and manipulation prior to the main training loop to ensure efficent training procedure.

## Dependicies 
**Python3**
**Tensorflow**
**Numpy**
**matplotlib.pyplot**
**sklearn.metrics**

## References
relevant references are linked throughout the report, alongside the lecture example for a UNet and succesful pull requests from previous years.
