# Prostate MRI Segmentation with 2D UNet

## Overview
This project implements a 2D UNet model for prostate cancer segmentation from MRI images. The data is provided in Nifti format and is processed into 2D slices. The model achieves a minimum Dice coefficient of 0.99 on the test set for the prostate label. 

## Files
- `modules.py`: Contains the 2D UNet model definition. 
- `dataset.py`: Handles the data loading and preprocessing. Uniform the image size. files dowmloaded from https://filesender.aarnet.edu.au/?s=download&token=76f406fd-f55d-497a-a2ae-48767c8acea2
- `train.py`: Code for training, validating, and saving the model. Only 1000 pieces of data are used in this code, and the amount of data can be changed depending on the device. And each epoch has 250 samples. 
- `predict.py`: Code for making predictions and visualizing the results for every example. There are 540 samples in total. 
- `README.md`: Project documentation with enquirement and code result. 

## Running process
1. Prepare dataset in the correct directory structure.
2. Run `train.py` to train the model. The path of testing dataset should be change on the different device. The optimizer used is Adam with an initial learning rate of 0.001, and training is conducted for 5 epochs with a batch size of 4.
3. Run `predict.py` to visualize predictions. The path of testing dataset should be change on the different device. The exact image can be seen if run the code that marked as a comment visualization.

## Libraries and Dependencies
1. Python 3.7+
2. NumPy: Efficient numerical operations on arrays.
3. Torch: Used for model implementation, training, and optimization.
4. NiBabel: For loading and handling MRI data in Nifti format.
5. Matplotlib: For plotting training loss and metrics.
6. scikit-image: For resizing MRI slices.
7. tqdm: For progress bar visualization during dataset loading and training.

# Result example
1. Train.py:
Five epoches Loss and curve.
Epoch 1/5, Train Loss: 0.03396223486494273, Val Loss: 0.02705383070490577
Epoch 2/5, Train Loss: 0.008331425035372377, Val Loss: 0.008183859249887368
Epoch 3/5, Train Loss: 0.00457998073939234, Val Loss: 0.009112877431184505
Epoch 4/5, Train Loss: 0.003934595878701657, Val Loss: 0.0090887246004334
Epoch 5/5, Train Loss: 0.003441510858479887, Val Loss: 0.010765670976516876
with Loss curve visulisation.
2. Pridict.py
Last 10 Sample and Average Dice Coefficient
Sample 531: Dice Coefficient = 0.999387800693512
Sample 532: Dice Coefficient = 0.999424397945404
Sample 533: Dice Coefficient = 0.9993460774421692
Sample 534: Dice Coefficient = 0.9993444681167603
Sample 535: Dice Coefficient = 0.9990527629852295
Sample 536: Dice Coefficient = 0.9992634654045105
Sample 537: Dice Coefficient = 0.9982938170433044
Sample 538: Dice Coefficient = 0.9984582662582397
Sample 539: Dice Coefficient = 0.9987109899520874
Sample 540: Dice Coefficient = 0.9988020658493042
Average Dice Coefficient: 0.9989294597396144

From the results, the Dice coefficient of the model on the test set is very high, and the Dice coefficient of almost all the samples is above 0.99, which means that the segmentation effect of the model is good, and the prediction results are almost perfect match between the real label. However, the train process takes some times on CPU, so there will be some parameter adjustment on the number of training sample and learning rate depends on the device.  