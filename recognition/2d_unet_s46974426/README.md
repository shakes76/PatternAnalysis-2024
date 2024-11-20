PLEASE NOTE: the version is different from pdf submission because could not re-submit?

COMP3710 2D UNet Report

Using 2D UNet to segment the HipMRI Study on Prostate Cancer dataset

The task for this report was to create 4 files, modules.py, train.py, dataset.py and predict.py to 
load and segment the HipMRI study as a 2d Unet using Pytorch and the direct task description taken 
from Blackboard is below.

Task Description from Blackboard: "Segment the HipMRI Study on Prostate Cancer (see Appendix for link) 
using the processed 2D slices (2D images) available here with the 2D UNet [1] with all labels having a 
minimum Dice similarity coefficient of 0.75 on the test set on the prostate label. You will need to load 
Nifti file format and sample code is provided in Appendix B. [Easy Difficulty]"

I quickly want to mention that I prefixed each commit with 'topic recognition' this was a force of habit, 
typically when I have worked on git repositories I first branch the solution named after a change request 
e.g. "CR-123" and prefix each commit with the name of the CR.

An initial test code was run to just visualise one of the slices before using 2D UNet to get a sense of 
what the images look like. The resulting image after test.py was run can be seen in 
slice_print_from_initial_test in the images folder.

The data loader was run in a simple for to check that it worked, it was ~50% successful when it errored due 
to image sizing issue. To resolve this, an image resizing function was added to be called by the data loader. 
The completed data_loader test output can be seen in data_loader_test.png in the images folder.

After messing around with fixing errors from the original versions I had tried of the modules, dataset, predict 
and train scripts I eventually gave up as they would not run.

I went online and found a similar example of a 2d UNet implemented using pytorch and adapted the code to suit 
my problem and reference to this repository can be seen below.

Author: milesial
Date: 11/02/2024
Title of program/source code: U-Net: Semantic segmentation with PyTorch
Code version: 475
Type (e.g. computer program, source code): computer program
Web address or publisher (e.g. program publisher, URL): https://github.com/milesial/Pytorch-UNet

Also, during this process, I discovered that the masks were in the segment datasets and the images were 
in the datasets not suffixed with 'seg' (I had it the wrong way around originally).

After attempting to run the train.py file and fixing errors as they occurred, I was eventually able to 
run the train.py code in full to generate some loss and dice coefficient-based validation plots.

I ran the train code for the first 5 epochs and a graph showing the batch loss and a graph showing the 
dice score can both be seen in the images folder. I then ran it for 50 epochs and the graphs similar to 
above are in the images folder. The console running progress can also be seen in the console_running image 
in the images folder.

This final part will outline a description of working principles of the algorithm and the problem it solves. 
The Pytorch UNet is comprised of four parts, an encoder, decoder, bottleneck and a convolutional layer. 
The modules script contains the UNet’s definition. It also includes the dice coefficient handling to calculate 
dice loss which measures the overlap of two images in order to quantify a segmentation model’s accuracy. 
I also added a function to combine two datasets (the segment images and masks), this is because datasets 
what include both segments and masks are typically used in UNet algorithms. The modules script also included 
some basic dataset classes, a method to load images, check uniqueness of masks and some basic plotting logic.

The train script initialises and loads the UNet model defined in the modules and then trains it on the 
segmentation dataset. Before this is done however, it is transformed and loaded as 2d data using the provided 
load_data_2d function in the task appendix. The train script handles defining the main train loop, iterating 
over the data in batches, calculating losses and dice scores, which are then plotted after the algorithm has 
completed. It also handles saving progress while the training loop completes each epoch, which is made up of 
a number of batches (typically 5-6 in this case).

The dataset script just contains the load_data_2d method as seen in the appendix of the task sheet. It also 
contains a data transformation function to make the image dimensions consistent.

Finally, the predict script’s purpose is to generate mask predictions of new images on a trained and saved UNet model.


