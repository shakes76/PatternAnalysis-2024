# Classification of Alzheimer's disease brain data using GFNet

## Author

Donghyug Jeong

Student ID: 47430004

## Project Overview

This project was developed as an attempt at a solution for Problem Number 5: "Classify Alzheimer’s disease (normal and AD) of the ADNI brain data using one of the latest vision transformers such as the GFNet set having a minimum accuracy of 0.8 on the test set."

The project uses the pytorch with GFNet model and the ADNI brain data set, and attempts to find the best combination of hyperparameters to maximise the accuracy of classification on the test set. In the process, it uses gfnet-xs architecture found in the original github repo [1].

## GFNet - Global Filter Network

Global Filter Networks is a transformer-style architecture, that uses a 2D discrete Fourier transform, an element-wise multiplication between frequency-domain features and learnable global filters, and a 2D inverse Fourier transform to replace the self-attention layer found in vision transformers [1]. According to Rao et al. [1], it "learns long-term spatial dependencies in the frequency with log-linear complexity".

The following is a gif created by Rao et al. [1] that demonstrates how GFNet works:

![intro](images/original_intro.gif)

## Global Filter Layer

GFNet consists of stacking Global Filter Layers and Feedforward Networks [1]. The Global Filter Layer uses the efficient Fast Fourier Transform algorithm to mix the tokens [1].

## Dependencies:

Older versions of below dependencies may work, but the following was the version used in the code, in conjunction with Python 3.12.4.

- pytorch: 2.4.1
- timm: 1.0.9 (requires searching on conda-forge)
- matplotlib: 3.9.2 (for plotting and visualising the data - actual model does not require it, but all .py files that can run import it)
- cuda: 12.6
- numpy: 1.26.3
- scikit-learn: 1.5.1
- torchvision: 0.19.1+cu118

The yaml file, `ADNI_GFNet_47430004.yaml` was created just in case these dependencies were not enough - the yaml file can be found in this repo, and can be used using the conda command

```
conda env create -f ADNI_GFNet_47430004.yaml
```

to create a new conda environment.

## Structure of Data

The code requires that the directory structures to the images are as follows:

Note: "/home/groups/comp3710/ADNI/AD_NC/" represents the directory on the UQ HPC Rangpur. This directory may be changed in the get_dataloaders() function of [dataset.py](/recognition/ADNI-GFNet-47430004/dataset.py). After the AD_NC directory, the structure must be met (i.e. train/ and test/ must exist with AD/ and NC/ directories in each).

```
/home/groups/comp3710/ADNI/AD_NC/
                                 train/
                                        AD/
                                            image.jpeg
                                        NC/
                                            image.jpeg
                                 test/
                                        AD/
                                            image.jpeg
                                        NC/
                                            image.jpeg
```

An image in the data set may look like this (This is an image of the NC set, within the training set):

<p align="center">
    <img src="/recognition/ADNI-GFNet-47430004/images/Sample_train_data_808819_88.jpeg" alt="Example ADNI brain data">
</p>

## Files in the Repo

`images/` - Directory that contains all the figures and images necessary for visually showing something.

`ADNI_GFNet_47430004.yaml` - yaml file which can be used to create a conda environment that is able to run the train.py. Details on using it are in [Dependencies](#dependencies).

`dataset.py` - Python file that handles all the dataset/dataloader processes.

`LICENSE` - License file containing the MIT License of the original repo found in [Inspiration](#inspiration).

`predict.py` - Python file used to test the accuracy of a given model, which should be located at `test/model/GFNet.pth`

`README.md` - This markdown file which describes everything about the repo.

`test_dataset.py` - Python file used to visually inspect that the dataset.py file is working correctly, especially the normalising transform.

`train.py` - Python file used to train the GFNet model, and save figures relating to its accuracy over epochs, losses over epochs and the best-case accuracy.

`utils.py` - Python file containing miscellaneous functions that help the training and predict process work. This file was taken directly from the repo listed in [Inspiration](#inspiration), and the source is also listed in the header of the python file.

## Usage

This section assumes that all dependencies are met.

### Training

```
python train.py
```

Many optional arguments can be found in the [train.py](/recognition/ADNI-GFNet-47430004/train.py) file, under the `get_args_parser()` function, and also under the [Hyperparameters](#hyperparameters) section.

The default execution of `train.py` allows the model to be trained using hyperparameters that I found useful in the context of the problem - classifying ADNI data. These hyperparameters can be found in [Hyperparameters](#hyperparameters).

The training script relies on [data set structure](#structure-of-data) being satisfied - otherwise, it will fail trying to load the data.

The training script will create a checkpoint of the best performing iteration so far, and also create a checkpoint every 20 epochs. Since the default setting for Epochs is 300, the script will create 16+ checkpoints (1 best, 15 interval) in the directory `test/model/run/`.

The training script also prints the results to `stdout` after each epoch, looking like this:

```
Epoch: [251]  [  0/337]  eta: 0:01:44  lr: 0.000043  loss: 0.0509 (0.0509)  time: 0.3102  data: 0.2525  max mem: 3558
Epoch: [251]  [336/337]  eta: 0:00:00  lr: 0.000043  loss: 0.0328 (0.0270)  time: 0.1512  data: 0.1042  max mem: 3558
Epoch: [251] Total time: 0:00:52 (0.1545 s / it)
Averaged stats: lr: 0.000043  loss: 0.0328 (0.0270)
Test:  [  0/141]  eta: 0:00:35  loss: 2.1723 (2.1723)  acc1: 68.7500 (68.7500)  time: 0.2485  data: 0.2170  max mem: 3558
Test:  [100/141]  eta: 0:00:05  loss: 0.3720 (1.9997)  acc1: 90.6250 (74.1491)  time: 0.1423  data: 0.0995  max mem: 3558
Test:  [140/141]  eta: 0:00:00  loss: 0.4723 (1.5649)  acc1: 89.0625 (78.4000)  time: 0.1427  data: 0.0979  max mem: 3558
Test: Total time: 0:00:19 (0.1402 s / it)
* Acc@1 78.400 loss 1.565
Accuracy of the network on the 141 test images: 78.4%
Max accuracy: 78.40%
```

Furthermore, the script creates a file called `log.txt` which has 1 line per epoch that looks like this:

```
{"train_lr": 4.2823776072812744e-05, "train_loss": 0.026965036172487494, "test_loss": 1.5648742771529136, "test_acc1": 78.4, "epoch": 251}
```

The dataset.py file applies augmentations such as:

```
train_transform = tf.Compose([
        tf.Grayscale(num_output_channels=1),
        tf.Resize((image_size, image_size)),
        tf.RandomRotation(degrees=15),
        tf.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        tf.ToTensor(),
        tf.Normalize(mean=[0.1156],
                     std=[0.2198],),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
    ])
```

In an attempt to minimise the model overfitting to the training data and enable generalisation to the test data. The training and test data were split by the task itself, and they were kept that way - they were not mixed around to attempt to keep the test set as consistent as possible.

### Predicting

```
python predict.py
```

I did not add an argsparser to predict.py. As a result, it relies on the model being used to predict existing in the directory `"test/model/GFNet.pth"`, relative to the location the script is run.

When the predict script is run, it loads the model and performs the test using the test set. This test set is normalised, but not augmented or shuffled unlike the training set, for consistent performance.

## Best Case Observed

Using the hyperparameters hard-coded into the argsparser (i.e. running `python train.py` with no arguments), the best test set accuracy of the model is **78.4%**.

### Accuracy over epochs

<p align="center">
    <img src="/recognition/ADNI-GFNet-47430004/images/Accs.png" alt="Test Accuracy vs Epochs">
</p>

This graph represents progression of the accuracy of the model, measured after every epoch. For the best case, the highest accuracy of **78.4%** was achieved in Epoch 251.

### Losses over epochs

<p align="center">
    <img src="/recognition/ADNI-GFNet-47430004/images/Losses.png" alt="Training and Test Loss vs Epochs">
</p>

This graph represents the training loss and the test loss of the model measured after every epoch.

### Test performance

<p align="center">
    <img src="/recognition/ADNI-GFNet-47430004/images/test.png" alt="Test Accuracy per batch, in best case">
</p>

This graph represents the performance of the model in the best-performing test per batch, where the model achieved an accuracy of **78.4%**.

## Hyperparameters

The following are the hyperparameters present in the model, with their default values and the argument to change their value. The parameters that are in **bold** are the ones that had many values tested to improve performance. The parameters with a strikethrough were not considered, but are present as they were present in the original code. Some parameters which were related to unused aspects, such as Model_EMA and the mixup function, were simply not included:

- **Batch size: 64 (`--batch-size`)**
- **Epochs: 300 (`--epochs`)**
- Input size: 224 (`--input-size`)
- **Dropout rate: 0.05 (`--drop`)**
- **Drop path rate: 0.3 (`--drop-path`)**
- Optimizer: adamw (`--opt`)
- Optimizer Epsilon: 1e-8 (`--opt-eps`)
- ~~Optimizer Betas: None (`--opt-betas`)~~
- Clip gradient norm: 1 (`--clip-grad`)
- SGD momentum: 0.9 (`--momentum`)
- **Weight decay: 0.05 (`--weight-decay`)**
- LR Scheduler: cosine (`--sched`)
- **Learning Rate: 5e-4 (`--lr`)**
- ~~Learning Rate Noise: None (`--lr-noise`)~~
- ~~Learning rate noise: 0.67 (`--lr-noise-pct`)~~
- ~~Learning rate noise std-dev: 1.0 (`--lr-noise-std`)~~
- **Warmup Learning Rate: 1e-5 (`--warmup-lr`)**
- Lower bound for Learning Rate: 1e-5 (`--min-lr`)
- **Epoch interval to decay Learning Rate: 30 (`--decay-epochs`)**
- **Warmup Epochs: 5 (`--warmup-epochs`)**
- Cooldown Epochs: 10 (`--cooldown-epochs`)
- Patience epochs: 10 (`--patience-epochs`)
- Learning Rate Decay Rate: 0.1 (`--decay-rate`)

## Reproducability

Due to the nature of machine learning and augmentation having varying effects on the training of the model, the model was able to consistently reach a similar accuracy/output of around mid~high 70%, but the consistency of training in terms of running the same hyperparameter and getting the exact same result was unmeasured due to time constraints. However, the prediction is reasonably consistent - given the same model, results returned have been always the exact same as expected in the tests I have conducted.

## Conclusion

The GFNet model was able to achieve a maximum accuracy of 78.4% on the test set, using the hyperparameters listed in [Hyperparameters](#hyperparameters).

For all training/validating sessions completed, overfitting was observed, where the training loss approached zero but the validation loss continuously increased. [Hyperparameters](#hyperparameters) were continuously modified to try and mitigate overfitting, but it was found that sometimes, more severe overfitting actually returned better accuracy on the test set.

The changes that were made to the hyperparameters are documented in the commit logs.

## Future Steps

To improve the model from here, I could:

- Adjust hyperparameters further to increase accuracy on validation set
- Change transformations applied to the training set to reduce overfitting
  - This was considered and implemented during the training process, however was discarded as test loss did decrease, but test accuracy decreased with it.
- Use more complex/verbose versions of the model such as GFNet-B or GFNet-H-B
  - This step could potentially make overfitting to the training data worse, experimenting should done before committing to the change.
- Experiment with the original repo's _DistillationLoss_, and potentially experiment with the scheduler and optimiser.
- Experiment with ModelEma

## Why the PR has 2 LICENSE files

Since the original repo (found in [Inspiration](#inspiration)) used the MIT License, a copy of the MIT License has also been included in this sub-folder, while also containing the Apache license of Shakes' repo.

## Inspiration

Significant portions of the code were taken from the following github repo:
https://github.com/shakes76/GFNet

This github repo is a fork of the official github repo of the original GFNet code by the authors of “GFNet: Global Filter Networks for Visual Recognition” [1].

## Official References/Bibliography

[1] Y. Rao, W. Zhao, Z. Zhu, J. Zhou, and J. Lu, “GFNet: Global Filter Networks for Visual Recognition,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 10960–10973, Sep. 2023, doi: 10.1109/TPAMI.2023.3263824.
