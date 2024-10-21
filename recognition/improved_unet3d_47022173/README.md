# 3D Improved UNet3D - Prostate 3D data set

Segmenting prostate 3d data set using the 3D Improved UNet3D.
All labels must have an f1-score of above 0.7.

## Author

Abdullah Badat (47022173)

## Project Overview

## Dataset and processing

## The 3D Improved UNet3D

The 3d improved UNet3D uses both skip connections across the 'U' and residual connections around context modules.

### Model Architecture

![unet_architecture](assets/unet_architecture.png)

### Hyperparameters

## Dependencies

## Repository layout

- dataset.py -
- driver.py -
- modules.py -
- predict.py -
- train.py -
- utils.py -

## Dataset

## Usage

```
usage: driver.py [-h] -m MODE -s SYSTEM [-sp SAVE_PATH] [-p MODEL_PATH] [-lr LEARNING_RATE] [-bs BATCH_SIZE] [-e EPOCHS] [-wd WEIGHT_DECAY] [-ss STEP_SIZE] [-g GAMMA]

options:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Train, debug or predict.
  -s SYSTEM, --system SYSTEM
                        Local or rangpur.
  -sp SAVE_PATH, --save-path SAVE_PATH
                        Path to save preds.
  -p MODEL_PATH, --model-path MODEL_PATH
                        Path to the model file for predict.
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate for optimizer.
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for loader.
  -e EPOCHS, --epochs EPOCHS
                        Epochs to run for training.
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay for optimizer.
  -ss STEP_SIZE, --step-size STEP_SIZE
                        Step size for scheduler.
  -g GAMMA, --gamma GAMMA
                        Gamma for scheduler.
```

### Example Usage

`python3 driver.py -m train -s rangpur -e 150 -bs 4 -sp saves_150_4`

Trains the model on rangpur. Runs for 150 epochs and uses a batch size of 4.
Saves the prediction checkpoints in folder saves_150_4.

`python3 driver.py -m predict -s local -sp saves_predict -p model.pth`
Runs inference on the test with a model loaded from model.pth. Saves prediction images to a folder called saves_predict.

## Results

### Comparing predictions to masks

### Loss curve and dice curve

## Discussion

## Conclusion

## References

[1] https://github.com/pykao/Modified-3D-UNet-Pytorch?utm_source=catalyzex.com

[2] https://arxiv.org/abs/1802.10508v1
