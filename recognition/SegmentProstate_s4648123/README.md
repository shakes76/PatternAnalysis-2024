Segment the (downsampled) Prostate 3D data set with the 3D Improved UNet3D with all labels having a minimum Dice similarity coefficient of 0.7 on the test set.
1. **“modules.py"** containing the source code of the components of your model. Each component must be implementated as a class or a function
2. **“dataset.py"** containing the data loader for loading and preprocessing your data
3. **“train.py"** containing the source code for training, validating, testing and saving your model. The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make sure to plot the losses and metrics during training
4. **“predict.py"** showing example usage of your trained model. Print out any results and / or provide visualisations where applicable
