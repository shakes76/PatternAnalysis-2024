We will attempt to use the latest vision transformers in order to classify Alzheimers from ADNI brain image data.


### Dataset Loading + Augmentation
- vision trasnfoemrs seem to want image sizes that are multiples of 16. So 224x224 might be the angle.
- Also need to normalise the data so we can help the model train better and converge faster.
- 
Resize to 224x224.
Random Horizontal Flip.
Random Rotation (small angles, e.g., ±10 degrees).
Random Brightness/Contrast Adjustments (slight, e.g., ±20%).
Normalization using the standard pre-trained stats for ImageNet  / empirically gathered statistics.


### Run files from this directory!