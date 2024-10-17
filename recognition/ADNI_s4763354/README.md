# Alzheimer’s disease Image Classification Task

This project aims to classify brain images from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset into two categories: normal and Alzheimer's disease (AD) using the GFNet vision transformer. The objective is to understand the use of Transformer model and its effectiveness on image classification. Ultimately, I hope to achieve an accuracy close to 0.8 on the test set. The GFNet architecture is selected for its efficiency and effectiveness in handling image classification tasks using Global Filters.

## Problem
Alzheimer's disease is a progressive neurodegenerative disorder that leads to cognitive decline and the dementia symptoms will get worse gradually year by year. Therefore, early and accurate diagnosis is very crucial to identify the cause and utilize effective treatment as early as possible. However, brain images could be complex to analyse and classify using simple models. This project addresses the challenge of classifying brain images to assist in the diagnosis of Alzheimer's disease using state-of-the-art Vision Transformer model - GFNet. This also provides a valuable tool for clinicians for automation and efficiency in medical dialogsis.

## Principle of GFNet

GFNet (Global Filter Network) is a cutting-edge vision transformer model that leverage global filter layers to replace self-attention layers in traditional transformer model [1]. It uses Fourier transforms to handle spatial features effectively by converting it to freqeuncy domains. It therefore learns long-term spatial dependencies in images efficiently. Unlike traditional CNNs, which typically depend on local receptive fields, GFNet applies frequency-domain filtering, performing element-wise multiplication between frequency-domain features and the global filter to capture global relationships across an entire image.  The global filtering mechanism allows the model to focus on both fine details and larger structural patterns in brain scans, making it an ideal choice for medical imaging tasks such as Alzheimer's disease classification. Its architecture is shown in the below figure. The uses of global features and fourier transform in GFNet reduces the number of parameters and result in faster training times and improved generalization which helps in differentiating the subtle differences between normal and AD brain images.

 <img src="images/gfnet.png" alt="description" width="300" height="200">



## Data Pre-processing 

### Dataset
The [ADNI brain dataset](https://adni.loni.usc.edu/) is used. It consists of 2D MRI slices labeled as either normal or Alzheimer's disease. Each 2D image is 256 x 240 pixels and represents one of 20 slices from a MRI scan collection. The dataset available in Rangpur is already divided into train and test sets, with 21520 and 9000 images respectively.  

### Preprocessing
For the pre-processing stage, it is carried out in 'dataset.py'. 
1. **Transformations** on training set to ensure model robustness:
   - **Resize**: All images are resized to 224x224 pixels to match the input size required by the GFNet model.
   - **Data Augmentation**: Random horizontal flipping, random rotations, and color jittering by adjusting brightness are used.
   - **Normalization**: Pixel values are normalized to standardize the input. 

   For the validation and test datasets, only resizing and normalization are applied, as no augmentation and shuffling is necessary for evaluation.

2. **Train-validation split**: Patient-wise splitting approach is used. Data is split based on unique patient IDs so that all images from the same patient do not appear in both sets. This prevents overfitting. The train-validation ratio is 8:2. 

3. **Balanced Sampling**: Below shows the class distribution of the original training dataset which is slightly imbalanced, with more NC samples than AD samples. To balanace the training set, class weights are first calculated based on the frequency of each class in the training set and a weighted random sampler is used to reduce the overrepresented class (NC) and to slightly increase the representation of the underrepresented class (AD) relative to the total. 
```
Class distribution before balancing:
------------------------------------
  Class 'AD' (Index 0): 10400 samples
  Class 'NC' (Index 1): 11120 samples
Total samples: 21520

Class distribution after balancing:
------------------------------------
  Class 'AD' (Index 0): 8563 samples
  Class 'NC' (Index 1): 8637 samples
Total samples: 17200
```


## Training and Validation
I initially trained the GFNet model using the [pre-existing GFNet classes](https://github.com/raoyongming/GFNet) on the preprocessed ADNI images, but I quickly realized that the performance was suboptimal due to the relatively small data size to a Transformer model. Transformer model usually requires a substantial amount of data for robust performance. Initially, my model overfitted with training accuracy around 100% but only 70% for validation accuracy. I adjusted hyperparameters to deal with overfitting like adding dropout, weight decay and changing the learning rates. Figure 1 and 2 show the training loss and accuracy after addressing overfitting. The final test accuracy was 67%.

While I have already trained my model using pre-existing GFNet model, I would like to compare it with pre-trained models as an additional experiemnt. I applied transfer learning by loading a pretrained GFNet and fine-tuned it on my dataset. I trained the GFNet_H_Ti and GFNet_H_B model. This approach yielded an improvement in test accuracy, reaching almost 77%. However, it also resulted in overfitting. I set dropout rate set to 0.7, used CosineAnnealingLR scheduler with 5e-6 learning rate, set weight_decay to 1e-2, applied early stopping and more. The gap between training and validation accuracy is smaller and the model generalizes better. 

  Train GFNet with pre-defined classes:
<div style="display: flex; justify-content: center; gap: 10px;">
  <figure>
    <img src="images/train_acc.png" alt="Training Accuracy Curve" width="400" height="200">
    <figcaption>Figure 1: Training Accuracy Curve</figcaption>
  </figure>

  <figure>
    <img src="images/train_loss.png" alt="Training Loss Curve" width="400" height="200">
    <figcaption>Figure 2: Training Loss Curve</figcaption>
  </figure>
</div>


  Fine-tune pretrain gfnet_h_ti model: 

<div style="display: flex; justify-content: center; gap: 10px;">
  <figure>
    <img src="images/image.png" alt="Pretrain GFNet_H_Ti Model - Accuracy" width="400" height="200">
    <figcaption>Figure 3: GFNet_H_Ti Model using Adam - Accuracy</figcaption>
  </figure>

  <figure>
    <img src="images/image-1.png" alt="Pretrain GFNet_H_Ti Model - Loss" width="400" height="200">
    <figcaption>Figure 4: GFNet_H_Ti Model using Adam - Loss</figcaption>
  </figure>
</div>

 Fine-tune pretrain gfnet_h_b model: 
<div style="display: flex; justify-content: center; gap: 10px;">
  <figure>
    <img src="images/image-8.png" alt="Pretrain GFNet_H_B Model - Accuracy" width="400" height="200">
    <figcaption>Figure 5: GFNet_H_B Model with AdamW, CosineLR Scheduler - Accuracy</figcaption>
  </figure>

  <figure>
    <img src="images/image-7.png" alt="Pretrain GFNet_H_B Model - Loss" width="400" height="200">
    <figcaption>Figure 6: GFNet_H_B Model with AdamW, Cosine LR Scheduler - Loss</figcaption>
  </figure>
</div>


## Result and Analysis

### Prediction

   Example input and prediction using gfnet_h_b model: 

   <img src="images/example_classifications.png" alt="gfnet_h_b model with AdamW">

   Confusion Matrix: 

   <img src="images/confusion_matrix.png" alt="gfnet_h_b model with AdamW" width="500" height="400">

Test accuracy: 

## Dependencies
The following packages are needed to be installed with specified versions for reproducibility:
  ```
  # Python >= 3.8
  torch>=2.4.1 
  torchvision>=0.19.1
  numpy>=1.26.3
  scikit-learn>=1.5.2
  matplotlib>=3.9.2
  tqdm>=4.66.5
  pandas>=2.2.3
  pillow>=10.2.0
  seaborn>=0.13.2
  ```

## How it works

If you would like to work with transfer learning and fine-tune pretrained model, download the pre-trained [GFNet_H_TI](https://drive.google.com/file/d/1Nrq5sfHD9RklCMl6WkcVrAWI5vSVzwSm/view?usp=sharing) and [GFNet_H_B](https://drive.google.com/file/d/1F900_-yPH7GFYfTt60xn4tu5a926DYL0/view?usp=sharing) and place them in the `ADNI_s4763354/` directory.

Steps of running each files: python ....py

 **Reproducibility**:
   - Include all scripts and configuration files necessary to reproduce the results.
   - Set r andom seeds for model initialization and data splitting.


Create a Slurm File for running the files: 
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100
#SBATCH --job-name=gfnet
#SBATCH -o output.out
#SBATCH --time=0-10:00:00
#SBATCH --partition=comp3710 
#SBATCH --account=comp3710 
conda activate torch
python train.py  #Replace with predict.py for testing
```





## References
- [1] Rao, Y., Zhao, W., Zhu, Z., Lu, J., & Zhou, J. (2021). Global filter networks for image classification. Advances in neural information processing systems, 34, 980-993.





Requirements
1. The readme file should contain a title, a description of the algorithm and the problem that it solves (approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results, if applicable.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation and testing splits of the data.
Marking Criteria
1. description and explanation of the working principles of the algorithm implemented and the problem it solves (5 Marks)
2. description of usage and comments throughout scripts (3 Marks)
3. proper formatting using GitHub markdown (2 Mark)
