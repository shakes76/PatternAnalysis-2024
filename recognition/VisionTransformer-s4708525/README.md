# Classify Alzheimer’s disease (normal and AD) of the ADNI brain data using Convolutional Vision Transformer (CvT)

This project aims to classify Alzheimer's disease (normal vs. AD) using the Convolutional Vision Transformer (CvT) on ADNI brain data.

## Convolutional Vision Transformer
The Convolutional Vision Transformer (CvT) adds two main features to improve the Vision Transformer: Convolutional Token Embedding and Convolutional Projection.

- Convolutional Token Embedding: This layer uses convolutions to create overlapping patches from images, which are then turned into tokens. It reduces the number of tokens while increasing their detail, similar to how CNNs work, without needing extra position embeddings.

- Convolutional Transformer Blocks: In these blocks, the Convolutional Projection uses depth-wise separable convolutions to handle the query, key, and value in the attention mechanism. This is more efficient than the standard linear approach used in earlier transformers.

The model only adds a classification token in the final stage, and a fully connected layer makes the final class prediction. CvT is designed to be more efficient while still capturing rich features for image classification [Wu et al., 2021].

Below is a visualization of the Convolutional Transformer Block, which includes the convolution projection as the first layer.

![Description of image](results/convTransformer.png)

## ADNI brain data
Alzheimer’s disease (AD) is a progressive neurodegenerative disorder that leads to cognitive decline due to brain tissue deterioration, affecting millions globally. The ADNI study monitors the progression of AD by tracking biomarkers (such as chemicals in the blood and brain changes in MRI/PET scans) along with clinical measures. These assessments are conducted across three disease stages: cognitively normal, mild cognitive impairment, and dementia (Alzheimer's Disease Neuroimaging Initiative, n.d.). 

The ADNI dataset is an extensive and widely utilized resource that includes longitudinal data across multiple domains such as clinical, imaging, genetic, and biomarker information. It contains diverse data types, including structural, functional, and molecular brain imaging, biofluid biomarkers, cognitive evaluations, genetic data, and demographic details (Alzheimer's Disease Neuroimaging Initiative, n.d.).

This dataset contains two classes which are Normal Control (NC) and Alzheimer's disease (AD). In addition, this dataset contains:

Training data:
- NC: 11120
- AD: 10400

Testing data:
- NC: 4540
- AD: 4460

## References
- Alzheimer's Disease Neuroimaging Initiative. (n.d.). ADNI data. Retrieved October 22, 2024, from https://adni.loni.usc.edu/data-samples/adni-data/
- Wu, H., Xiao, B., Codella, N., Liu, M., Dai, X., Yuan, L., & Zhang, L. (2021). CvT: Introducing Convolutions to Vision Transformers. *CoRR*, abs/2103.15808. Available at: [https://arxiv.org/abs/2103.15808](https://arxiv.org/abs/2103.15808).


1. The readme file should contain a title, a description of the algorithm and the problem that it solves
(approximately a paragraph), how it works in a paragraph and a figure/visualisation.
2. It should also list any dependencies required, including versions and address reproduciblility of results,
if applicable.
3. provide example inputs, outputs and plots of your algorithm
4. The read me file should be properly formatted using GitHub markdown
5. Describe any specific pre-processing you have used with references if any. Justify your training, validation
and testing splits of the data.




