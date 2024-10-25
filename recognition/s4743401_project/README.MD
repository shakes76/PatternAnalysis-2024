#For ViT, we'll need the below for modules. 

1. Patch embedding
    - input image is divided into square patches. 
      Each patch is then linearly transformed into a vector.
      This results in a sequence of patch embeddings, which serve as the input tokens for the subsequent layers.

2. Positional embedding
    - Adds positional information
    - learned and added to the patch embeddings at the input stage
    
3. Encoder layers
    - Multi-Head Self-Attention
        - self-attention mechanism to capture the relationships between different patches in the input sequence
        - computes a weighted sum of all patch embeddings, where the weights are determined by the relevance of 
          each patch to the current one
        - keeps focus on important patches

    - Feedforward Neural Networks
        -  introduce non-linearity and allow the model to learn complex relationships between patches

    - Layer Normalization and Residual Connections
        - helps stabilize and speed up training by normalizing the inputs to each sub-layer
        - Residual connections(IF WE USE THEM), also known as skip connections, 
          add the original input embeddings to the output of each sub-layer
          
https://medium.com/thedeephub/building-vision-transformer-from-scratch-using-pytorch-an-image-worth-16x16-words-24db5f159e27
https://www.philschmid.de/image-classification-huggingface-transformers-keras
https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
