"""
Source code of parts for the vision transformer (i.e the ML model).
From the paper
https://arxiv.org/pdf/2010.11929, the main idea is to
- Split image into patches
> Apply a sequence of linear embeddings (like, instead of tokens/text, do it on the patches)
And that's the input to the transformer

I.e image patches == tokens. Then we just continue as a normal transformer.

https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/11-vision-transformer.html

https://www.akshaymakes.com/blogs/vision-transformer
I.e


1. Create patch embeddings
2.  Pass embedddings through transformer blocks
3. Perform classification

See exploration.ipynb for more details.
"""
import torch
from torch import nn
from torchinfo import summary

BATCH_SIZE = 32

PATCH_SIZE = 16
IMAGE_WIDTH = 224
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_CHANNELS = 3
EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE**2 # 16 * 16 * 3 = 768
NUM_OF_PATCHES = int((IMAGE_WIDTH * IMAGE_HEIGHT) / PATCH_SIZE**2)
print(EMBEDDING_DIMS, NUM_OF_PATCHES)




class PatchEmbeddingLayer(nn.Module):
    """
    Turns 2D images from the 

    This effectively converts images to 'tokens' (like word tokens), in a way, so then we can use transformer terminology and concepts like usual.
    """
    def __init__(self, in_channels, patch_size, embedding_dim,):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels

        ## For each 16x16 patch, create an embedding vector (embedding layer thing), with size 768.
        # I.e each embedding vector is 16x16x768; can be done with a CONV2D layer!! (https://www.akshaymakes.com/blogs/vision-transformer) <--- GOATED
        self.conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size)
        
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=2)

        self.class_token_embeddings = nn.Parameter(torch.rand((BATCH_SIZE, 1, EMBEDDING_DIMS), requires_grad=True))
        
        self.position_embeddings = nn.Parameter(torch.rand((1, NUM_OF_PATCHES + 1, EMBEDDING_DIMS), requires_grad=True)) # [batch_size, num_patches+1, embeddings_dims]

    def forward(self, x):
        """
        Input: 
        - x: the input image tensor with shape (batch_size, channels, height, width), e.g., (32, 3, 224, 224)
        """
        # 1. Apply the convolutional layer to the input, which divides it into patches and projects them into the embedding space.
        patches = self.conv_layer(x)  # Output shape: (batch_size, embedding_dim, num_patch_rows, num_patches_cols) e.g [32, 768, 14, 14].
        
        # 2. Rearrange the tensor dimensions to (B, NUM_PATCHROW, NUM_PATCHCOL , EMBEDDING_DIMS)

        patches = patches.permute((0, 2, 3, 1))  
        
        # 3. Flatten the patches. The output shape will now be [batch_size, num_of_patches, embedding_dims] e.g [32, 196, 768].
        patches_flattened = self.flatten_layer(patches)
        
        # 4. Concatenate the class token embeddings at the start of the sequence of patches.
        # This class token is used for the final classification.
        output = torch.cat((self.class_token_embeddings, patches_flattened), dim=1) # [batch_size, num_of_patches+1, embeddiing_dims]
        
        # 5. Add positional embeddings to the patch embeddings to retain positional information.
        output += self.position_embeddings # e.g [32, 197, 768]
        
        return output  # Output shape: (batch_size, num_patches + 1, embedding_dim) e.g [32, 197, 768]