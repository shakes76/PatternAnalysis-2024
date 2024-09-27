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
    
class MultiHeadSelfAttentionBlock(nn.Module):
  def __init__(self,
               embedding_dims = 768, # Hidden Size D in the ViT Paper Table 1
               num_heads = 12,  # Heads in the ViT Paper Table 1
               attn_dropout = 0.0 # Default to Zero as there is no dropout for the the MSA Block as per the ViT Paper
               ):
    super().__init__()

    self.embedding_dims = embedding_dims
    self.num_heads = num_heads
    self.attn_dropout = attn_dropout

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)

    self.multiheadattention =  nn.MultiheadAttention(num_heads = num_heads,
                                                     embed_dim = embedding_dims,
                                                     dropout = attn_dropout,
                                                     batch_first = True,
                                                    ) # Expects (batchsize, seq_len, embed_dim)

  def forward(self, x):
    x = self.layernorm(x)
    output,_ = self.multiheadattention(query=x, key=x, value=x,need_weights=False)
    return output


class MachineLearningPerceptronBlock(nn.Module):
  def __init__(self, embedding_dims, mlp_size, mlp_dropout):
    super().__init__()
    self.embedding_dims = embedding_dims
    self.mlp_size = mlp_size
    self.dropout = mlp_dropout

    self.layernorm = nn.LayerNorm(normalized_shape = embedding_dims)
    self.mlp = nn.Sequential(
        nn.Linear(in_features = embedding_dims, out_features = mlp_size),
        nn.GELU(),
        nn.Dropout(p = mlp_dropout),
        nn.Linear(in_features = mlp_size, out_features = embedding_dims),
        nn.Dropout(p = mlp_dropout)
    )

  def forward(self, x):
    return self.mlp(self.layernorm(x))

# https://aws.amazon.com/what-is/transformers-in-artificial-intelligence/#:~:text=Each%20transformer%20block%20has%20two,the%20input%20when%20making%20predictions.
class TransformerBlock(nn.Module):
  def __init__(self, embedding_dims = 768,
               mlp_dropout=0.1,
               attn_dropout=0.0,
               mlp_size = 3072,
               num_heads = 12,
               ):
    super().__init__()

    self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims = embedding_dims,
                                                 num_heads = num_heads,
                                                 attn_dropout = attn_dropout)

    self.mlp_block = MachineLearningPerceptronBlock(embedding_dims = embedding_dims,
                                                    mlp_size = mlp_size,
                                                    mlp_dropout = mlp_dropout,
                                                    )

  def forward(self,x):
    x = self.msa_block(x) + x # Skip connections
    x = self.mlp_block(x) + x

    return x
  
class ViT(nn.Module):
  def __init__(self, img_size = 224,
               in_channels = 3,
               patch_size = 16,
               embedding_dims = 768,
               num_transformer_layers = 12, # from table 1 above
               mlp_dropout = 0.1,
               attn_dropout = 0.0,
               mlp_size = 3072,
               num_heads = 12,
               num_classes = 1000):
    super().__init__()

    ## Get the patch layer.
    self.patch_embedding_layer = PatchEmbeddingLayer(in_channels = in_channels,
                                                     patch_size=patch_size,
                                                     embedding_dim = embedding_dims)

    # Throw it in the encoder
    # ViT's are encoder only architectures!! (I think)!!
    # The * unpacks the list generated from list comprehension, to instead be a ton of params for Sequential e.g Sequential(tb1, tb2, tb3, tb...).
    self.transformer_encoder = nn.Sequential(*[TransformerBlock(embedding_dims = embedding_dims,
                                              mlp_dropout = mlp_dropout,
                                              attn_dropout = attn_dropout,
                                              mlp_size = mlp_size,
                                              num_heads = num_heads) for _ in range(num_transformer_layers)])


    # Classify it
    # eq 4 if the ViT paper 2020
    self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape = embedding_dims),
                                    nn.Linear(in_features = embedding_dims,
                                              out_features = num_classes))

  def forward(self, x):
    # Get patch embeddings from the image
    patches = self.patch_embedding_layer(x)
    
    # Pass the patch embeddings through the transformer encoder
    encoded_patches = self.transformer_encoder(patches)
    
    # Extract the CLS token (the first token in the sequence)
    cls_token = encoded_patches[:, 0]
    
    # Pass the CLS token through the classifier for final output
    output = self.classifier(cls_token)
    
    return output
  
