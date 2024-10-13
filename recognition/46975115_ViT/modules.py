import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, img_size=224, patch_size=16, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_classes = num_classes
        
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), num_layers=12
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        #path embeding
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        #cls
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        #transformer encoder
        x = self.transformer_encoder(x)
        
        #dropout
        x = self.dropout(x)
        
        #clf head
        cls_output = self.head(x[:, 0])
        return cls_output
