import torch
import torch.nn as nn
from GFNet.gfnet import GFNet, GFNetPyramid

class GFNetClassifier(nn.Module):
    def __init__(self, num_classes=2, img_size=224):
        super(GFNetClassifier, self).__init__()
        #Load gfnet-h-b
        # self.model = GFNetPyramid(
        #     img_size=img_size,
        #     patch_size=4,
        #     num_classes=num_classes,
        #     embed_dim=[96, 192, 384, 768],
        #     depth=[2, 2, 10, 2],
        #     mlp_ratio=[4, 4, 4, 4],
        #     drop_path_rate=0.3,
        # )

        #Load gfnet-h-ti
        self.model = GFNetPyramid(img_size=224, patch_size=4, num_classes=1000)  

        #Load gfnet-xs
        #self.model = GFNet(img_size=224, patch_size=16, num_classes=1000, embed_dim=384)  

        self.head = self.model.head
        self.blocks = self.model.blocks

    def forward(self, x):
        return self.model(x)


def load_pretrained_model(num_classes=2, img_size=224):
    model = GFNetClassifier(num_classes=num_classes, img_size=img_size)
    
    # Load the pretrained weights
    state_dict = torch.load("gfnet-h-ti.pth")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Remove any keys that don't match the new model architecture
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)

    # Load the filtered state dict
    model.load_state_dict(model_dict, strict=False)
    
    return model