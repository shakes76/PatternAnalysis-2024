import torch.nn as nn

class PatchSplitter(nn.Module):
    def __init__(self, patchLen: int, imgDims: tuple[int, int, int], dimOut) -> None:
        super().__init__()
        self.patchLen = patchLen
        self.imgDims = imgDims
        self.tokenOut = dimOut
        channels, height, width = self.imgDims
        #* Ensure that the image can be correctly split into patches
        assert height % self.patchLen == 0
        assert width % self.patchLen == 0
        self.nPatches = (height / self.patchLen) * (width / self.patchLen)
        self.partition = nn.Unfold(
            kernel_size = self.patchLen,
            stride = self.patchLen,
            padding = 0
        )
        self.linear = nn.Linear((self.patchLen ** 2) * channels, dimOut)
    
    def forward(self, x):
        x = self.partition(x).transpose(1, 0)
        x = self.linear(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

