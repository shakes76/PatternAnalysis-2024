import torch
import torch.nn as nn
from torch import Tensor

class VisionTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

class PatchProcesser(nn.Module):
    
    def __init__(self, batchSize: int, nPatches:int, patchLen: int, device: str = "cpu") -> None:
        super().__init__()
        self.batchSize = batchSize
        self.nPatches = nPatches
        self.patchLen = patchLen
        self.device = device

    def addPredToken(self, x: Tensor):
        '''
        Adds the class predition token to
        each images set of flattend 
        linear layers.

        Input:
            x: Tensor - Pytorch Tensor of flattened imgs
        Returns:
            Tensor - x with additional blank tokens for class predictions
        '''
        print(x)
        pToken = torch.zeros(self.batchSize, 1, self.patchLen).to(self.device)
        print(pToken)
        return torch.cat((pToken, x), dim = 1)

    def encodePos(self, x: Tensor):
        pass

class PatchSplitter(nn.Module):
    '''
    PatchSplitter module for VisionTransformer.
    Splits image into the required number of patches based of patchLen
    and runs subsequent patches through an initial linear layer
    '''
    def __init__(self, patchLen: int, imgDims: tuple[int, int, int], dimOut: int) -> None:
        '''
        Inputs:
            patchLen: int - Length/Width of the sqaure patches to be generated
            imgDims: (int, int, int) - (channels, height, width) of the img
            dimOut: int - output dimension of each patch's linear layer
        Returns: None
        '''
        super().__init__()
        self.patchLen = patchLen
        self.imgDims = imgDims
        self.tokenOut = dimOut
        b, channels, height, width = self.imgDims
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
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute forward step of the PatchSplitter Module,
        through both an unfold layer to split the img into
        pathces and then an initial linear layer.

        Input:
            x: Tensor - pytorch tensor representing the img, must
                match the dimmensions specified of the PatchSplitter
        Returns: Tensor - Computed values after the final linear Layer
        '''
        #assert x.size() == self.imgDims #* Checking given img matches img dims
        x = self.partition(x).transpose(2, 1)
        x = self.linear(x)
        return x