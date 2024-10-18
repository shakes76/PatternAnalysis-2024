import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import functools
class VisionTransformer(nn.Module):

    def __init__(self, depth:int,  patchDim: int, imgDims: tuple[int, int, int], patchLen: int, mhaHeadFactor:int, hiddenMul: int, device: str = "cpu") -> None:
        super(VisionTransformer, self).__init__()
        self.pDim = patchDim
        self.imgDims = imgDims
        self.pLen = patchLen
        self.device = device
        self.mhaHeadFactor = mhaHeadFactor
        self.depth = depth
        self.hiddenMul = hiddenMul

        self.pSplitter = PatchSplitter(self.pDim, self.imgDims, self.pLen, device = self.device)

        self.nPatches  = self.pSplitter.nPatches
        
        self.n1 = nn.LayerNorm(self.pLen, device = self.device)
        self.l1 = nn.Linear(self.pLen, 2, device = self.device)

        self.pProcessor = PatchProcessor(
            self.nPatches,
            self.pLen,
            self.device
        )

        self.softMax = nn.Softmax(dim = -1)

        assert self.pLen % self.mhaHeadFactor == 0
        self.blocks = nn.ModuleList(
            [TransformerEncBlk(self.pLen, self.pLen // self.mhaHeadFactor, self.hiddenMul, device = self.device) for i in range(self.depth)]
        )
    
    def forward(self, x):
        x= self.pSplitter(x)
        x = self.pProcessor(x)
        for block in self.blocks:
            x = block(x)
        x = self.l1(self.n1(x)[:, 0])
        return x

class TransformerEncBlk(nn.Module):
    def __init__(self, dim: int, nHeads: int, hiddenMul:int , device:str = 'cpu') -> None:
        super(TransformerEncBlk, self).__init__()
        self.dim = dim
        self.nHeads = nHeads
        self.hiddenMul = hiddenMul
        self.device = device

        self.n1 = nn.LayerNorm(self.dim, device = self.device)
        self.mha = nn.MultiheadAttention(
            self.dim,
            self.nHeads,
            dropout = 0.2,
            batch_first = True,
            device = self.device
        )
        self.n2 = nn.LayerNorm(self.dim, device = self.device)
        self.l1 = nn.Linear(self.dim, self.dim * self.hiddenMul, device = self.device)
        self.d1 = nn.Dropout(p = 0.2)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(self.dim * self.hiddenMul, self.dim, device = self.device)

    def forward(self, x: Tensor) -> Tensor:
        y = self.n1(x)
        y, _ = self.mha(y, y, y)
        x = x + y
        z = self.d1(self.activation(self.l1(self.n2(x))))
        z = self.l2(z)
        x = x + z
        return x

class PatchProcessor(nn.Module):
    '''
    PatchProcessor module for VisionTransformer.
    Adds the class prediction token to the patches and then 
    performs sinusoidal encoding to the result
    '''
    def __init__(self, nPatches:int, patchLen: int, device: str = "cpu") -> None:
        '''
        Inputs:
            batchSize: int - number of images taken in the batch
            nPatches: number of patches for each image
            patchLen: int - Length of flattened patches
            device: CUDA device to be computed on
        Returns: None
        '''
        super(PatchProcessor, self).__init__()
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
        pToken = torch.zeros(x.size()[0], 1, self.patchLen, device = self.device)
        return torch.cat((pToken, x), dim = 1)

    def encodePos(self, x: Tensor):
        '''
        Applies sinusoidal encoding to the given patches

        Input:
            x: Tensor - Pytorch Tensor of flattened imgs
        Returns:
            Tensor - x after sinusodial encoding
        '''

        def singleTokenArr(i):
            return [i / np.power(10000, 2 * (j // 2) / self.patchLen) for j in range(self.patchLen)] 

        posTable = np.array([singleTokenArr(i) for i in range(self.nPatches + 1)])
        posTable[:, 0::2] = np.sin(posTable[:, 0::2])
        posTable[:, 1::2] = np.sin(posTable[:, 1::2])

        posEncoding = torch.tensor(posTable, device = self.device, dtype = torch.float32).unsqueeze(0)
        return x + posEncoding
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        Compute the forward step of the Patch Processor,
        adds the prediction token and performs positional 
        encoding on the given tensor

        Inputs:
            x: Tensor - pytorch tensor of the flattened img patches
        Returns:
            Tensor - Computed values after patch processing
        '''
        return self.encodePos(self.addPredToken(x))

class PatchSplitter(nn.Module):
    '''
    PatchSplitter module for VisionTransformer.
    Splits image into the required number of patches based of patchLen
    and runs subsequent patches through an initial linear layer
    '''
    def __init__(self, patchLen: int, imgDims: tuple[int, int, int], dimOut: int, device: str = 'cpu') -> None:
        '''
        Inputs:
            patchLen: int - Length/Width of the sqaure patches to be generated
            imgDims: (int, int, int) - (channels, height, width) of the img
            dimOut: int - output dimension of each patch's linear layer
        Returns: None
        '''
        super(PatchSplitter, self).__init__()
        self.patchLen = patchLen
        self.imgDims = imgDims
        self.tokenOut = dimOut
        self.device = device
        channels, height, width = self.imgDims
        #* Ensure that the image can be correctly split into patches
        assert height % self.patchLen == 0
        assert width % self.patchLen == 0
        self.nPatches = (height // self.patchLen) * (width // self.patchLen)
        self.partition = nn.Unfold(
            kernel_size = self.patchLen,
            stride = self.patchLen,
            padding = 0,
        )
        self.linear = nn.Linear((self.patchLen ** 2) * channels, dimOut, device = self.device)
    
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
        return self.linear(self.partition(x).transpose(2, 1))
    
class ConvolutionalPatchSplitter(nn.Module):
    '''
    #TODO
    '''
    def __init__(self, kernal: int, imgDims: tuple[int, int, int], outputDim: int, stride: int, padding: int, device: str = 'cpu') -> None:
        '''
        #TODO
        '''
        super(ConvolutionalPatchSplitter, self).__init__()
        self.kernal = kernal
        self.imgDims = imgDims
        self.stride = stride
        self.device = device
        self.outputDim = outputDim
        self.padding = padding
        channels, height, width = self.imgDims
        #* Ensure that the image can be correctly split into patches
        assert height % self.kernal == 0
        assert width % self.kernal == 0
        self.conv = nn.Conv2d(channels, self.outputDim, self.kernal, self.stride, padding= self.padding, device = self.device)
        self.norm = nn.BatchNorm2d(self.outputDim, device = self.device)
    
    def forward(self, x: Tensor) -> Tensor:
        '''
        #TODO
        '''
        return self.norm(self.conv(x))
    
class ConvolutionalTransformerEncBlk(nn.Module):
    def __init__(self, imgDim: tuple[int, int, int], nHeads: int, hiddenMul:int, device:str = 'cpu', final:bool = False) -> None:
        super(ConvolutionalTransformerEncBlk, self).__init__()
        self.imgDim = imgDim
        self.dim = self.imgDim[0]
        self.nHeads = nHeads
        self.hiddenMul = hiddenMul
        self.device = device
        self.final = final

        self.n1 = nn.LayerNorm(self.dim, device = self.device)
        self.conv = nn.Conv2d(self.imgDim[0], self.imgDim[0], 3, 1, 1, device=self.device)
        self.conv2 = nn.Conv2d(self.imgDim[0], self.imgDim[0], 3, 2, 1, device=self.device)
        print(self.nHeads)
        print(self.dim)
        self.mha = nn.MultiheadAttention(
            self.dim,
            self.nHeads,
            dropout = 0.5,
            batch_first = True,
            device = self.device
        )
        self.n2 = nn.LayerNorm(self.dim, device = self.device)
        self.l1 = nn.Linear(self.dim, self.dim * self.hiddenMul, device = self.device)
        self.d1 = nn.Dropout(p = 0.5)
        self.activation = nn.GELU()
        self.l2 = nn.Linear(self.dim * self.hiddenMul, self.dim, device = self.device)
        self.d2 = nn.Dropout(p = 0.5)

    def forward(self, x: Tensor) -> Tensor:
        size = x.size()[-1]
        q = rearrange(self.conv(x), 'b c h w -> b (h w) c')
        k = rearrange(self.conv2(x), 'b c h w -> b (h w) c')
        v = rearrange(self.conv2(x), 'b c h w -> b (h w) c')
        x = rearrange(x, 'b c h w -> b (h w) c')
        y, _ = self.mha(q, k, v)
        x = x + y
        z = self.d1(self.activation(self.l1(self.n2(x))))
        x = x + z
        return rearrange(x, 'b (h w) c -> b c h w', h = size, w = size)
    
class ConvolutionalTransformerBlk(nn.Module):
    def __init__(self, imgDims: tuple[int, int, int], embChannels:int, kernal:int, stride:int, padding:int, nHeads:int, hiddenMul: int,
                depth:int, final:bool = False, device: str = "cpu") -> None:
        super(ConvolutionalTransformerBlk, self).__init__()
        self.imgDims = imgDims
        self.kernal = kernal
        self.embChannels = embChannels
        self.stride = stride
        self.padding = padding
        self.device = device
        self.nHeads = nHeads
        self.hiddenMul = hiddenMul
        self.final = final
        self.depth = depth

        self.enc = ConvolutionalPatchSplitter(
            self.kernal,
            self.imgDims,
            self.embChannels,
            self.stride,
            self.padding,
            self.device
        )
        newsize = int((self.imgDims[-1] + 2 * self.padding - self.kernal) / self.stride + 1)
        self.newImgdims = (self.embChannels + 1 if self.final else self.embChannels, newsize, newsize)
        
        
        self.n1 = nn.LayerNorm(int(self.newImgdims[0]), device = self.device)
        self.l1 = nn.Linear(int(self.newImgdims[0]), 2, device = self.device)

        self.blocks = nn.ModuleList(
            [ConvolutionalTransformerEncBlk(self.newImgdims, self.nHeads, self.hiddenMul, self.device) for i in range(self.depth)]
        )
    
    def forward(self, x):
        x = self.enc(x)
        if (self.final):
            size = [s for s in x.size()]
            size[1] = 1
            classToken = torch.rand(size, device=self.device)
            x = torch.cat((classToken, x), dim = 1)
        for block in self.blocks:
            x = block(x)
        if (self.final):
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.l1(self.n1(x)[:, 0])
        return x
    
class ConvolutionalVisionTransformer(nn.Module):
    def __init__(self, device):
        super(ConvolutionalVisionTransformer, self).__init__()
        self.device = device

        self.b1 = ConvolutionalTransformerBlk((1, 224, 224), 64, 7, 4, 0, 1, 1, 1, device=self.device)

        self.b2 = ConvolutionalTransformerBlk((64, 56, 56), 192, 3, 2, 0, 3, 1, 2, device=self.device)

        self.b3 = ConvolutionalTransformerBlk((192, 28, 28), 383, 3, 2, 0, 6, 1, 8, device=self.device, final=True)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x  
