#the source code of the components of your model. Each component must be
#implementated as a class or a function

import sys
import torch
import torch.nn as nn
#from pyimgaug3d.augmentation import GridWarp, Flip, Identity

# dice loss function from
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch?fbclid=IwAR3q7bjIDoKFlc5IDGpd24TW8QhQdzbxh2TrIP6FCXb7A8FaluU_HhTqmHA
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predict, target):
        # flatten tensors
        predict = predict.view(-1)
        target = target.view(-1)

        # calculate the intersect value
        intersect = (predict * target).sum()
        # compute dice score
        dice = (2.*intersect + self.smooth)/(predict.sum() + target.sum() + self.smooth)

        return 1 - dice

def test_GPU_connection(force_CPU):
    '''
    tests whether pytorch can detect a GPU.
    If no GPU is detected, prints an error message and ends the program.
    Above behaviour overridden by force_CPU, given at runtime

    Parameters:
    force_CPU (Bool): if true, the model will use CPU to make the model. default: false
    
    return:
        none
    '''
    if torch.cuda.is_available():
        return 'cuda'
    elif force_CPU:
        return 'cpu'
    else:
        sys.exit("No GPU detected")
        return


class conv_seg(nn.Module):
    '''
    class for convolution steps of Unet
    '''
    def __init__(self, in_channel_n, out_channel_n):
        super(conv_seg, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel_n, out_channel_n, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel_n),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel_n, out_channel_n, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel_n),
            nn.ReLU(inplace=True),
        )

    def forward(self, out):
        return self.conv(out)

class Unet2d(nn.Module):
    def __init__(self, in_channel_n=1, out_channel_n=1, features=[64, 128, 256, 512]):
        super(Unet2d, self).__init__()
        self.upward = nn.ModuleList()
        self.downward = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)

        # downwards portion of Unet
        for feature in features:
            self.downward.append(conv_seg(in_channel_n, feature))
            in_channel_n = feature

        # bottom of the Unet
        self.bottom = conv_seg(features[-1], features[-1]*2)

        #upwards portion of Unet
        for feature in reversed(features):
            self.upward.append(nn.ConvTranspose3d(feature*2, feature, 2, 2))
            self.upward.append(conv_seg(feature*2, feature))

        #final conv layer
        self.last = nn.Conv2d(features[0], out_channel_n, 1)

    def forward(self, out):
        skips = []
        # downwards portion
        for down in self.downward:
            out = down(out)
            skips.append(out)
            out = self.pool(out)

        # bottom portion
        out = self.bottom(out)
        skips.reverse()

        # upwards portion
        for i in range(0, len(self.upward), 2):
            out = self.upward[i](out)
            skip = skips[i//2]
            #if out 
            out = torch.cat((skip, out), dim=1)
            out = self.upward[i+1](out)

        # final conv layer
        out = self.last(out)
        return out
        
def test():
    x = torch.randn((3,1,160,160))
    model = Unet2d(1,1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape[0] == x.shape[0], "Batch size mismatch"
    assert preds.shape[2:] == x.shape[2:], "Spatial dimensions mismatch"
    assert preds.shape[1] == 1, "Output channels mismatch"

if __name__ == "__main__":
    #put on the cluster
    test()


        