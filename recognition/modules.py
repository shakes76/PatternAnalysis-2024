import torch
import torch.nn as nn

class encoderBlock(nn.Module):
	#convolution+ReLU+maxPool
	def __init__(self, in_channels, out_channels):
		super(encoderBlock, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
		)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		x = self.conv(x)
		pooled = self.pool(x)
		return x, pooled

def decoderBlock():
	#convolution+concatenation+convolution+ReLU
    def __init__(self, inchannels, out_channels):
        super(decoderBlock, self).__init()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride = 2)
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace=True)
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
                nn.ReLU(inplace=True),
                )

    def forward(self, x, skip_features):
        x = self.upconv(x)
        x = torch.cat((x, skip_features), dim = 1) 
        x = self.conv(x)
        return x

def bottleneck():
	#convolutionalLayer1
	#convolutionalLayer2

def skipConnections():
    
def outputLayer():

