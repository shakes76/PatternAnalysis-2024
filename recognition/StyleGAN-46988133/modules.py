"""
modules.py created by Matthew Lockett 46988133
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparameters as hp

class MappingNetwork(nn.Module):
    """
    Within a StyleGAN model, the Mapping Network is used to transform a latent space vector z
    (a vector of random noise) into an intermediate latent vector w. The intermediate latent space
    vector is used to provide more control over the images features when compared to a normal GAN 
    model.

    REF: This class was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    """
    
    def __init__(self):
        """
        An instance of the MappingNetwork StyleGAN model.
        """
        super(MappingNetwork, self).__init__()

        layers = []
        layers.append(PixelNorm()) # Start by normalising the latent vector

        # For each layer complete a pass of one fully connected neural network and save
        for i in range(hp.MAPPING_LAYERS):
            layers.append(fully_connected(hp.LATENT_SIZE, hp.LATENT_SIZE))

        # Store all fully connected layers sequentially
        self.mapping = nn.Sequential(*layers)


    def forward(self, z):
        """
        Performs a forward pass on an input latent space vector, and turns it into
        an intermediate latent space vector w.
        
        Param: z: A latent space vector
        Return: The intermediate latent space vector w
        """
        return self.mapping(z)
    

class Generator(nn.Module):
    """
    Within the StyleGAN model, a generator is used to produce an output image that should accurately 
    represent the desired image classification, or in this case the ADNI dataset images. This generator 
    incorporates a mapping network which produces a style vector to influence the style of the 
    output image. It also uses a series of generator layers, which perform convolutions, noise injections
    and AdaIN passes to converge the feature map into the desired output image. 

    For the ADNI dataset, the images are of resolution 256x256, and thus this generator will start with a constant 
    4x4 resolution feature map, and scale it up to the full 256x256 resolution required, as a greyscale image.
    
    REF: This class was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    """

    def __init__(self):
        """
        An instance of the Generator class. Each generator requires a mapping network, a series of generator layers
        and a final output of an image with the number of channels being hp.NUM_CHANNELS.
        """
        # The first generator layer needs to start with a constant learned input feature map
        self.constant = nn.Parameter(torch.randn(1, hp.LATENT_SIZE, 4, 4))

        # Initialise the mapping network for this generator 
        self.mapping = MappingNetwork()

        # Models all generator layers from a 4x4 feature map up to a scaled 256x256 feature map
        self.gen_layers = nn.ModuleList([
            GenLayer(hp.LATENT_SIZE, hp.LATENT_SIZE, first_layer=True), # Output: 4x4 Feature Map
            GenLayer(hp.LATENT_SIZE, hp.LATENT_SIZE // 2),              # Output: 8x8 Feature Map
            GenLayer(hp.LATENT_SIZE // 2, hp.LATENT_SIZE // 4),         # Output: 16x16 Feature Map
            GenLayer(hp.LATENT_SIZE // 4, hp.LATENT_SIZE // 8),         # Output: 32x32 Feature Map
            GenLayer(hp.LATENT_SIZE // 8, hp.LATENT_SIZE // 16),        # Output: 64x64 Feature Map
            GenLayer(hp.LATENT_SIZE // 16, hp.LATENT_SIZE // 32),       # Output: 128x128 Feature Map
            GenLayer(hp.LATENT_SIZE // 32, hp.LATENT_SIZE // 64),       # Output: 256x256 Feature Map
        ])

        # Used to convert the feature map back into a greyscale image
        self.final_output = nn.Conv2d(hp.LATENT_SIZE // 64, hp.NUM_CHANNELS, kernel_size=1)


    def forward(self, z):
        """
        Completes a forward pass of the generator, fully converting the 4x4 constant feature map into a full 
        resolution 256x256 greyscale image resembling the ADNI dataset images.

        Param: z: A latent space vector input into the mapping network. 
        Return: A 256x256 greyscale image, based on the ADNI dataset.
        """
        # Generate the style vector 'w'
        w = self.mapping(z)

        # Create the constant feature map input into the generator
        x = self.constant

        # Pass the feature map into each layer of the generator
        for layer in self.gen_layers:
            x = layer(x, w)
            
        return torch.tanh(self.final_output(x))


class GenLayer(nn.Module):
    """
    Within the StyleGAN model, the generator will need multiple layers to upscale the 
    initial feature map size into the desired resolution of the final output image. Each layer
    also consists of a series of convolutions, noise injections and AdaIN passes, that are 
    ultimately used to generate desired feature maps within the generator. 
    
    REF: This function was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    REF: It was also inspired by the following website: 
    REF: https://blog.paperspace.com/implementation-stylegan-from-scratch/ 
    """

    def __init__(self, in_channels, out_channels, first_layer=False):
        """
        An instance of one GenLayer class. Each layer consists of multiple convolutions,
        noise injections and AdaIN passes. 

        Param: in_channels: The number of channels for the input feature map, x. 
        Param: out_channels: The number of channels for the output feature map of this layer
        Param: upscale: Determines if the feature map needs to be upscaled or not.
        """
        super(GenLayer, self).__init__()

        # Determines if this GenLayer is the first layer of the generator
        self.first_layer = first_layer

        # Model both convolutions needed with a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Model the noise injection into each layer, after each convolution
        self.noise1 = Noise(out_channels)
        self.noise2 = Noise(out_channels)

        # Need to perform a non-linear activation after the noise injection 
        self.lrelu = nn.LeakyReLU(hp.LRELU_SLOPE_ANGLE, inplace=True)

        # Model the Adaptive Instance Normalisation (AdaIN) after each noise injection
        self.adain1 = AdaIN(out_channels)
        self.adain2 = AdaIN(out_channels)


    def forward(self, x, w):
        """
        Performs a forward pass of one generator layer. There is a distinction between the first 
        layer of the generator and subsequent layers, wherein the first layer does not need upscaling
        and an extra convolution is not needed.

        Param: x: The input feature map into this generator layer. 
        Param: w: The input style vector into this generator layer.
        """
        # This is the first layer of the generator, do not include upscaling or extra conv
        if self.first_layer:
            x = self.adain1(self.lrelu(self.noise1(x)), w)

        # Upscale by doubling the resolution of the feature map and apply extra conv
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.adain1(self.lrelu(self.noise1(self.conv1(x))), w)

        x = self.adain2(self.lrelu(self.noise2(self.conv2(x))), w)

        return x
        

def fully_connected(in_channels, out_channels):
    """
    Represents one fully connected layer of a standard neural network with leaky
    ReLu activations.

    Param: in_channels: The size of the input vector to this layer
    Param: out_channels: The size of the output vector this layer creates
    Return: A sequential model of one fully connected layer 
    REF: This function was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    """
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.LeakyReLU(hp.LRELU_SLOPE_ANGLE)
    )


class PixelNorm(nn.Module):
    """
    Used to perform pixel normalisation on a vectorised image or latent space vector. The vector
    is normalised relative to it's own mean and variance.

    REF: This code was inspired by the following website:
    REF: https://blog.paperspace.com/implementation-stylegan-from-scratch/
    """

    def __init__(self):
        """
        An instance of the PixelNorm class.
        """
        super(PixelNorm, self).__init__()
        self.epsilon = hp.EPSILON # Used to avoid divison by zero

    def forward(self, x):
        """
        Normalises the input vector x, reltaive to it's own mean and variance. 

        Param: x: The input vector to be normalised.
        Return: A normalised output vector.
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon) 


class AdaIN(nn.Module):
    """
    Within a StyleGAN model, the Adaptive Instance Normalisation (AdaIN) is used
    to adjust the mean and variance of feature maps based on the style vector w. This
    injection of the style vector's information allows the model to control the output of
    the image's style.

    REF: This function was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    REF: It was also inspired by the following website: 
    REF: https://blog.paperspace.com/implementation-stylegan-from-scratch/ 
    """

    def __init__(self, channels):
        """
        An instance of the AdaIN class. The instance normalisation is initialised based on the 
        channels of the input tensor x, and the scale and shift vectors are created as a 
        fully connected layer each.

        Param: channels: The amount of channels of the input tensor x.
        """
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(num_features=channels, eps=hp.EPSILON)
        self.style_scale = fully_connected(hp.LATENT_SIZE, hp.LATENT_SIZE)
        self.style_shift = fully_connected(hp.LATENT_SIZE, hp.LATENT_SIZE)

    def forward(self, x, w):
        """
        Performs AdaIN on the input tensor x, based on the input style vector w.

        Param: x: An input tensor representing a layer of information within the generator. 
        Param: w: The style vector that is used to scale and shift the x tensor.
        Return: The AdaIN of the x and w tensors.
        """
        x = self.instance_norm(x)
        scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        shift = self.style_shift(w).unsqueeze(2).unsqueeze(3)

        return scale * x + shift
    

class Noise(nn.Module):
    """
    Within a StyleGAN model, the generator receives multiple injections of noise into each of
    it's layers to improve the output image's finer details and quality. This class facilitates 
    the noise injections, and even creates a learnable noise parameter to improve training further.

    REF: This class was inspired by the following website:
    REF: https://blog.paperspace.com/implementation-stylegan-from-scratch/
    """

    def __init__(self, channels):
        """
        An instance of the Noise class. A learnable noise parameter is created so that training can be
        enhanced by learning an appropriate noise scale. 

        Param: channels: The amount of channels within the input tensor x.
        """
        super(Noise, self).__init__()
        self.learned_noise = nn.Parameter(torch.zeros(1, channels, 1, 1))


    def forward(self, x):
        """
        Generates random noise and scales it by a learned factor, then applies it to the input tensor x.

        Param: x: An input tensor x, representing the information within a layer of the generator. 
        Return: The input tensor x with additional noise added.
        """
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.learned_noise * noise



