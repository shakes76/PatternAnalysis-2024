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
        
        Param: z: A latent space vector.
        Return: The intermediate latent space vector w.
        """
        return self.mapping(z)
    

class Generator(nn.Module):
    """
    Within the StyleGAN model, a generator is used to produce an output image that should accurately 
    represent the desired image classification, or in this case the ADNI dataset images. This generator 
    incorporates a mapping network which produces a style vector to influence the style of the 
    output image. It also uses a series of generator layers, which perform convolutions, noise injections
    and AdaIN passes to converge the feature map into the desired output image. Label embedding and progressive image 
    resolution generation have also been implemented.

    For the ADNI dataset, the images are of resolution 256x256, and thus this generator will start with a constant 
    4x4 resolution feature map, and scale it up to grayscale images in the range of 8x8 to 256x256 to progressively 
    be trained. The final desired resolution will be 256x256.
    
    REF: This class was inspired by code generated by ChatGPT-4o via the following prompts:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    REF: Prompt: How can I use nn.Embedding to embed AD and CN labels into the latent space vector of a StyleGAN?
    REF: Prompt: How do I employ progressive GAN tactics to a StyleGAN to generate images of various sizes, so that 
    REF: it can be progressively trained. 
    """

    def __init__(self):
        """
        An instance of the Generator class. Each generator requires a mapping network, a series of generator layers
        and a final output of an image with the number of channels being hp.NUM_CHANNELS.
        """
        super(Generator, self).__init__()

        # The first generator layer needs to start with a constant learned input feature map
        self.constant = nn.Parameter(torch.ones((1, hp.LATENT_SIZE, 4, 4)))

        # Initialise the mapping network for this generator 
        self.mapping = MappingNetwork()

        # Stores multiple from_gray and Genlayers depending on image resolution
        self.from_gray = nn.ModuleList() # Converts feature map into image
        self.gen_layers = nn.ModuleList() # Converts style vector into feature map

        # Generate from_gray and GenLayer lists
        for i in range(len(hp.GEN_FACTORS) - 1):

            # Special case for the first layer of the generator
            if i == 0:
                in_channels = int(hp.GEN_FACTORS[i])
                self.gen_layers.append(GenLayer(in_channels, in_channels, first_layer=True))
                self.from_gray.append(nn.Conv2d(in_channels, hp.NUM_CHANNELS, kernel_size=1))

            # All other layers
            else:
                in_channels = int(hp.GEN_FEATURE_SIZE * hp.GEN_FACTORS[i])

            out_channels = int(hp.GEN_FEATURE_SIZE * hp.GEN_FACTORS[i + 1])

            # Generate GenLayers and from_gray layers based on input and output sizes
            self.gen_layers.append(GenLayer(in_channels, out_channels))
            self.from_gray.append(nn.Conv2d(out_channels, hp.NUM_CHANNELS, kernel_size=1))

        # Used to embed the AD and CN labels into the style vector
        self.label_embedding = torch.nn.Embedding(num_embeddings=hp.LABEL_DIMENSIONS, embedding_dim=hp.EMBED_DIMENSIONS)
        self.embedding_layer = fully_connected(in_channels=hp.LATENT_SIZE + hp.EMBED_DIMENSIONS, out_channels=hp.LATENT_SIZE)


    def forward(self, z1, z2=None, mixing_ratio=hp.MIXING_RATIO, labels=None, depth=0, alpha=1.0):
        """
        Completes a forward pass of the generator, fully converting the 4x4 constant feature map into the desired resolution 
        based on the depth of the network. During the forward pass, the methods of mixing regularisation, label embedding and 
        alpha blending for progressive training are all employed. The final output for the lowest depth should produce 
        a 256x256 grayscal image representing an image from the ADNI dataset.

        Param: z1: A latent space vector input into the mapping network. 
        Param: z2: A potential second latent space vector used for mixing regularisation.
        Param: mixing_ratio: A ratio used for mixing regularisation.
        Param: labels: The AD and CN labels that will be embedded into the style vector if passed.
        Param: depth: The current depth of the network which indicates image resolution output.
        Param: alpha: A value between 0.0 and 1.0, used for blending two image resolutions together.
        Return: A grayscale image with resolution based on the depth of the network.
        REF: The mixing regularisation portion of this function was inspired by code generated by ChatGPT-4o
        REF: Based on the following prompt: How can I implement mixing regularization into my StyleGAN model?
        """
        # Generate a style vector
        w1 = self.mapping(z1)

        # Apply mixing of style vectors if required
        if z2 is not None:
            w2 = self.mapping(z2)
            w = self.mix_style_vectors(w1, w2, mixing_ratio)
        else:
            w = w1

        if labels is not None:
            embedded_labels = self.label_embedding(labels)

            # Embed the labels into the style vector
            w = self.embedding_layer(torch.cat([w, embedded_labels], dim=1))

        # Create the constant feature map input into the generator
        x = self.constant

        # Pass the feature map and mixed style vector into each layer of the generator based on current depth
        for i, layer in enumerate(self.gen_layers[:depth + 2]):
            x = layer(x, w)

            # Store previous lower-resolution feature map for blending
            if i == depth:
                x_old = x

        # Blend the old resolution with the new current resolution if required
        if (depth > 0) and (alpha < 1.0):

            # Convert the old feature map into an image 
            out_old = self.from_gray[depth](x_old)

            # Convert the new feature map into an image
            out_new = self.from_gray[depth + 1](x)

            # Upscale the old image to match the same size as the new image
            out_old = F.interpolate(out_old, size=out_new.shape[-2:], mode='bilinear', align_corners=False)

            # Blend the old and new images together
            out = alpha * out_new + (1 - alpha) * out_old

        # No blending required, convert current resolution feature map to image
        else:
            out = self.from_gray[depth + 1](x)
            
        return torch.tanh(out)
    

    def mix_style_vectors(self, w1, w2, mixing_ratio):
        """
        Given two style vectors, mixes them based on the mixing_ratio.

        Param: w1: A style vector. 
        Param: w2: Another style vector.
        Param: mixing_ratio :A mixing ratio used to combine the two style vectors.
        Returns: A mixed style vector of w1 and w2.
        REF: This function was inspired by code generated by ChatGPT-4o
        REF: Based on the following prompt: How can I implement mixing regularization into my StyleGAN model?
        """
        return w1 * (1 - mixing_ratio) + w2 * mixing_ratio


class GenLayer(nn.Module):
    """
    Within the StyleGAN model, the generator will need multiple layers to upscale the 
    initial feature map size into the desired resolution of the final output image. Each layer
    also consists of a series of convolutions, noise injections and AdaIN passes, that are 
    ultimately used to generate desired feature maps within the generator. 
    
    REF: This class was inspired by code generated by ChatGPT-4o via the following prompt:
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
        Returns: x: An altered feature map, based on one GenLayer pass.
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
    
    
class Discriminator(nn.Module):
    """
    Within the StyleGAN model, the Discriminator is used to classify an input image as either being 
    real or fake. This is done by taking in an input image and applying a series of DiscLayers
    to the image to reduce it down to a tensor of size [batch_size, 1] which will contain scalar
    values indicating the probability that the input image is fake or not. An extension has also been 
    made to this Discriminator to allow it to also generate an output for what class it thinks an image 
    realtes to, either AD or CN. And Progressive GAN architecture has been utilised to train on incrementally
    increasing sizes of images.

    REF: This class was inspired by code generated by ChatGPT-4o via the following two prompts:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    REF: Prompt: How can I use nn.Embedding to embed AD and CN labels into the discriminator of a StyleGAN?
    REF: This class was also inspired by code geenrated by ChatGPT-o1-preview via the following prompts:
    REF: Prompt: How should I write my discriminator to output both a rating for fake/real and a rating for class 
    REF: AD or CN while embedding the class information into the discriminator as well?
    REF: Prompt: How could I implement a method like progressive growing into the Discriminator?
    REF: It was also inspired by the following website: 
    REF: https://blog.paperspace.com/implementation-stylegan-from-scratch/ 
    """

    def __init__(self):
        """
        An instance of the Discriminator class. Initialisaes all requirements needed
        for label embedding, progressive discriminator and class classifications.
        """
        super(Discriminator, self).__init__()

        # Stores multiple from_gray and DiscLayers depending on image resolution
        self.from_gray = nn.ModuleList()
        self.disc_layers = nn.ModuleList()

        # Generate the from_gray and DiscLayer lists
        for i in range(len(hp.DISC_FACTORS) - 1):

            if i == 0:
                in_channels = int(hp.DISC_FACTORS[i])
            else:
                # Determine the feature map sizes inputs and output
                in_channels = int(hp.DISC_FEATURE_SIZE * hp.DISC_FACTORS[i])

            out_channels = int(hp.DISC_FEATURE_SIZE * hp.DISC_FACTORS[i + 1])

            # Generate discriminator layers based on the inputs and outputs
            self.disc_layers.append(DiscLayer(in_channels, out_channels))

            # Create feature map extraction layers for grayscale images
            self.from_gray.append(nn.Conv2d(hp.NUM_CHANNELS, in_channels, kernel_size=1))

        # The last feature map size output of the DiscLayers
        last_feature_size = int(hp.DISC_FEATURE_SIZE * hp.DISC_FACTORS[-1])

        # Final convolutional layer of the feature map processing
        self.final_conv = nn.Conv2d(last_feature_size, last_feature_size, kernel_size=3, padding=1)
        self.leaky = nn.LeakyReLU(negative_slope=hp.LRELU_SLOPE_ANGLE, inplace=True)

        # The amount of features present after flattening the feature map and embedding labels for a 4x4 resolution
        self.feature_size = last_feature_size * 4 * 4

        # Used to embed the AD and CN labels into the feature map
        self.label_embedding = torch.nn.Embedding(num_embeddings=hp.LABEL_DIMENSIONS, embedding_dim=self.feature_size)

        # Output layer for real/fake image classification
        self.real_fake_layer = nn.Linear(self.feature_size, 1)

        # Output layer for class label classificaton
        self.class_layer = nn.Linear(self.feature_size, hp.LABEL_DIMENSIONS)

        # Used to normalise the output to a value representative of a real or fake image 
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, labels=None, depth=0, alpha=1.0):
        """
        Represents a forward pass of the Discriminator, which takes in a given input image 'x'
        and outputs scalar values determining if the input image is real or fake. If labels are passed
        for either the AD or CN classes, then the labels will be embedded into the input image via label projection. 
        This function also handles various resolutions of x, which is used to implement a progressive discriminator.

        Param: x: An input image of a potential range of resolutions (8x8 <-> 256x256), from the ADNI dataset.
        Param: labels: The AD and CN labels that will be embedded into the input image.
        Param: depth: The current depth of the network which indicates the current resolutions of the input image.
        Param: alpha: A Fade-In alpha value beteen 0 and 1.
        Returns: (real/fake prediction, class prediction, feature map)
        """
        # Current depth at base resolution of (8x8), no fade-in required
        if depth == 0:

            # Assign smallest feature map size to lowest resolution
            output = self.from_gray[-1](x)
            output = self.disc_layers[-1](output)

        # Not at the base resolution
        else:

            # Extract feature map from new from_gray layer
            output_new = self.from_gray[-(depth + 1)](x)
            output_new = self.disc_layers[-(depth + 1)](output_new)

            # Extract feature map from old from_gray layer
            output_old = F.avg_pool2d(x, kernel_size=2) # Downscale to old resolution
            output_old = self.from_gray[-depth](output_old)

            # Perform Fade-In between new and old layers
            output = (alpha * output_new) + ((1 - alpha) * output_old)

            # Perform the rest of the feature map extraction 
            for layer in self.disc_layers[-depth:]:
                output = layer(output)

        # Final convolution layer
        output = self.leaky(self.final_conv(output))

        # Flatten the features to be used in a linear layer
        output = output.view(output.size(0), -1) 

        # Compute the real/fake logits 
        real_fake_logits = self.real_fake_layer(output)
        
        # Perform Discriminator label projection
        if labels is not None:

            # Generate a vector representation of each label
            embedded_labels = self.label_embedding(labels)

            # Compute the dot product between the feature map and labels
            projection = torch.sum(output * embedded_labels, dim=1, keepdim=True)

            # Add dot product computation to real/fake logits
            real_fake_logits += projection

        # Output for real/fake images
        real_fake_output = self.sigmoid(real_fake_logits)

        # Output class prediction (logits)
        class_ouput = self.class_layer(output)

        return real_fake_output, class_ouput, output
    

class DiscLayer(nn.Module):
    """
    Within the StyleGAN model, the discriminator will need a series of layers to downscale the
    256x256 input image into a single real number. The DiscLayer represents one of these layers, 
    and uses two convolutions for feature extraction and downsamples by average pooling the 
    feature map, effectively reducing the feature map resolution in half with each layer.

    REF: This class was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you show me code for a StyleGan model and break down each section so that I can understand it?
    """

    def __init__(self, in_channels, out_channels):
        """
        An instance of the DiscLayer class. Each layer has two convolutions for feature extraction and
        an average pool function to downscale the feature map.

        Param: in_channels: The number of channels for the input feature map, x. 
        Param: out_channels: The number of channels for the output feature map of this layer
        """
        super(DiscLayer, self).__init__()

        # Model both convolutions needed with a 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Need to perform a non-linear activation after each convolution
        self.lrelu = nn.LeakyReLU(hp.LRELU_SLOPE_ANGLE, inplace=True)

        # Downscale the feature map by a factor of 2
        self.downscale = nn.AvgPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        """
        Represents one froward pass of a discriminator layer. The input feature map x has 
        it's features extracted via two convolutional layers and is then downscaled by a 
        factor of 2.

        Params: x: The input feature map. 
        Returns: The feature map downscaled with feature extraction.
        """
        # Apply feature extractions
        x = self.lrelu(self.conv1(x))
        #x = self.lrelu(self.conv2(x))

        # Downscale the feature map
        x = self.downscale(x)

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


def weights_init(model):
    """
    Takes an initialised model and reinitialises it by randomly sampling a normal
    distribution with mean=0 and stdev=0.02. 

    Param: model: A model that was just initialised.
    REF: This code was inspired by code generated by ChatGPT-o1-preview via the following prompt.
    REF: Prompt: Should I add a weight initialization for the creation of my model to intialise all 
    REF: convolution weights to a normal distribution? 
    """
    classname = model.__class__.__name__
    
    # Initialise all convolution layer weights
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        if model.bias is not None:
            nn.init.constant_(model.bias.data, 0)

    # Initialise all linear layer weights
    elif classname.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def compute_gradient_penalty(disc, real_images, fake_images, device):
    """
    Used to compute the gradient penalty that is normally utilised in WGAN-GP
    loss. The gradient pentaly output will be utilised to try and produce 
    smoother gradients for the discriminator and try to make training more stable.

    Param: disc: A discriminator model.
    Param: real_images: Real images from the ADNI dataset.
    Param: fake_images: Images produced by the generator.
    Param: device: A Cuda or cpu device.
    Returns: The WGAN-GP gradient penalty loss scaled by hp.LAMBDA.
    REF: This function was inspired by code generated by ChatGPT-4o via the following prompt:
    REF: Prompt: Can you explain what the Gradient Penalty (WGAN-GP) does?
    """
    # Gather real image dimensions
    batch_size, C, H, W = real_images.size()

    # Generate random weights so that the real and fake images can be combined
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_images)
    
    # Generate interpolated images between the real and fake images
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images.detach()
    interpolated.requires_grad_(True)
    
    # Isolate the output of the discriminator based on the interpolated images
    interpolated_pred, _, _ = disc(interpolated)
    
    # Compute the gradients of the discriminator based on the interpolated images 
    gradients = torch.autograd.grad(
        outputs=interpolated_pred,
        inputs=interpolated,
        grad_outputs=torch.ones_like(interpolated_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Compute the norm of the gradients
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Compute the gradient penalty
    gradient_penalty = torch.mean(((gradient_norm - 1) ** 2)) * hp.LAMBDA
    
    return gradient_penalty


class PixelNorm(nn.Module):
    """
    Used to perform pixel normalisation on a vectorised image or latent space vector. The vector
    is normalised relative to it's own mean and variance.

    REF: This class was inspired by the following website:
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

    REF: This class was inspired by code generated by ChatGPT-4o via the following prompt:
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
        self.style_scale = fully_connected(hp.LATENT_SIZE, channels)
        self.style_shift = fully_connected(hp.LATENT_SIZE, channels)

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
