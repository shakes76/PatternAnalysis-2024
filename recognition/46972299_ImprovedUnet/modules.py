"""
This file contains a pytorch class-based implementation of the Improved 3D Unet. The Network itself is its own class, and each component
has been split up into their own classes for ease of use.

@author Carl Flottmann
"""
import torch
import torch.nn as nn


class Improved3DUnet(nn.Module):
    """
    Contains the implementation of the Improved 3D Unet. Hyperparameters such as the filter base and slope for every
    Leaky ReLU module are available in this class as constants.

    Inherits:
        nn.Module: pytorch abstract module class.
    """
    # hyperparameters
    LEAKY_RELU_SLOPE = 0.02
    FILTER_BASE = 8

    def __init__(self, input_channels: int, num_classes: int) -> None:
        """
        Initialise the Unet object.

        Args:
            input_channels (int): number of input channels for the dataset.
            num_classes (int): number of segmentation classes for the dataset.
        """
        super(Improved3DUnet, self).__init__()
        # === Encoder ===

        # layer 1
        self.encoder1_conv = Conv(input_channels, self.FILTER_BASE)
        self.encoder1_context = Context(
            self.FILTER_BASE, self.FILTER_BASE, self.LEAKY_RELU_SLOPE)
        self.encoder1_sum = Sum(self.FILTER_BASE, self.LEAKY_RELU_SLOPE)

        # layer 2
        self.encoder2_conv = Conv2(self.FILTER_BASE, self.FILTER_BASE * 2)
        self.encoder2_context = Context(
            self.FILTER_BASE * 2, self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)
        self.encoder2_sum = Sum(self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)

        # layer 3
        self.encoder3_conv = Conv2(self.FILTER_BASE * 2, self.FILTER_BASE * 4)
        self.encoder3_context = Context(
            self.FILTER_BASE * 4, self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)
        self.encoder3_sum = Sum(self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)

        # layer 4
        self.encoder4_conv = Conv2(self.FILTER_BASE * 4, self.FILTER_BASE * 8)
        self.encoder4_context = Context(
            self.FILTER_BASE * 8, self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)
        self.encoder4_sum = Sum(self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)

        # layer 5
        self.encoder5_conv = Conv2(self.FILTER_BASE * 8, self.FILTER_BASE * 16)
        self.encoder5_context = Context(
            self.FILTER_BASE * 16, self.FILTER_BASE * 16, self.LEAKY_RELU_SLOPE)
        self.encoder5_sum = Sum(self.FILTER_BASE * 16, self.LEAKY_RELU_SLOPE)

        # === Decoder ===
        self.decoder_concat = Concat()
        self.decoder_upscale = Upscale()

        # layer 4
        self.decoder4_upsample = Upsample(
            self.FILTER_BASE * 16, self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)
        self.decoder4_localise = Localise(
            self.FILTER_BASE * 16, self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)

        # layer 3
        self.decoder3_upsample = Upsample(
            self.FILTER_BASE * 8, self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)
        self.decoder3_localise = Localise(
            self.FILTER_BASE * 8, self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)
        self.decoder3_segment = Segment(self.FILTER_BASE * 4, num_classes)

        # layer 2
        self.decoder2_upsample = Upsample(
            self.FILTER_BASE * 4, self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)
        self.decoder2_localise = Localise(
            self.FILTER_BASE * 4, self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)
        self.decoder2_segment = Segment(self.FILTER_BASE * 2, num_classes)
        self.decoder2_sum = Sum(num_classes, self.LEAKY_RELU_SLOPE)

        # layer 1
        self.decoder1_upsample = Upsample(
            self.FILTER_BASE * 2, self.FILTER_BASE, self.LEAKY_RELU_SLOPE)
        self.decoder1_conv = Conv(self.FILTER_BASE * 2, self.FILTER_BASE)
        self.decoder1_segment = Segment(self.FILTER_BASE, num_classes)
        self.decoder1_sum = Sum(num_classes, self.LEAKY_RELU_SLOPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the model and retrieve the output. Applyies the softmax before
        returning the model output.

        Args:
            x (torch.Tensor): data to pass through the model.

        Returns:
            torch.Tensor: the model output after applying softmax.
        """
        # === Encoding ===

        # layer 1
        encoder1_conv = self.encoder1_conv(x)
        encoder1_context = self.encoder1_context(encoder1_conv)
        encoder1_sum = self.encoder1_sum(encoder1_conv, encoder1_context)

        # layer 2
        encoder2_conv = self.encoder2_conv(encoder1_sum)
        encoder2_context = self.encoder2_context(encoder2_conv)
        encoder2_sum = self.encoder2_sum(encoder2_conv, encoder2_context)

        # layer 3
        encoder3_conv = self.encoder3_conv(encoder2_sum)
        encoder3_context = self.encoder3_context(encoder3_conv)
        encoder3_sum = self.encoder3_sum(encoder3_conv, encoder3_context)

        # layer 4
        encoder4_conv = self.encoder4_conv(encoder3_sum)
        encoder4_context = self.encoder4_context(encoder4_conv)
        encoder4_sum = self.encoder4_sum(encoder4_conv, encoder4_context)

        # layer 5
        encoder5_conv = self.encoder5_conv(encoder4_sum)
        encoder5_context = self.encoder5_context(encoder5_conv)
        encoder5_sum = self.encoder5_sum(encoder5_conv, encoder5_context)

        # === Decoding ===

        # layer 4
        decoder4_upsample = self.decoder4_upsample(encoder5_sum)
        decoder4_concat = self.decoder_concat(decoder4_upsample, encoder4_sum)
        decoder4_localise = self.decoder4_localise(decoder4_concat)

        # layer 3
        decoder3_upsample = self.decoder3_upsample(decoder4_localise)
        decoder3_concat = self.decoder_concat(decoder3_upsample, encoder3_sum)
        decoder3_localise = self.decoder3_localise(decoder3_concat)
        decoder3_segment = self.decoder3_segment(decoder3_localise)
        decoder3_upscale = self.decoder_upscale(decoder3_segment)

        # layer 2
        decoder2_upsample = self.decoder2_upsample(decoder3_localise)
        decoder2_concat = self.decoder_concat(decoder2_upsample, encoder2_sum)
        decoder2_localise = self.decoder2_localise(decoder2_concat)
        decoder2_segment = self.decoder2_segment(decoder2_localise)
        decoder2_sum = self.decoder2_sum(decoder2_segment, decoder3_upscale)
        decoder2_upscale = self.decoder_upscale(decoder2_sum)

        # layer 1
        decoder1_upsample = self.decoder1_upsample(decoder2_localise)
        decoder1_concat = self.decoder_concat(decoder1_upsample, encoder1_sum)
        decoder1_conv = self.decoder1_conv(decoder1_concat)
        decoder1_segment = self.decoder1_segment(decoder1_conv)
        decoder1_sum = self.decoder1_sum(decoder1_segment, decoder2_upscale)

        return torch.softmax(decoder1_sum, dim=1)


class Conv(nn.Module):
    """
    Single-strided 3x3x3 convolution module. Used at the start of the Unet. Applied instance normalisation
    after the convolution operation.

    Inherits:
        nn.Module: pytorch abstract module class.
    """

    def __init__(self, input_filters: int, output_filters: int) -> None:
        """
        Initialise the single-strided 3x3x3 convolution module.

        Args:
            input_filters (int): number of input filters to the module.
            output_filters (int): number of output filters to the module.
        """
        super(Conv, self).__init__()

        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output after applying instance normalisation.
        """
        return self.process(x)


class Conv2(nn.Module):
    """
    Double-strided 3x3x3 convolution module. Used in the Unet encoder. Applied instance normalisation
    after the convolution operation.

    Inherits:
        nn.Module: pytorch abstract module class.
    """

    def __init__(self, input_filters: int, output_filters: int) -> None:
        """
        Initialise the double-strided 3x3x3 convolution module.

        Args:
            input_filters (int): number of input filters to the module.
            output_filters (int): number of output filters to the module.
        """
        super(Conv2, self).__init__()

        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output after applying instance normalisation.
        """
        return self.process(x)


class Context(nn.Module):
    """
    Context module used in the Unet encoder. Applies instance normalisation after the final convolution before the output.
    Contains, in sequential order, a 3x3x3 convolution, instance normalisation, leaky ReLU, dropout, 3x3x3 convolution,
    and instance normalisation. The dropout probability hyperparameter is available as a constant.

    Inherits:
        nn.Module: pytorch abstract module class.
    """
    # hyperparameters
    DROPOUT_PROB = 0.3

    def __init__(self, input_filters: int, output_filters: int, slope: float) -> None:
        """
        Initialise the context module.

        Args:
            input_filters (int): number of input filters to the module.
            output_filters (int): number of output filters to the module.
            slope (float): the negative slope for the Leaky ReLU component.
        """
        super(Context, self).__init__()

        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope),
            nn.Dropout(self.DROPOUT_PROB),
            nn.Conv3d(output_filters, output_filters, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output after applying instance normalisation.
        """
        return self.process(x)


class Sum(nn.Module):
    """
    Summation module that performs an element-wise sum of the two inputs, and applied instance normalisation
    and activation using a Leaky ReLU modue.

    Inherits:
        nn.Module: pytorch abstract module class.
    """

    def __init__(self, output_filters: int, slope: float) -> None:
        """
        Initialise the summation module.

        Args:
            output_filters (int): number of output filters to the module.
            slope (float): the negative slope for the Leaky ReLU component.
        """
        super(Sum, self).__init__()

        self.process = nn.Sequential(
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope)
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x1 (torch.Tensor): data to pass through the module.
            x2 (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output.
        """
        return self.process(x1 + x2)


class Upsample(nn.Module):
    """
    Upsampling module used in the Unet decoder. Applies activation to the output using a Leaky ReLU. The
    scale factor and mode are available as hyperparameter constants. This module uses a 3x3x3 convolution
    with instance normalisation and a leaky ReLU.

    Inherits:
        nn.Module: pytorch abstract module class.
    """
    # hyperparameters
    SCALE_FACTOR = 2
    MODE = "nearest"

    def __init__(self, input_filters: int, output_filters: int, slope: float) -> None:
        """
        Initialise the upsampling module.

        Args:
            input_filters (int): number of input filters to the module.
            output_filters (int): number of output filters to the module.
            slope (float): the negative slope for the Leaky ReLU component.
        """
        super(Upsample, self).__init__()

        self.input_filters = input_filters
        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output.
        """
        out = nn.functional.interpolate(
            x, scale_factor=self.SCALE_FACTOR, mode=self.MODE)
        return self.process(out)


class Localise(nn.Module):
    """
    Localisation module used in the Unet decoder. Applies activation to the output using a Leaky ReLU. This module
    uses a 3x3x3 convolution followed by instance normalisation and a Leaky ReLU, followed by a 1x1x1 convolution,
    with instance normalisation and a leaky ReLU.

    Inherits:
        nn.Module: pytorch abstract module class.
    """

    def __init__(self, input_filters: int, output_filters: int, slope: float) -> None:
        """
        Initialise the localisation module.

        Args:
            input_filters (int): number of input filters to the module.
            output_filters (int): number of output filters to the module.
            slope (float): the negative slope for the Leaky ReLU component.
        """
        super(Localise, self).__init__()

        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope),
            nn.Conv3d(output_filters, output_filters, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output.
        """
        return self.process(x)


class Segment(nn.Module):
    """
    Segmentation module used in the Unet decoder. Uses a 1x1x1 convolution for the segmentation to create an output
    the size of the number of classes.

    Inherits:
        nn.Module: pytorch abstract module class.
    """

    def __init__(self, input_filters: int, num_classes: int) -> None:
        """
        Initialise the segmentation module.

        Args:
            input_filters (int): number of input filters to the module.
            num_classes (int): number of segmentation classes in the data.
        """
        super(Segment, self).__init__()

        self.process = nn.Conv3d(
            input_filters, num_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output.
        """
        return self.process(x)


class Upscale(nn.Module):
    """
    Upscale module used in the Unet decoder. Performs an interpolation of the input.
    The scale factor and mode are available as hyperparameter constants.

    Inherits:
        nn.Module: pytorch abstract module class.
    """
    # hyperparameters
    SCALE_FACTOR = 2
    MODE = "nearest"

    def __init__(self) -> None:
        """
        Initialise the upscaling module.
        """
        super(Upscale, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output.
        """
        return nn.functional.interpolate(x, scale_factor=self.SCALE_FACTOR, mode=self.MODE)


class Concat(nn.Module):
    """
    Concatenation module used in the Unet decoder.

    Inherits:
        nn.Module: pytorch abstract module class.
    """
    # hyperparameters
    DIMENSION = 1  # accross number of channels

    def __init__(self) -> None:
        """
        Initialise the concatenation module.
        """
        super(Concat, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass through the module.

        Args:
            x1 (torch.Tensor): data to pass through the module.
            x2 (torch.Tensor): data to pass through the module.

        Returns:
            torch.Tensor: the module output.
        """
        return torch.cat((x1, x2), dim=self.DIMENSION)
