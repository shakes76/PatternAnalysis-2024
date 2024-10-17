"""
Contains the components of the Unet

@author Carl Flottmann
"""
import torch
import torch.nn as nn

class Improved3DUnet(nn.Module):    
    # hyperparameters
    LEAKY_RELU_SLOPE = 0.02
    FILTER_BASE = 8

    # debug options
    SHAPE_DEBUG = False

    def __init__(self, input_channels: int, num_classes: int) -> None:
        super(Improved3DUnet, self).__init__()
        # === Encoder ===
        
        # layer 1
        self.encoder1_conv = Conv(input_channels, self.FILTER_BASE)
        self.encoder1_context = Context(self.FILTER_BASE, self.FILTER_BASE, self.LEAKY_RELU_SLOPE)
        self.encoder1_sum = Sum(self.FILTER_BASE, self.LEAKY_RELU_SLOPE)
        
        # layer 2
        self.encoder2_conv = Conv2(self.FILTER_BASE, self.FILTER_BASE * 2)
        self.encoder2_context = Context(self.FILTER_BASE * 2, self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)
        self.encoder2_sum = Sum(self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)

        # layer 3
        self.encoder3_conv = Conv2(self.FILTER_BASE * 2, self.FILTER_BASE * 4)
        self.encoder3_context = Context(self.FILTER_BASE * 4, self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)
        self.encoder3_sum = Sum(self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)

        # layer 4
        self.encoder4_conv = Conv2(self.FILTER_BASE * 4, self.FILTER_BASE * 8)
        self.encoder4_context = Context(self.FILTER_BASE * 8, self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)
        self.encoder4_sum = Sum(self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)

        # layer 5
        self.encoder5_conv = Conv2(self.FILTER_BASE * 8, self.FILTER_BASE * 16)
        self.encoder5_context = Context(self.FILTER_BASE * 16, self.FILTER_BASE * 16, self.LEAKY_RELU_SLOPE)
        self.encoder5_sum = Sum(self.FILTER_BASE * 16, self.LEAKY_RELU_SLOPE)

        # === Decoder ===
        self.decoder_concat = Concat()

        # layer 4
        self.decoder4_upsample = Upsample(self.FILTER_BASE * 16, self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)
        self.decoder4_localise = Localise(self.FILTER_BASE * 16, self.FILTER_BASE * 8, self.LEAKY_RELU_SLOPE)

        # layer 3
        self.decoder3_upsample = Upsample(self.FILTER_BASE * 8, self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)
        self.decoder3_localise = Localise(self.FILTER_BASE * 8, self.FILTER_BASE * 4, self.LEAKY_RELU_SLOPE)
        self.decoder3_segment = Segment(self.FILTER_BASE * 4, num_classes)
        self.decoder3_upscale = Upscale(self.FILTER_BASE * 2)

        # layer 2
        self.decoder2_upsample = Upsample(self.FILTER_BASE * 4, self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)
        self.decoder2_localise = Localise(self.FILTER_BASE * 4, self.FILTER_BASE * 2, self.LEAKY_RELU_SLOPE)
        self.decoder2_segment = Segment(self.FILTER_BASE * 2, num_classes)
        self.decoder2_sum = Sum(num_classes, self.LEAKY_RELU_SLOPE)
        self.decoder2_upscale = Upscale(self.FILTER_BASE)

        # layer 1
        self.decoder1_upsample = Upsample(self.FILTER_BASE * 2, self.FILTER_BASE, self.LEAKY_RELU_SLOPE)
        self.decoder1_conv = Conv(self.FILTER_BASE * 2, self.FILTER_BASE)
        self.decoder1_segment = Segment(self.FILTER_BASE, num_classes)
        self.decoder1_sum = Sum(num_classes, self.LEAKY_RELU_SLOPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # === Encoding ===
        
        # layer 1
        encoder1_conv = self.encoder1_conv(x)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder1_conv: {encoder1_conv.shape}")
        encoder1_context = self.encoder1_context(encoder1_conv)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder1_context: {encoder1_context.shape}")
        encoder1_sum = self.encoder1_sum(encoder1_conv, encoder1_context)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder1_sum: {encoder1_sum.shape}")

        # layer 2
        encoder2_conv = self.encoder2_conv(encoder1_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder2_conv: {encoder2_conv.shape}")
        encoder2_context = self.encoder2_context(encoder2_conv)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder2_context: {encoder2_context.shape}")
        encoder2_sum = self.encoder2_sum(encoder2_conv, encoder2_context)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder2_sum: {encoder2_sum.shape}")
        
        # layer 3
        encoder3_conv = self.encoder3_conv(encoder2_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder3_conv: {encoder3_conv.shape}")
        encoder3_context = self.encoder3_context(encoder3_conv)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder3_context: {encoder3_context.shape}")
        encoder3_sum = self.encoder3_sum(encoder3_conv, encoder3_context)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder3_sum: {encoder3_sum.shape}")
        
        # layer 4
        encoder4_conv = self.encoder4_conv(encoder3_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder4_conv: {encoder4_conv.shape}")
        encoder4_context = self.encoder4_context(encoder4_conv)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder4_context: {encoder4_context.shape}")
        encoder4_sum = self.encoder4_sum(encoder4_conv, encoder4_context)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder4_sum: {encoder4_sum.shape}")
        
        # layer 5
        encoder5_conv = self.encoder5_conv(encoder4_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder5_conv: {encoder5_conv.shape}")
        encoder5_context = self.encoder5_context(encoder5_conv)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder5_context: {encoder5_context.shape}")
        encoder5_sum = self.encoder5_sum(encoder5_conv, encoder5_context)
        self.SHAPE_DEBUG and print(f"[MODEL] encoder5_sum: {encoder5_sum.shape}")
        
        # === Decoding ===

        # layer 4
        decoder4_upsample = self.decoder4_upsample(encoder5_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder4_upsample: {decoder4_upsample.shape}")
        decoder4_concat = self.decoder_concat(decoder4_upsample, encoder4_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder4_concat: {decoder4_concat.shape}")
        decoder4_localise = self.decoder4_localise(decoder4_concat)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder4_localise: {decoder4_localise.shape}")
        
        # layer 3
        decoder3_upsample = self.decoder3_upsample(decoder4_localise)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder3_upsample: {decoder3_upsample.shape}")
        decoder3_concat = self.decoder_concat(decoder3_upsample, encoder3_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder3_concat: {decoder3_concat.shape}")
        decoder3_localise = self.decoder3_localise(decoder3_concat)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder3_localise: {decoder3_localise.shape}")
        decoder3_segment = self.decoder3_segment(decoder3_localise)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder3_segment: {decoder3_segment.shape}")
        decoder3_upscale = self.decoder3_upscale(decoder3_segment)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder3_upscale: {decoder3_upscale.shape}")
        
        # layer 2
        decoder2_upsample = self.decoder2_upsample(decoder3_localise)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder2_upsample: {decoder2_upsample.shape}")
        decoder2_concat = self.decoder_concat(decoder2_upsample, encoder2_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder2_concat: {decoder2_concat.shape}")
        decoder2_localise = self.decoder2_localise(decoder2_concat)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder2_localise: {decoder2_localise.shape}")
        decoder2_segment = self.decoder2_segment(decoder2_localise)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder2_segment: {decoder2_segment.shape}")
        decoder2_sum = self.decoder2_sum(decoder2_segment, decoder3_upscale)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder2_sum: {decoder2_sum.shape}")
        decoder2_upscale = self.decoder2_upscale(decoder2_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder2_upscale: {decoder2_upscale.shape}")
        
        # layer 1
        decoder1_upsample = self.decoder1_upsample(decoder2_localise)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder1_upsample: {decoder1_upsample.shape}")
        decoder1_concat = self.decoder_concat(decoder1_upsample, encoder1_sum)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder1_concat: {decoder1_concat.shape}")
        decoder1_conv = self.decoder1_conv(decoder1_concat)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder1_conv: {decoder1_conv.shape}")
        decoder1_segment = self.decoder1_segment(decoder1_conv)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder1_segment: {decoder1_segment.shape}")
        decoder1_sum = self.decoder1_sum(decoder1_segment, decoder2_upscale)
        self.SHAPE_DEBUG and print(f"[MODEL] decoder1_sum: {decoder1_sum.shape}")

        return torch.softmax(decoder1_sum, dim=1)
        


class Conv(nn.Module):
    def __init__(self, input_filters: int, output_filters: int) -> None:
        super(Conv, self).__init__()
        
        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)

class Conv2(nn.Module):
    def __init__(self, input_filters: int, output_filters: int) -> None:
        super(Conv2, self).__init__()
        
        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)

class Context(nn.Module):
    # hyperparameters
    DROPOUT_PROB = 0.3
    
    def __init__(self, input_filters: int, output_filters: int, slope: float, c=None) -> None:
        super(Context, self).__init__()
        
        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope),
            nn.Dropout(self.DROPOUT_PROB),
            nn.Conv3d(output_filters, output_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)


class Sum(nn.Module):
    def __init__(self, output_filters: int, slope: float) -> None:
        super(Sum, self).__init__()
        
        self.process = nn.Sequential(
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope)
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.process(x1 + x2)

class Upsample(nn.Module):
    # hyperparameters
    SCALE_FACTOR = 2
    MODE = "nearest"
    
    def __init__(self, input_filters: int, output_filters: int, slope: float) -> None:
        super(Upsample, self).__init__()
        
        self.input_filters = input_filters
        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = nn.functional.interpolate(x, scale_factor=self.SCALE_FACTOR, mode=self.MODE)
        return self.process(out)

class Localise(nn.Module):
    def __init__(self, input_filters: int, output_filters: int, slope: float) -> None:
        super(Localise, self).__init__()
        
        self.process = nn.Sequential(
            nn.Conv3d(input_filters, output_filters, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope),
            nn.Conv3d(output_filters, output_filters, kernel_size=1, stride=1, padding=0, bias=False),
            nn.InstanceNorm3d(output_filters),
            nn.LeakyReLU(slope)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)

class Segment(nn.Module):
    def __init__(self, input_filters: int, num_classes: int) -> None:
        super(Segment, self).__init__()
        
        self.process = nn.Conv3d(input_filters, num_classes, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.process(x)

class Upscale(nn.Module):
    # hyperparameters
    SCALE_FACTOR = 2
    MODE = "nearest"

    def __init__(self, output_filters: int) -> None:
        super(Upscale, self).__init__()
        self.output_filters = output_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, scale_factor=self.SCALE_FACTOR, mode=self.MODE)

class Concat(nn.Module):
    # hyperparameters
    DIMENSION = 1 # accross number of channels
    
    def __init__(self) -> None:
        super(Concat, self).__init__()
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.cat((x1, x2), dim=self.DIMENSION)
