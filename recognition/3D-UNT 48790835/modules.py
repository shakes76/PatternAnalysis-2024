import torch
import torch.nn as nn

# Hyper parameters
negativeSlope = 10 ** -2
pDrop = 0.3


class Improved3DUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[16, 32, 64, 128, 256]):
        super(Improved3DUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.features_reversed = list(reversed(features))

        self.lrelu = nn.LeakyReLU(negative_slope=negativeSlope)
        self.dropout = nn.Dropout3d(p=pDrop)
        self.upScale = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.softmax = nn.Softmax(dim=1)

        self.convs_context = nn.ModuleList()
        self.contexts = nn.ModuleList()
        self.norm_relus_context = nn.ModuleList()
        self.convs_norm_relu_local = nn.ModuleList()
        self.convs_local = nn.ModuleList()
        self.upSamples = nn.ModuleList()

        for i in range(5):
            if i == 0:
                self.convs_context.append(
                    nn.Conv3d(self.in_channels, self.features[i], kernel_size=3, stride=1, padding=1, bias=False))
                self.convs_local.append(
                    nn.Conv3d(self.features_reversed[i + 1], self.features_reversed[i + 1], kernel_size=1, stride=1,
                              padding=0, bias=False))
            elif i == 4:
                self.convs_context.append(
                    nn.Conv3d(self.features[i - 1], self.features[i], kernel_size=3, stride=2, padding=1, bias=False))
                self.convs_local.append(
                    nn.Conv3d(self.features_reversed[i - 1], self.out_channels, kernel_size=1, stride=1, padding=0,
                              bias=False))
            else:
                self.convs_context.append(
                    nn.Conv3d(self.features[i - 1], self.features[i], kernel_size=3, stride=2, padding=1, bias=False))
                self.convs_local.append(
                    nn.Conv3d(self.features_reversed[i - 1], self.features_reversed[i], kernel_size=1, stride=1,
                              padding=0, bias=False))

            conv = self.norm_lrelu_conv(features[i], self.features[i])
            self.contexts.append(self.context(conv, conv))
            if i < 4:
                norm_lrelu = self.norm_lrelu(self.features[i])
                self.norm_relus_context.append(norm_lrelu)

        for p in range(4):
            self.convs_norm_relu_local.append(
                self.conv_norm_lrelu(self.features_reversed[p], self.features_reversed[p]))
            self.upSamples.append(self.up_sample(self.features_reversed[p], self.features_reversed[p + 1]))

        self.norm_local0 = nn.InstanceNorm3d(self.features_reversed[1])
        self.deep_segment_2_conv = nn.Conv3d(self.features_reversed[1], self.out_channels, kernel_size=1, stride=1,
                                             padding=0, bias=False)
        self.deep_segment_3_conv = nn.Conv3d(self.features_reversed[2], self.out_channels, kernel_size=1, stride=1,
                                             padding=0, bias=False)

    def up_sample(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            self.lrelu,
            self.upScale,
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            self.lrelu
        )

    def context(self, conv1, conv2):
        return nn.Sequential(
            conv1,
            self.dropout,
            conv2
        )

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            self.lrelu
        )

    def norm_lrelu(self, feat):
        return nn.Sequential(
            nn.InstanceNorm3d(feat),
            self.lrelu
        )

    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        residuals = dict()
        skips = dict()
        out = x

        # Contextualization level 1 to 5
        for i in range(5):
            out = self.convs_context[i](out)
            residuals[i] = out
            out = self.contexts[i](out)
            out += residuals[i]
            if i < 4:
                out = self.norm_relus_context[i](out)
                skips[i] = out

        # Localization level 1
        out = self.upSamples[0](out)
        out = self.convs_local[0](out)
        out = self.norm_local0(out)
        out = self.lrelu(out)

        # Localization level 2-5
        for j in range(4):
            out = torch.cat([out, skips[3 - j]], dim=1)
            out = self.convs_norm_relu_local[j](out)
            if j == 1:
                ds2 = out
            elif j == 2:
                ds3 = out
            if j == 3:
                out = self.convs_local[j + 1](out)
            else:
                out = self.convs_local[j + 1](out)
            if j < 3:
                out = self.upSamples[j + 1](out)

        # Segment layer summation
        ds2_conv = self.deep_segment_2_conv(ds2)
        ds2_conv_upscale = self.upScale(ds2_conv)

        ds3_conv = self.deep_segment_3_conv(ds3)
        ds2_ds3_upscale = ds2_conv_upscale + ds3_conv

        ds2_ds3_upscale_upscale = self.upScale(ds2_ds3_upscale)
        out += ds2_ds3_upscale_upscale

        # Sigmoid Layer
        out = torch.sigmoid(out)

        return out
