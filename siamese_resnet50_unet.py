import torch
import torch.nn as nn
from torchgeo.models import resnet50, ResNet50_Weights

NUM_CHANNELS = 13

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class SiameseUNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.SENTINEL2_ALL_MOCO)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + NUM_CHANNELS, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(128, 1, kernel_size=1, stride=1)  # Single-channel output

    def forward(self, x1, x2, with_output_feature_map=False):
        pre_pools1 = dict()
        pre_pools1[f"layer_0"] = x1
        x1 = self.input_block(x1)
        pre_pools1[f"layer_1"] = x1
        x1 = self.input_pool(x1)

        pre_pools2 = dict()
        pre_pools2[f"layer_0"] = x2
        x2 = self.input_block(x2)
        pre_pools2[f"layer_1"] = x2
        x2 = self.input_pool(x2)

        for i, block in enumerate(self.down_blocks, 2):
            x1 = block(x1)
            x2 = block(x2)
            if i == (SiameseUNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools1[f"layer_{i}"] = x1
            pre_pools2[f"layer_{i}"] = x2

        x1 = self.bridge(x1)
        x2 = self.bridge(x2)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{SiameseUNetWithResnet50Encoder.DEPTH - 1 - i}"
            x1 = block(x1, pre_pools1[key])
            x2 = block(x2, pre_pools2[key])

        x = torch.cat([x1, x2], dim=1)
        x = self.out(x)
        del pre_pools1
        del pre_pools2
        if with_output_feature_map:
            return x, x1, x2
        else:
            return x

        def forward(self, x1, x2, with_output_feature_map=False):
            pre_pools1 = dict()
            pre_pools1[f"layer_0"] = x1
            x1 = self.input_block(x1)
            pre_pools1[f"layer_1"] = x1
            x1 = self.input_pool(x1)

            pre_pools2 = dict()
            pre_pools2[f"layer_0"] = x2
            x2 = self.input_block(x2)
            pre_pools2[f"layer_1"] = x2
            x2 = self.input_pool(x2)

            for i, block in enumerate(self.down_blocks, 2):
                x1 = block(x1)
                x2 = block(x2)
                if i == (SiameseUNetWithResnet50Encoder.DEPTH - 1):
                    continue
                pre_pools1[f"layer_{i}"] = x1
                pre_pools2[f"layer_{i}"] = x2

            x1 = self.bridge(x1)
            x2 = self.bridge(x2)

            for i, block in enumerate(self.up_blocks, 1):
                key = f"layer_{SiameseUNetWithResnet50Encoder.DEPTH - 1 - i}"
                x1 = block(x1, pre_pools1[key])
                x2 = block(x2, pre_pools2[key])

            x = torch.cat([x1, x2], dim=1)
            x = self.out(x)
            del pre_pools1
            del pre_pools2
            if with_output_feature_map:
                return x, x1, x2
            else:
                return x
