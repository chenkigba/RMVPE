import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import N_MELS


def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Center-pad/crop spatial dims of x to match ref (H, W)."""
    _, _, h, w = x.shape
    _, _, hr, wr = ref.shape
    # Pad if smaller
    pad_t = max(0, (hr - h) // 2)
    pad_b = max(0, hr - h - pad_t)
    pad_l = max(0, (wr - w) // 2)
    pad_r = max(0, wr - w - pad_l)
    if pad_t or pad_b or pad_l or pad_r:
        x = F.pad(x, (pad_l, pad_r, pad_t, pad_b))  # (left, right, top, bottom)
        _, _, h, w = x.shape
    # Crop if larger
    crop_t = max(0, (h - hr) // 2)
    crop_l = max(0, (w - wr) // 2)
    if h > hr or w > wr:
        x = x[:, :, crop_t:crop_t + hr, crop_l:crop_l + wr]
    return x


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (7, 7), padding=(3, 3)),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (7, 7), padding=(3, 3)),
            nn.BatchNorm2d(out_channels, momentum=momentum)
        )
        self.skip = nn.Conv2d(in_channels, out_channels, (1, 1))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(out_channels, out_channels // 2, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, (1, 1)),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f)
        s = self.skip(x)
        f = f + s
        a = self.se(f)
        f = a * f
        x = self.relu(f)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 5), padding=(2, 2), momentum=0.01):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, kernel_size, n_blocks, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.en1 = ConvBlock(in_channels, out_channels, (7, 7), (3, 3))
        blocks = []
        for i in range(n_blocks - 1):
            blocks.append(ConvBlockRes(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x1 = self.en1(x)
        x = self.blocks(x1)
        return x1, x


class Inter(nn.Module):
    def __init__(self, kernel_size, num_layers, in_channels):
        super(Inter, self).__init__()
        blocks = []
        for i in range(num_layers):
            blocks.append(ConvBlockRes(in_channels, in_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, kernel_size, num_layers, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.Upsample(scale_factor=(2, 1))
        self.de1 = ConvBlock(in_channels, out_channels, (7, 7), (3, 3))
        blocks = []
        for i in range(num_layers - 1):
            blocks.append(ConvBlockRes(out_channels, out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.up(x)
        x = self.de1(x)
        x = self.blocks(x)
        return x


class DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, out_channels):
        super(DeepUnet, self).__init__()
        self.en1 = Encoder(kernel_size, n_blocks, in_channels, out_channels)
        self.en2 = Encoder(kernel_size, n_blocks, out_channels, out_channels * 2)
        self.en3 = Encoder(kernel_size, n_blocks, out_channels * 2, out_channels * 4)
        self.en4 = Encoder(kernel_size, n_blocks, out_channels * 4, out_channels * 8)
        self.en5 = Encoder(kernel_size, n_blocks, out_channels * 8, out_channels * 16)

        self.inter = Inter(kernel_size, inter_layers, out_channels * 16)

        self.de5 = Decoder(kernel_size, en_de_layers, out_channels * 16, out_channels * 8)
        self.de4 = Decoder(kernel_size, en_de_layers, out_channels * 16, out_channels * 4)
        self.de3 = Decoder(kernel_size, en_de_layers, out_channels * 8, out_channels * 2)
        self.de2 = Decoder(kernel_size, en_de_layers, out_channels * 4, out_channels)
        self.de1 = Decoder(kernel_size, en_de_layers, out_channels * 2, out_channels)

        self.down_s = nn.MaxPool2d((2, 1))

    def forward(self, x):
        x1_1, x1_2 = self.en1(x)
        x = self.down_s(x1_2)
        x2_1, x2_2 = self.en2(x)
        x = self.down_s(x2_2)
        x3_1, x3_2 = self.en3(x)
        x = self.down_s(x3_2)
        x4_1, x4_2 = self.en4(x)
        x = self.down_s(x4_2)
        x5_1, x5_2 = self.en5(x)

        x = self.inter(x5_2)

        x = self.de5(x)
        x = _match_size(x, x4_1)
        x = torch.cat([x, x4_1], dim=1)
        x = self.de4(x)
        x = _match_size(x, x3_1)
        x = torch.cat([x, x3_1], dim=1)
        x = self.de3(x)
        x = _match_size(x, x2_1)
        x = torch.cat([x, x2_1], dim=1)
        x = self.de2(x)
        x = _match_size(x, x1_1)
        x = torch.cat([x, x1_1], dim=1)
        x = self.de1(x)

        return x


class DeepUnet0(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, out_channels):
        super(DeepUnet0, self).__init__()
        self.en1 = Encoder(kernel_size, n_blocks, in_channels, out_channels)
        self.en2 = Encoder(kernel_size, n_blocks, out_channels, out_channels * 2)
        self.en3 = Encoder(kernel_size, n_blocks, out_channels * 2, out_channels * 4)

        self.inter = Inter(kernel_size, inter_layers, out_channels * 4)

        self.de3 = Decoder(kernel_size, en_de_layers, out_channels * 4, out_channels * 2)
        self.de2 = Decoder(kernel_size, en_de_layers, out_channels * 2, out_channels)
        self.de1 = Decoder(kernel_size, en_de_layers, out_channels * 2, out_channels)

        self.down_s = nn.MaxPool2d((2, 1))

    def forward(self, x):
        x1_1, x1_2 = self.en1(x)
        x = self.down_s(x1_2)
        x2_1, x2_2 = self.en2(x)
        x = self.down_s(x2_2)
        x3_1, x3_2 = self.en3(x)

        x = self.inter(x3_2)

        x = self.de3(x)
        x = _match_size(x, x3_1)
        x = torch.cat([x, x3_1], dim=1)
        x = self.de2(x)
        x = _match_size(x, x2_1)
        x = torch.cat([x, x2_1], dim=1)
        x = self.de1(x)

        return x

