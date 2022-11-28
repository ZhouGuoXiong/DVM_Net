
import torch
import torch.nn as nn


from ShuffleAttention import ShuffleAttention
from epsanet import PSAModule



class SM(nn.Module):
    def __init__(self, in_channels):
        super(SM, self).__init__()
        self.in_channel = in_channels



        self.depth_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            groups=in_channels
        )
        self.point_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

        self.depth_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            stride=2,
            padding=1,
            groups=in_channels
        )
        self.point_conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=1,
            groups=1
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=5, stride=2, groups=4),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=2),
            nn.BatchNorm2d(in_channels, momentum=0.1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2, groups=4),
            # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=7, stride=2),
            nn.BatchNorm2d(in_channels, 0.1),
            nn.ReLU()
        )


        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels, 0.1),
            nn.ReLU()
        )
        self.BN = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.Re = nn.ReLU(in_channels)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.c8 = OctConv2d_v1(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1)
        self.psa = PSAModule(in_channels, in_channels)

        self.at = ShuffleAttention(channel=in_channels)
    def forward(self, x):

        x1 = x
        x2 = x
        x3 = x

        x1 = self.depth_conv1(x1)
        x1 = self.point_conv1(x1)
        x1 = self.BN(x1)
        x1 = self.Re(x1)

        x2 = self.depth_conv2(x2)
        x2 = self.point_conv2(x2)
        x2 = self.BN(x2)
        x2 = self.Re(x2)

        x3 = self.maxpool(x3)

        x = torch.add(x1, x2)
        x = torch.add(x, x3)
        # x = self.CA(x)
        #x = self.CA(x)
        x = self.conv3(x)
        # x = self.c8(x)
        x = self.up(x)
        x = self.at(x)
        # xm_batchsize, C, height, width = x.size()


        return x








