import torch
import torch.nn as nn
from torch.nn import Softmax

from High_Frequency_Module import HighFrequencyModule
from diversebranchblock import DiverseBranchBlock


class EGM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EGM, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels


        # up
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, dilation=2),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, dilation=3),
            nn.BatchNorm2d(out_channels, 0.1),
            nn.ReLU()
        )

        #down
        self.conv3 = nn.Sequential(
            DiverseBranchBlock(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),

            #nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU()
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=4),
        #     nn.BatchNorm2d(out_channels, momentum=0.1),

        # )

        self.depth_conv4 = DiverseBranchBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels
        )
        self.point_conv4 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        self.conv5 = nn.Sequential(
           # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=2  ),
            DiverseBranchBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU()

        )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, groups=4),
        #     nn.BatchNorm2d(out_channels, momentum=0.1),
        #
        # )

        self.depth_conv6 = DiverseBranchBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,

            padding=1,
            groups=in_channels
        )
        self.point_conv6 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,

            padding=0,
            groups=1
        )

        self.BN = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.Re = nn.ReLU()


        self.mp = nn.MaxPool2d(2)

        self.softmax = Softmax(dim=-1)

        self.si = nn.Sigmoid()

    def forward(self, x):
        x1 = x

        c1 = self.conv1(x)
        c1 = self.conv2(c1)

        c2 = self.conv3(x1)
        m_batchsize, C, height, width = c2.size()


        c3 = self.depth_conv4(c2)
        c3 = self.point_conv4(c3)
        c3 = torch.add(c2, c3)
        c3 = self.BN(c3)
        c3 = self.Re(c3)

        c3 = self.conv5(c3)
        c4 = self.depth_conv6(c3)
        c4 = self.point_conv6(c4)
        c4 = torch.add(c4, c3)
        c4 = self.BN(c4)

        c = torch.add(c1, c4)
        #c = self.sc(c)
        c = self.si(c)
        return c





