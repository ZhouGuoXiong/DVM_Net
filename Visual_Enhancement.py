import torch
import torch.nn as nn
from torch.nn import functional as F, Softmax

from ECA import ECABlock
from diversebranchblock import DiverseBranchBlock


class Visual_Enhance(nn.Module):
    def __init__(self, in_channels):
        super(Visual_Enhance, self).__init__()
        self.in_channel = in_channels


        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 16, kernel_size=1, stride=2),
            nn.BatchNorm2d(in_channels // 16, momentum=0.1),
            nn.ReLU(in_channels // 16)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 4, kernel_size=1, stride=2),
            nn.BatchNorm2d(in_channels // 4, 0.1),
            nn.PReLU(in_channels // 4)
        )
        self.conv3 = nn.Sequential(
           # DiverseBranchBlock(in_channels=in_channels // 4, out_channels=in_channels // 8, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=in_channels // 4, out_channels=in_channels // 8, kernel_size=3, stride=2),
            nn.BatchNorm2d(in_channels // 8, momentum=0.1),
            nn.PReLU(in_channels // 8)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels // 8, out_channels=in_channels // 16, kernel_size=3, stride=2),
            #DiverseBranchBlock(in_channels=in_channels // 8, out_channels=in_channels // 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels // 16, momentum=0.1),
            nn.PReLU(in_channels // 16)
        )

        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        )

        self.rblock1 = ResidualBlock(in_channels= in_channels, out_channels=in_channels)

        self.mp = nn.MaxPool2d(4)

        self.softmax = Softmax(dim=-1)
        self.sk = ECABlock(channels=in_channels)

        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = x
        c2 = self.conv2(x)
        c2 = self.conv3(c2)
        c2 = self.conv4(c2)
        m_batchsize, C, height, width = c2.size()

        c1 = self.conv1(x1)
     #   c1 = self.mp(c1)
        c1 = c1.view(m_batchsize, C, height, width)

        x = torch.mul(c1, c2)



        x = self.up(x)
        x = self.rblock1(x)
        # x = F.relu(self.conv1(x))
        # x = self.rblock1(x)
        # x = F.relu(self.conv2(x))
        # x = self.rblock2(x)
        # x = self.sk(x)
        x = self.Sigmoid(x)

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels

        self.conv1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)
