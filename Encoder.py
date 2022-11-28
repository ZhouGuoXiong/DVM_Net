import torch.nn as nn

from EGM import EGM
from SM import SM


class Encoder(nn.Module):
    def __init__(self, input_channel, input_size):
        super(Encoder, self).__init__()
        bn_momentum = 0.1

        # 卷积
        self._pre_treat_2 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=1, stride=1)
        # layer_1
        self._layer_1 = nn.Sequential(

            EGM(in_channels=64, out_channels=64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # 批一归化，均值为0方差为1
            nn.BatchNorm2d(64, momentum=bn_momentum),

            SM(in_channels=64)





        )
        # skip_connection_1 & down_sample
        # 池化（最大）
        self._down_sample_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_2
        self._layer_2 = nn.Sequential(

            EGM(in_channels=64,out_channels=64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),

            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.PReLU(128),

            SM(in_channels=128)
        )
        # skip_connection_2 & down_sample
        self._down_sample_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_3
        self._layer_3 = nn.Sequential(


            EGM(in_channels=128, out_channels=128),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.PReLU(256),

            SM(in_channels=256)

        )
        # skip_connection_3 & down_sample
        self._down_sample_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_4
        self._layer_4 = nn.Sequential(

            EGM(in_channels=256, out_channels=256),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.PReLU(512),


            SM(in_channels=512)

        )
        # skip_connection_4 & down_sample
        self._down_sample_4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # layer_5
        self._layer_5 = nn.Sequential(

            EGM(in_channels=512, out_channels=512),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024, momentum=bn_momentum),
            nn.PReLU(1024),

            SM(in_channels=1024)

        )



    def forward(self, x):


        x = self._pre_treat_2(x)

        # layer 1
        x = self._layer_1(x)

        skip_1 = x



        x = self._down_sample_1(x)
        # layer 2
        x = self._layer_2(x)
      
        skip_2 = x


        x = self._down_sample_2(x)
        # layer 3
        x = self._layer_3(x)
        # x = self.dem3(x)
        skip_3 = x


        x = self._down_sample_3(x)
        # layer 4
        x = self._layer_4(x)


        # x = self.dem4(x)
        skip_4 = x


        x = self._down_sample_4(x)
        x = self._layer_5(x)

        # x = self.dem5(x)
        return x, skip_1, skip_2, skip_3, skip_4
