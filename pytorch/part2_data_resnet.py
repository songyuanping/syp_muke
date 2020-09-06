import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # print('out:',out.shape,'extra:',self.extra(x).shape)
        out = F.relu(self.extra(x) + out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())
        self.blk1 = ResBlock(64, 128, stride=2)
        self.blk2 = ResBlock(128, 256, stride=2)
        self.blk3 = ResBlock(256, 512, stride=2)
        self.blk4 = ResBlock(512, 512, stride=1)
        self.outlayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        # print('after conv:', x.shape)
        # [b,512,h,w]=>[b,512,1,1]
        x = F.adaptive_avg_pool2d(x, [1, 1])
        # print('after pool:', x.shape)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    blk = ResBlock(64, 128, stride=4)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)
    x = torch.randn(2, 3, 32, 32)
    model = ResNet18()
    out = model(x)
    print('resnet18:', out.shape)


if __name__ == '__main__':
    main()
