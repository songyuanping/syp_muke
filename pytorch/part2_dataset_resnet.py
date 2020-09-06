import torch
from torch import nn, optim
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # print('out:',out.shape,'self.extra(x):',self.extra(x).shape)

        out = out + self.extra(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_class):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
        )
        self.blk1 = ResBlock(16, 32, stride=2)
        self.blk2 = ResBlock(32, 64, stride=2)
        self.blk3 = ResBlock(64, 128, stride=2)
        self.blk4 = ResBlock(128, 256, stride=2)
        self.outlayer = nn.Linear(256 * 7 * 7, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)

        # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    blk = ResBlock(64, 128,3)
    tmp = torch.rand(2, 64, 224, 224)
    out = blk(tmp)
    print('block:', out.shape)

    model=ResNet18(5)
    tmp=torch.rand(2,3,224,224)
    out=model(tmp)
    print('resnet:',out.shape)

    p=sum(map(lambda p:p.numel(),model.parameters()))
    print('parameters size:',p)

if __name__ == '__main__':
    main()
