import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms


# from torchvision.models import resnet18

class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        """
        :param ch_in: 输入通道数
        :param ch_out: 输出通道数
        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.extra = nn.Sequential()
        if ch_out != ch_in:
            # [b,ch_in,h,w] => [b,ch_out,h,w]
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(ch_out),
            )

    def forward(self, x):
        """
        :param x: [b,ch,h,w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # short cut.
        # extra module: [b, ch_in, h, w] => [b, ch_out, h, w]
        # element-wise add:
        out = F.selu(self.extra(x) + out)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16)
        )
        self.block1 = ResBlock(16, 16)
        self.block2 = ResBlock(16, 32)
        self.outlayer = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        """
        :param x: 输入图片
        :return:
        """
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


def main():
    batch_size = 32
    cifar_train = datasets.CIFAR10('cifar10', train=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]), download=True)
    cifar_train = DataLoader(cifar_train, shuffle=True, batch_size=batch_size)
    cifar_test = datasets.CIFAR10('cifar10', train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]), download=True)
    cifar_test = DataLoader(cifar_test, shuffle=True, batch_size=batch_size)
    x, label = iter(cifar_train).next()
    print('x: ', x.shape, 'label: ', label.shape)
    device = torch.device('cuda')
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)
    for epoch in range(1000):
        model.train()
        loss = 0
        for batch_index, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch: ', epoch, 'loss:', loss.item())
        # if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_num = 0
            for x, label in cifar_test:
                # [b, 3, 32, 32]
                # [b]
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred, label).float().sum().item()
                total_num += x.size(0)
            acc = total_correct / total_num
            print('epoch:', epoch, 'acc: ', acc)


if __name__ == '__main__':
    main()
