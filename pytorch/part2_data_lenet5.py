import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_unit = nn.Sequential(nn.Linear(32 * 3 * 3, 120),
                                     nn.ReLU(),
                                     nn.Linear(120, 10), )
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tmp)
        # print('conv out:', out.shape)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size, -1)
        logits = self.fc_unit(x)
        return logits


def main():
    batch_size = 400
    cifar_train = datasets.CIFAR10('cifar10', train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    cifar_train = DataLoader(cifar_train, batch_size=batch_size, shuffle=True)
    cifar_test = datasets.CIFAR10('cifar10', train=False, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    cifar_test = DataLoader(cifar_test, batch_size=batch_size, shuffle=True)
    x, label = iter(cifar_train).next()
    print('x:', x.shape, 'label:', label.shape)
    device = torch.device('cuda')
    model = Lenet5().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    print(model)
    for epoch in range(50):
        model.train()
        for batch_index, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        # 不保存计算图
        with torch.no_grad():
            total_correct = 0
            total_sum = 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += pred.eq(label).float().sum().item()
                total_sum += x.size(0)
            acc = total_correct / total_sum
            print('epoch:', epoch, 'test acc:', acc)


if __name__ == '__main__':
    main()
