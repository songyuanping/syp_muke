import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim, nn
from part2_data_lenet5 import Lenet5
from part2_data_resnet import ResNet18


def main():
    batch_size = 200
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
    device = torch.device('cuda:0')
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001)
    print(model)
    for epoch in range(10):
        model.train()
        for batch_index, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
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
