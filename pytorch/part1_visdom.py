import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from visdom import Visdom


batch_size = 200
learning_rate = 0.005
epochs = 30

train_loader = torch.utils.data.DataLoader(datasets.MNIST(
    './mnist_data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        # 正则化后图片变成全黑
        # transforms.Normalize((0.1307,), (0.3081,))
    ])), shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(datasets.MNIST(
    './mnist_data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        # 正则化后图片变成全黑
        # transforms.Normalize((0.1307,), (0.3081,))
    ])), shuffle=True, batch_size=batch_size)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.SELU(inplace=True),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = MLP()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()
# 需要先运行 python -m visdom.server
viz = Visdom()
viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss&acc.',
                                                   legend=['loss', 'acc.']))
global_step = 0

for epoch in range(epochs):
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = net(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')
        if batch_index % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()
            ))
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net(data)
        test_loss += criteon(logits, target).item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()
    viz.line([[test_loss, correct / len(test_loader.dataset)]], [global_step],
             win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28), win='image', opts=dict(title='test image'))
    viz.text(str(pred.detach().numpy()), win='pred', opts=dict(title='pred'))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))
