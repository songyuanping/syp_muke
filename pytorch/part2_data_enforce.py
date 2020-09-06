import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from visdom import Visdom
from torchvision import datasets, transforms

batch_size = 400
epochs = 100
learning_rate = 0.005

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=True, download=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomRotation([90, 270]),
        transforms.Resize([32, 32]),
        transforms.RandomCrop([28, 28]),
        transforms.ToTensor()]
    )), shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./mnist_data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ])), shuffle=True, batch_size=batch_size
)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(nn.Linear(784, 256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 256),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, 10),
                                   )

    def forward(self, input):
        return self.model(input)


device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

viz = Visdom()

viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
viz.line([[0.0, 0.0]], [0.], win='test', opts=dict(title='test loss & acc.', legend=['loss', 'acc.']))
global_step = 0

for epoch in range(epochs):
    net.train()
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        optimizer.zero_grad()
        loss = criteon(logits, target)
        loss.backward()
        optimizer.step()
        global_step += 1
        viz.line([loss.item()], [global_step], win='train_loss', update='append')
        if batch_index % 50 == 0:
            print('Train epoch: {} [{}/{} ({:.2f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                        100. * batch_index / len(train_loader), loss.item())
            )
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.to(device)
        logits = net(data)
        test_loss += criteon(logits, target).item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()
    viz.line([[test_loss, correct / len(test_loader.dataset)]], [global_step], win='test', update='append')
    viz.images(data.view(-1, 1, 28, 28), win='x')
    viz.text(str(pred.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f},Accuracy:{}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
    ))
