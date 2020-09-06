import torch
import torchvision
from part1_utils import plot_image, plot_curve, one_hot
from torch import nn
from torch import optim
from torch.nn import functional as F

batch_size = 512

# step 1 load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    'mnist_data/', train=False, download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])), batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'image sample')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x:[b,1,28,28]
        # h1=relu(xw1+b1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.1)
train_loss = []
for epoch in range(10):
    for batch_idx, (x, y) in enumerate(train_loader):
        # x:[b,1,28,28],y[b]
        x = x.view(x.size(0), 28 * 28)
        out = net(x)
        y_onehot = one_hot(y)
        loss = F.mse_loss(out, y_onehot)
        # 将梯度清零
        optimizer.zero_grad()
        # 反向求导
        loss.backward()
        # loss=loss-lr*grad
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % 100 == 0:
            print(epoch, batch_idx, loss.item())
plot_curve(train_loss)

total_correct = 0
for x, y in test_loader:
    x = x.view(x.size(0), -1)
    out = net(x)
    pred = out.argmax(dim=1)
    correct = pred.eq(y).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print('test acc:', acc)
x, y = next(iter(test_loader))
out = net(x.view(x.size(0), -1))
pred = out.argmax(dim=1)
plot_image(x, pred, 'test')
