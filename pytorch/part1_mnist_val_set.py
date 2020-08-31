import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

batch_size = 250
learning_rate = 0.01
epochs = 100

train_db = datasets.MNIST('./mnist_data', train=True, download=True, transform=
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]))
train_loader = torch.utils.data.DataLoader(train_db, shuffle=True, batch_size=batch_size)
test_db = datasets.MNIST('./mnist_data', train=False, download=True, transform=
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3781,))
]))
test_loader = torch.utils.data.DataLoader(test_db, shuffle=True, batch_size=batch_size)

print('train: ', len(train_db), 'test: ', len(test_db))
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
print('db1: ', len(train_db), 'db2: ', len(val_db))
train_loader = torch.utils.data.DataLoader(train_db, shuffle=True, batch_size=batch_size)
val_loader = torch.utils.data.DataLoader(val_db, shuffle=True, batch_size=batch_size)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 256),
            # nn.Dropout(0.5),
            nn.SELU(inplace=True),
            nn.Linear(256, 256),
            nn.SELU(inplace=True),
            nn.Linear(256, 64),
            # nn.Dropout(0.5),
            nn.SELU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model(x)


device = torch.device('cuda:0')
net = MLP().to(device)
# momentum参数在SGD这种比较原始的优化器中需要设置，在Adam等优化器中则不需要，weight_decay表示使用l2范式
optimizer = optim.SGD(net.parameters(), momentum=0.9, lr=learning_rate, weight_decay=0.01)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_index, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        data, target = data.to(device), target.to(device)

        logits = net(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_index * len(data), len(train_loader.dataset),
                       100. * batch_index / len(train_loader), loss.item()
            ))
    test_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.view(-1, 28 * 28)

        data, target = data.to(device), target.to(device)

        logits = net(data)
        test_loss += criteon(logits, target).item()
        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * correct / len(val_loader.dataset)
    ))

test_loss = 0
correct = 0
for data, target in test_loader:
    data = data.view(-1, 28 * 28)

    data, target = data.to(device), target.to(device)

    logits = net(data)
    test_loss += criteon(logits, target).item()
    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()
test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
))
