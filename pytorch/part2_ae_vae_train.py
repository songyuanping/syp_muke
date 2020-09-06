import torch
import visdom
from part2_vae import VAE
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    lr = 1e-3
    epochs = 1000
    batch_size = 128

    mnist_train = datasets.MNIST('mnist_data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test = datasets.MNIST('mnist_data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    mnist_test = DataLoader(mnist_test, batch_size=batch_size)
    x, _ = iter(mnist_train).next()
    # print('x:', x.shape, x)
    device = torch.device('cuda')
    # model=AE().to(device)
    model = VAE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(model)
    viz = visdom.Visdom()

    for epoch in range(epochs):
        for batch_index, (x, _) in enumerate(mnist_train):
            # [b,1,28,28]
            x = x.to(device)
            x_hat, kld = model(x)
            loss = criteon(x_hat, x)
            if kld is not None:
                elbo = -loss - 1.0 * kld
                loss = -elbo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss.item(), 'kld:', kld.item())

        x, _ = iter(mnist_test).next()
        x = x.to(device)
        with torch.no_grad():
            x_hat, kld = model(x)
        # 一行显示八张图片
        viz.images(x, nrow=8, win='x', opts=dict(title='x'))
        viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))


if __name__ == '__main__':
    main()
