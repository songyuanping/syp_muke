import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # [b,784] => [b,200]
        # u:[b,50]
        # sigma:[b,50]
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 100),
            nn.ReLU(inplace=True),
        )
        # [b,50]=>[b,784]
        self.decoder = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 784),
            nn.Sigmoid(),
        )
        self.criteon = nn.MSELoss()

    def forward(self, x):
        """
        :param x: [b,1,28,28]
        :return:
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        h_ = self.encoder(x)
        # [b,100] => [b,50]
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.randn_like(sigma)
        x_hat = self.decoder(h)
        x_hat = x_hat.view(batch_size, 1, 28, 28)
        kld = 0.5 * torch.sum(torch.pow(mu, 2) +
                              torch.pow(sigma, 2) -
                              torch.log(1e-8 + torch.pow(sigma, 2)) - 1
                              ) / (batch_size * 28 * 28)
        return x_hat, kld
