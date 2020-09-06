from torch import nn


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        # [b,784]=>[b,100]
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
        # [b,100]=>[b,784]
        self.decoder = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x: [b,1,28,28]
        :return:
        """
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, 1, 28, 28)
        return x, None
