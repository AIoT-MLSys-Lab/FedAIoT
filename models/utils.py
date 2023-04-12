from torch import nn


class ResNetCustomNorm(nn.Module):
    def __init__(self, num_channels):
        super(ResNetCustomNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=2,
                                 num_channels=num_channels,)

    def forward(self, x):
        x = self.norm(x)
        return x
