import torch
from torch import nn


class convolution1(nn.Module):
    def __init__(self):
        super(convolution1, self).__init__()

        self.conv_block = nn.Sequential(
            # input the desired layers here
            nn.Identity()
        )

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.conv_block(x)
