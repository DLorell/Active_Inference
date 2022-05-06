import torch.nn as nn
import torch


class Surprise(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return -torch.log(inputs)
