import torch
from torch import nn, optim
from torch.autograd import Variable
from core_insure.assessor.base_model import BaseModel


class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FFNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLu()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        y = self.relu(y)
        y = self.linear2(y)
        return y