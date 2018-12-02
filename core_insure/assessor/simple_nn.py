import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from assessor.base_model import BaseModel
import os


class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(FFNN, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.relu = F.relu
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.linear(x)
        y = self.relu(y)
        y = self.linear2(y)
        return y


class NNModel(BaseModel):
    def __init__(self, config):
        input_size = config.get('input_size', 1)
        output_size = config.get('output_size', 1)
        lr = config.get('lr', 0.001)
        hidden_size = config.get('hidden_size', 100)

        self.loss = nn.SmoothL1Loss()
        self.model = FFNN(input_size, output_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.epochs = config.get('epochs', 100)

    def _torch_var(self, value):
        return Variable(torch.Tensor(value))

    def train(self, x_inputs, y_labels, x_val=None, y_val=None):
        for epoch in range(self.epochs):
            y_pred = self.model(self._torch_var(x_inputs))
            loss = self.loss(y_pred, self._torch_var(y_labels))

            val_loss = 0
            if x_val and y_val:
                y_pred_val = self.model(self._torch_var(x_val))
                val_loss = self.loss(y_pred_val, self._torch_var(y_val))

            print(f'Epoch {epoch}, Train_Loss: {loss}, Val_Loss: {val_loss}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self, x):
        y_pred = self.model(self._torch_var(x))
        formatted_y = y_pred.data.numpy()
        return formatted_y

    def save(self, filepath):
        model_path = os.path.join(filepath, 'nn_model.pth')
        torch.save(self.model.state_dict(), model_path)

    def load(self, filepath):
        model_path = os.path.join(filepath, 'nn_model.pth')
        self.model.load_state_dict(torch.load(model_path))